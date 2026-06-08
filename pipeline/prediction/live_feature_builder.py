from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    CURRENT_FIGHTER_FEATURES_PATH,
    LIVE_CARD_PATH,
    LIVE_FEATURE_AUDIT_PATH,
)
from ufc_feature_engineering import add_v5_engineered_features


class LiveFeatureBuilderError(RuntimeError):
    """Raised when live model features cannot be created."""


@dataclass(frozen=True)
class LiveFeatureBuildResult:
    """Container for live feature builder outputs."""

    live_feature_df: pd.DataFrame
    feature_audit_df: pd.DataFrame


LIVE_CARD_REQUIRED_COLUMNS = [
    "event_id",
    "event_name",
    "fight_id",
    "red_fighter",
    "blue_fighter",
]

RED_ID_CANDIDATES = ["red_fighter_id", "r_id", "red_id"]
BLUE_ID_CANDIDATES = ["blue_fighter_id", "b_id", "blue_id"]
RED_NAME_CANDIDATES = ["red_fighter", "r_name", "red_name", "fighter_1", "fighter_a"]
BLUE_NAME_CANDIDATES = ["blue_fighter", "b_name", "blue_name", "fighter_2", "fighter_b"]
FIGHTER_ID_CANDIDATES = ["fighter_id", "id", "ufcstats_fighter_id"]



def build_live_model_features(
    *,
    feature_columns: list[str],
    live_card_path: str | Path = LIVE_CARD_PATH,
    current_fighter_features_path: str | Path = CURRENT_FIGHTER_FEATURES_PATH,
) -> LiveFeatureBuildResult:
    """Build model-ready live features for Prediction V2.

    MVP behavior:
    1. If the live card already contains all requested model features, return it.
    2. Otherwise, join current fighter features by fighter IDs and assemble common
       V5 differential/engineered features.

    This is intentionally a bridge implementation. Feature Builder V2 should later
    replace this with a shared training/live feature plugin system.
    """

    live_card_path = Path(live_card_path)
    current_fighter_features_path = Path(current_fighter_features_path)

    if not feature_columns:
        raise LiveFeatureBuilderError("feature_columns cannot be empty.")

    if not live_card_path.exists():
        raise LiveFeatureBuilderError(f"Live card not found: {live_card_path}")

    live_card_df = pd.read_parquet(live_card_path)
    live_card_df = _standardize_live_card_columns(live_card_df)
    live_card_df = _deduplicate_columns(live_card_df)
    _validate_live_card(live_card_df)

    # Best-case path: a previous pipeline already produced model-ready features.
    if _has_all_columns(live_card_df, feature_columns):
        live_feature_df = _attach_feature_audit_columns(live_card_df.copy(), feature_columns)
        live_feature_df = _deduplicate_columns(live_feature_df)
        return LiveFeatureBuildResult(
            live_feature_df=live_feature_df,
            feature_audit_df=_build_feature_audit(live_feature_df, feature_columns),
        )

    if not current_fighter_features_path.exists():
        raise LiveFeatureBuilderError(
            "Live card does not contain all model features and current fighter "
            f"features were not found: {current_fighter_features_path}"
        )

    current_features_df = pd.read_parquet(current_fighter_features_path)
    current_features_df = _deduplicate_columns(current_features_df)
    fighter_id_column = _find_first_existing_column(current_features_df, FIGHTER_ID_CANDIDATES)

    if fighter_id_column is None:
        raise LiveFeatureBuilderError(
            "Current fighter features must include one fighter ID column. "
            f"Checked: {FIGHTER_ID_CANDIDATES}"
        )

    joined_df = _join_current_fighter_features(
        live_card_df=live_card_df,
        current_features_df=current_features_df,
        fighter_id_column=fighter_id_column,
    )

    live_feature_df = _assemble_requested_features(joined_df, feature_columns)
    live_feature_df = _attach_feature_audit_columns(live_feature_df, feature_columns)
    live_feature_df = _deduplicate_columns(live_feature_df)
    feature_audit_df = _build_feature_audit(live_feature_df, feature_columns)
    feature_audit_df = _deduplicate_columns(feature_audit_df)

    return LiveFeatureBuildResult(
        live_feature_df=live_feature_df,
        feature_audit_df=feature_audit_df,
    )



def write_live_feature_outputs(
    result: LiveFeatureBuildResult,
    *,
    live_feature_output_path: str | Path,
    feature_audit_path: str | Path = LIVE_FEATURE_AUDIT_PATH,
) -> None:
    """Persist live feature and audit outputs."""

    live_feature_output_path = Path(live_feature_output_path)
    feature_audit_path = Path(feature_audit_path)

    live_feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_audit_path.parent.mkdir(parents=True, exist_ok=True)

    live_feature_df = _deduplicate_columns(result.live_feature_df)
    feature_audit_df = _deduplicate_columns(result.feature_audit_df)

    live_feature_df.to_parquet(live_feature_output_path, index=False)
    feature_audit_df.to_parquet(feature_audit_path, index=False)



def _standardize_live_card_columns(live_card_df: pd.DataFrame) -> pd.DataFrame:
    out = live_card_df.copy()

    red_id_column = _find_first_existing_column(out, RED_ID_CANDIDATES)
    blue_id_column = _find_first_existing_column(out, BLUE_ID_CANDIDATES)

    if red_id_column and red_id_column != "red_fighter_id":
        out["red_fighter_id"] = out[red_id_column]

    if blue_id_column and blue_id_column != "blue_fighter_id":
        out["blue_fighter_id"] = out[blue_id_column]

    out = _fill_display_name_column(
        out,
        target_column="red_fighter",
        candidates=RED_NAME_CANDIDATES,
        id_column="red_fighter_id",
    )
    out = _fill_display_name_column(
        out,
        target_column="blue_fighter",
        candidates=BLUE_NAME_CANDIDATES,
        id_column="blue_fighter_id",
    )

    if "r_id" not in out.columns and "red_fighter_id" in out.columns:
        out["r_id"] = out["red_fighter_id"]

    if "b_id" not in out.columns and "blue_fighter_id" in out.columns:
        out["b_id"] = out["blue_fighter_id"]

    return out



def _fill_display_name_column(
    df: pd.DataFrame,
    *,
    target_column: str,
    candidates: list[str],
    id_column: str,
) -> pd.DataFrame:
    """Fill canonical display-name column from common fallback columns."""

    out = df.copy()

    if target_column not in out.columns:
        out[target_column] = ""

    target = out[target_column].astype("string").fillna("").str.strip()

    for candidate in candidates:
        if candidate == target_column or candidate not in out.columns:
            continue

        candidate_values = out[candidate].astype("string").fillna("").str.strip()
        target = target.mask(target.eq(""), candidate_values)

    # Final fallback keeps the formatter from crashing while still making missing
    # name problems visible in outputs/audits.
    if id_column in out.columns:
        id_values = out[id_column].astype("string").fillna("").str.strip()
        target = target.mask(target.eq(""), "fighter_id:" + id_values)

    out[target_column] = target.astype(str)
    return out



def _validate_live_card(live_card_df: pd.DataFrame) -> None:
    missing = [column for column in LIVE_CARD_REQUIRED_COLUMNS if column not in live_card_df.columns]
    if missing:
        raise LiveFeatureBuilderError(f"Live card missing required columns: {missing}")

    if "red_fighter_id" not in live_card_df.columns or "blue_fighter_id" not in live_card_df.columns:
        raise LiveFeatureBuilderError(
            "Live card must include fighter IDs. Names are display-only; feature joins use IDs."
        )



def _join_current_fighter_features(
    *,
    live_card_df: pd.DataFrame,
    current_features_df: pd.DataFrame,
    fighter_id_column: str,
) -> pd.DataFrame:
    current_features_df = current_features_df.copy()
    current_features_df[fighter_id_column] = current_features_df[fighter_id_column].astype(str)

    red_features = current_features_df.add_prefix("r_state_")
    blue_features = current_features_df.add_prefix("b_state_")

    out = live_card_df.copy()
    out["red_fighter_id"] = out["red_fighter_id"].astype(str)
    out["blue_fighter_id"] = out["blue_fighter_id"].astype(str)

    out = out.merge(
        red_features,
        left_on="red_fighter_id",
        right_on=f"r_state_{fighter_id_column}",
        how="left",
    )
    out = out.merge(
        blue_features,
        left_on="blue_fighter_id",
        right_on=f"b_state_{fighter_id_column}",
        how="left",
    )
    out = _deduplicate_columns(out)

    out["red_feature_match"] = out[f"r_state_{fighter_id_column}"].notna().map({True: "matched_by_id", False: "missing_by_id"})
    out["blue_feature_match"] = out[f"b_state_{fighter_id_column}"].notna().map({True: "matched_by_id", False: "missing_by_id"})
    out["feature_match_type"] = out.apply(
        lambda row: "both_matched" if row["red_feature_match"] == "matched_by_id" and row["blue_feature_match"] == "matched_by_id" else "missing_fighter_features",
        axis=1,
    )

    return out



def _assemble_requested_features(joined_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = joined_df.copy()

    # Create r_pre_* and b_pre_* aliases from current-fighter-state columns so the
    # existing engineered V5 formulas can be reused rather than duplicated.
    out = _add_prefight_alias_columns(out)
    out = _deduplicate_columns(out)

    # Generic diff assembly for columns such as elo_diff, ewm_elo_diff, and
    # recent_form_elo_diff when matching red/blue state values exist.
    new_feature_values = {}
    for feature in feature_columns:
        if feature in out.columns:
            continue

        if feature.endswith("_diff"):
            base_name = feature[: -len("_diff")]
            red_value = _resolve_state_series(out, side="r", base_name=base_name)
            blue_value = _resolve_state_series(out, side="b", base_name=base_name)

            if red_value is not None and blue_value is not None:
                new_feature_values[feature] = red_value - blue_value

    if new_feature_values:
        out = pd.concat([out, pd.DataFrame(new_feature_values, index=out.index)], axis=1)
        out = _deduplicate_columns(out)

    # Reuse the existing V5 engineered feature formulas for features such as
    # striking_edge, grappling_edge, chin_risk_diff, etc.
    out = add_v5_engineered_features(out)
    out = _deduplicate_columns(out)

    missing_feature_values = {
        feature: 0.0
        for feature in feature_columns
        if feature not in out.columns
    }
    if missing_feature_values:
        out = pd.concat([out, pd.DataFrame(missing_feature_values, index=out.index)], axis=1)
        out = _deduplicate_columns(out)

    return out



def _add_prefight_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alias_values = {}

    for column in list(out.columns):
        if column.startswith("r_state_"):
            base = column.replace("r_state_", "", 1)
            if base not in {"fighter_id", "id", "ufcstats_fighter_id"}:
                alias = f"r_pre_{base}"
                if alias not in out.columns:
                    alias_values[alias] = out[column]
        elif column.startswith("b_state_"):
            base = column.replace("b_state_", "", 1)
            if base not in {"fighter_id", "id", "ufcstats_fighter_id"}:
                alias = f"b_pre_{base}"
                if alias not in out.columns:
                    alias_values[alias] = out[column]

    if alias_values:
        out = pd.concat([out, pd.DataFrame(alias_values, index=out.index)], axis=1)

    return out



def _resolve_state_series(df: pd.DataFrame, *, side: str, base_name: str) -> pd.Series | None:
    candidates = [
        f"{side}_state_{base_name}",
        f"{side}_pre_{base_name}",
        f"{side}_{base_name}",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce").fillna(0.0)

    return None



def _attach_feature_audit_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = _deduplicate_columns(df)
    feature_matrix = out[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    out["feature_count_expected"] = len(feature_columns)
    out["feature_count_actual"] = len([column for column in feature_columns if column in out.columns])
    out["nonzero_feature_count"] = (feature_matrix != 0).sum(axis=1)
    out["zero_feature_pct"] = 1.0 - (out["nonzero_feature_count"] / len(feature_columns))
    out["passes_feature_validation"] = out["feature_count_actual"] == out["feature_count_expected"]
    out["passes_model_data_quality"] = out.get("feature_match_type", "unknown").ne("missing_fighter_features") if isinstance(out.get("feature_match_type"), pd.Series) else True

    if "red_feature_match" not in out.columns:
        out["red_feature_match"] = "prebuilt_features"
    if "blue_feature_match" not in out.columns:
        out["blue_feature_match"] = "prebuilt_features"
    if "feature_match_type" not in out.columns:
        out["feature_match_type"] = "prebuilt_features"

    return _deduplicate_columns(out)



def _build_feature_audit(live_feature_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    columns = [
        "event_id",
        "event_name",
        "fight_id",
        "red_fighter",
        "blue_fighter",
        "red_fighter_id",
        "blue_fighter_id",
        "red_feature_match",
        "blue_feature_match",
        "feature_match_type",
        "feature_count_expected",
        "feature_count_actual",
        "nonzero_feature_count",
        "zero_feature_pct",
        "passes_feature_validation",
        "passes_model_data_quality",
    ]

    df = _deduplicate_columns(live_feature_df)
    return df[[column for column in columns if column in df.columns]].copy()



def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with duplicate column names removed, keeping first occurrence."""

    if not df.columns.duplicated().any():
        return df.copy()
    return df.loc[:, ~df.columns.duplicated()].copy()



def _has_all_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)



def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None
