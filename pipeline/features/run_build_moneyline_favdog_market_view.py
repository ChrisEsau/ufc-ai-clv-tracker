"""Build wide favorite/dog market-aware moneyline feature view.

Run from repo root:

    python -m pipeline.features.run_build_moneyline_favdog_market_view

This runner creates a production-style training view for configurable
market-aware ensemble models.

Inputs:
    data/features/moneyline_rfs_feature_view.parquet
    data/market/historical_market_outcomes.parquet

Outputs:
    data/features/moneyline_favdog_market_feature_view.parquet
    data/audits/moneyline_favdog_market_feature_view_validation.parquet

The view is intentionally wide. It is not Top60-specific. Downstream ensemble
configs decide which favpersp__/dogpersp__ columns each child model uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    HISTORICAL_MARKET_OUTCOMES_PATH,
    MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_AUDIT_PATH,
    MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_PATH,
    FEATURES_DIR,
    ensure_data_dirs,
)


MONEYLINE_RFS_FEATURE_VIEW_PATH = FEATURES_DIR / "moneyline_rfs_feature_view.parquet"

MARKET_FEATURES = [
    "favorite_odds",
    "dog_odds",
    "favorite_implied_probability",
    "dog_implied_probability",
    "market_prob_gap",
    "abs_market_prob_gap",
    "favorite_odds_abs",
    "dog_odds_abs",
    "market_price_width",
]

BASE_META_COLUMNS = [
    "event_id",
    "event_name",
    "date",
    "fight_id",
    "r_name",
    "b_name",
    "r_id",
    "b_id",
    "winner",
    "winner_id",
    "target",
]

EXCLUDED_FEATURE_COLUMNS = {
    "target",
    "event_id",
    "event_name",
    "date",
    "fight_id",
    "r_name",
    "b_name",
    "r_id",
    "b_id",
    "r_nick_name",
    "b_nick_name",
    "winner",
    "winner_id",
    "base_fight_id",
    "state_fight_id",
    "row_perspective",
}

EXCLUDED_PREFIXES = (
    "r_",
    "b_",
)

EXCLUDED_SUFFIXES = (
    "_has_state",
    "_either_has_state",
    "_both_have_state",
)


@dataclass(frozen=True)
class BuildSummary:
    input_feature_rows: int
    output_rows: int
    clean_feature_count: int
    market_moneyline_rows: int
    market_moneyline_fights: int
    both_odds_rows: int
    coverage_pct: float


def _audit_row(
    check_name: str,
    status: str,
    observed: object,
    details: str = "",
    severity: str = "fatal",
) -> dict[str, object]:
    return {
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "observed": observed,
        "details": details,
    }


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def american_profit_per_100(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return odds
    return 10000.0 / abs(odds)


def resolve_clean_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Resolve model-eligible clean columns from the red-minus-blue feature view.

    The production fav/dog view treats selected clean columns as signed
    red-minus-blue features. It excludes metadata, side-specific raw state columns,
    and RFS availability flags. Future feature sets that need non-signed transforms
    should declare transform rules explicitly.
    """

    clean_features: list[str] = []

    for column in feature_df.columns:
        if column in EXCLUDED_FEATURE_COLUMNS:
            continue

        if column.startswith(EXCLUDED_PREFIXES):
            continue

        if column.endswith(EXCLUDED_SUFFIXES):
            continue

        if not pd.api.types.is_numeric_dtype(feature_df[column]):
            continue

        clean_features.append(column)

    return clean_features


def prepare_moneyline_market(market_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare one historical moneyline price per fight/fighter."""

    required = [
        "fight_id",
        "market_key",
        "bookmaker",
        "outcome_fighter_id",
        "outcome_label",
        "outcome_side",
        "american_odds",
        "implied_probability",
    ]
    missing = [column for column in required if column not in market_df.columns]
    if missing:
        raise ValueError(f"Historical market file missing required columns: {missing}")

    ml = market_df[market_df["market_key"].eq("moneyline")].copy()
    ml = ml.dropna(subset=["fight_id", "outcome_fighter_id", "american_odds"])

    ml["american_odds"] = _safe_numeric(ml["american_odds"])
    ml["implied_probability"] = _safe_numeric(ml["implied_probability"])

    ml = ml.dropna(subset=["american_odds", "implied_probability"])

    # Historical file currently uses legacy_consensus. Keep deterministic priority
    # in case synthetic or future sources are present.
    priority = {
        "legacy_consensus": 0,
        "synthetic_legacy_consensus": 1,
    }
    ml["bookmaker_priority"] = ml["bookmaker"].map(priority).fillna(99)

    ml = (
        ml.sort_values(
            [
                "fight_id",
                "outcome_fighter_id",
                "bookmaker_priority",
                "historical_market_timestamp",
            ],
            ascending=[True, True, True, True],
        )
        .drop_duplicates(["fight_id", "outcome_fighter_id"], keep="first")
        .copy()
    )

    return ml


def join_red_blue_odds(feature_df: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    """Join historical red and blue moneyline odds to feature rows."""

    required = ["fight_id", "r_id", "b_id", "target"]
    missing = [column for column in required if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Feature view missing required columns: {missing}")

    price_cols = [
        "fight_id",
        "outcome_fighter_id",
        "outcome_label",
        "outcome_side",
        "american_odds",
        "implied_probability",
        "bookmaker",
    ]

    red_prices = ml[price_cols].rename(
        columns={
            "outcome_fighter_id": "r_id",
            "outcome_label": "r_market_label",
            "outcome_side": "r_market_side",
            "american_odds": "r_american_odds",
            "implied_probability": "r_implied_probability",
            "bookmaker": "r_bookmaker",
        }
    )

    blue_prices = ml[price_cols].rename(
        columns={
            "outcome_fighter_id": "b_id",
            "outcome_label": "b_market_label",
            "outcome_side": "b_market_side",
            "american_odds": "b_american_odds",
            "implied_probability": "b_implied_probability",
            "bookmaker": "b_bookmaker",
        }
    )

    joined = feature_df.merge(red_prices, on=["fight_id", "r_id"], how="left")
    joined = joined.merge(blue_prices, on=["fight_id", "b_id"], how="left")

    joined["has_red_odds"] = joined["r_american_odds"].notna()
    joined["has_blue_odds"] = joined["b_american_odds"].notna()
    joined["has_both_odds"] = joined["has_red_odds"] & joined["has_blue_odds"]

    return joined


def build_favdog_market_view(
    feature_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> tuple[pd.DataFrame, BuildSummary]:
    """Build the wide favorite/dog market-aware feature view."""

    clean_features = resolve_clean_feature_columns(feature_df)
    if not clean_features:
        raise ValueError("No clean feature columns resolved from feature view.")

    ml = prepare_moneyline_market(market_df)
    joined = join_red_blue_odds(feature_df, ml)

    usable = joined[joined["has_both_odds"]].copy()

    if usable.empty:
        raise ValueError("No rows have both red and blue historical moneyline odds.")

    usable["target"] = pd.to_numeric(usable["target"], errors="coerce")
    usable = usable.dropna(subset=["target"]).copy()
    usable["target"] = usable["target"].astype(int)

    # Favorite is the side with higher implied probability. Tie-break on lower
    # American odds, which handles near-pickem prices like -112 / -108.
    red_is_favorite = (
        usable["r_implied_probability"].gt(usable["b_implied_probability"])
        | (
            usable["r_implied_probability"].eq(usable["b_implied_probability"])
            & usable["r_american_odds"].le(usable["b_american_odds"])
        )
    )

    usable["favorite_side"] = np.where(red_is_favorite, "red", "blue")
    usable["dog_side"] = np.where(red_is_favorite, "blue", "red")

    usable["favorite_fighter_id"] = np.where(red_is_favorite, usable["r_id"], usable["b_id"])
    usable["dog_fighter_id"] = np.where(red_is_favorite, usable["b_id"], usable["r_id"])

    usable["favorite_fighter_name"] = np.where(red_is_favorite, usable["r_name"], usable["b_name"])
    usable["dog_fighter_name"] = np.where(red_is_favorite, usable["b_name"], usable["r_name"])

    usable["favorite_odds"] = np.where(red_is_favorite, usable["r_american_odds"], usable["b_american_odds"])
    usable["dog_odds"] = np.where(red_is_favorite, usable["b_american_odds"], usable["r_american_odds"])

    usable["favorite_implied_probability"] = np.where(
        red_is_favorite,
        usable["r_implied_probability"],
        usable["b_implied_probability"],
    )
    usable["dog_implied_probability"] = np.where(
        red_is_favorite,
        usable["b_implied_probability"],
        usable["r_implied_probability"],
    )

    usable["favorite_bookmaker"] = np.where(red_is_favorite, usable["r_bookmaker"], usable["b_bookmaker"])
    usable["dog_bookmaker"] = np.where(red_is_favorite, usable["b_bookmaker"], usable["r_bookmaker"])

    usable["favorite_won"] = np.where(red_is_favorite, usable["target"].eq(1), usable["target"].eq(0)).astype(int)
    usable["dog_won"] = 1 - usable["favorite_won"]

    usable["market_prob_gap"] = usable["favorite_implied_probability"] - usable["dog_implied_probability"]
    usable["abs_market_prob_gap"] = usable["market_prob_gap"].abs()
    usable["favorite_odds_abs"] = usable["favorite_odds"].abs()
    usable["dog_odds_abs"] = usable["dog_odds"].abs()
    usable["market_price_width"] = (usable["favorite_odds"] - usable["dog_odds"]).abs()

    usable["favorite_profit_per_100"] = usable["favorite_odds"].apply(american_profit_per_100)
    usable["dog_profit_per_100"] = usable["dog_odds"].apply(american_profit_per_100)

    # Build perspective features. Current clean feature columns are treated as
    # signed red-minus-blue features.
    favorite_sign = np.where(red_is_favorite, 1.0, -1.0)

    perspective_data: dict[str, pd.Series] = {}
    for column in clean_features:
        values = _safe_numeric(usable[column]).fillna(0.0)
        fav_values = values * favorite_sign
        perspective_data[f"favpersp__{column}"] = fav_values
        perspective_data[f"dogpersp__{column}"] = -fav_values

    perspective_df = pd.DataFrame(perspective_data, index=usable.index)

    meta_cols = [column for column in BASE_META_COLUMNS if column in usable.columns]
    extra_cols = [
        "favorite_side",
        "dog_side",
        "favorite_fighter_id",
        "dog_fighter_id",
        "favorite_fighter_name",
        "dog_fighter_name",
        "favorite_won",
        "dog_won",
        "favorite_odds",
        "dog_odds",
        "favorite_implied_probability",
        "dog_implied_probability",
        "market_prob_gap",
        "abs_market_prob_gap",
        "favorite_odds_abs",
        "dog_odds_abs",
        "market_price_width",
        "favorite_profit_per_100",
        "dog_profit_per_100",
        "favorite_bookmaker",
        "dog_bookmaker",
    ]

    output = pd.concat(
        [
            usable[meta_cols + extra_cols].reset_index(drop=True),
            perspective_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output["test_year"] = pd.to_datetime(output["date"], errors="coerce").dt.year

    summary = BuildSummary(
        input_feature_rows=len(feature_df),
        output_rows=len(output),
        clean_feature_count=len(clean_features),
        market_moneyline_rows=len(ml),
        market_moneyline_fights=int(ml["fight_id"].nunique()),
        both_odds_rows=int(joined["has_both_odds"].sum()),
        coverage_pct=float(len(output) / len(feature_df)) if len(feature_df) else 0.0,
    )

    return output, summary


def build_validation_audit(
    *,
    output_df: pd.DataFrame,
    summary: BuildSummary,
) -> pd.DataFrame:
    """Build validation audit rows."""

    rows: list[dict[str, object]] = []

    rows.append(
        _audit_row(
            "output_rows_positive",
            "PASS" if len(output_df) > 0 else "FAIL",
            len(output_df),
        )
    )

    rows.append(
        _audit_row(
            "coverage_at_least_50pct",
            "PASS" if summary.coverage_pct >= 0.50 else "WARN",
            round(summary.coverage_pct, 4),
            severity="warn",
        )
    )

    target_nulls = int(output_df["target"].isna().sum()) if "target" in output_df.columns else len(output_df)
    rows.append(
        _audit_row(
            "target_non_null",
            "PASS" if target_nulls == 0 else "FAIL",
            target_nulls,
        )
    )

    target_match = int((output_df["favorite_won"] + output_df["dog_won"]).eq(1).sum())
    rows.append(
        _audit_row(
            "favorite_dog_targets_complementary",
            "PASS" if target_match == len(output_df) else "FAIL",
            f"{target_match} / {len(output_df)}",
        )
    )

    market_null_cols = [
        "favorite_odds",
        "dog_odds",
        "favorite_implied_probability",
        "dog_implied_probability",
    ]
    market_nulls = int(output_df[market_null_cols].isna().sum().sum())
    rows.append(
        _audit_row(
            "market_features_non_null",
            "PASS" if market_nulls == 0 else "FAIL",
            market_nulls,
        )
    )

    fav_cols = [column for column in output_df.columns if column.startswith("favpersp__")]
    dog_cols = [column for column in output_df.columns if column.startswith("dogpersp__")]

    rows.append(
        _audit_row(
            "favorite_perspective_features_exist",
            "PASS" if fav_cols else "FAIL",
            len(fav_cols),
        )
    )
    rows.append(
        _audit_row(
            "dog_perspective_features_exist",
            "PASS" if dog_cols else "FAIL",
            len(dog_cols),
        )
    )
    rows.append(
        _audit_row(
            "favorite_dog_feature_counts_match",
            "PASS" if len(fav_cols) == len(dog_cols) else "FAIL",
            f"fav={len(fav_cols)}, dog={len(dog_cols)}",
        )
    )

    numeric_cols = output_df.select_dtypes(include=[np.number]).columns.tolist()
    inf_count = 0
    if numeric_cols:
        numeric = output_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        inf_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())

    rows.append(
        _audit_row(
            "no_infinite_numeric_values",
            "PASS" if inf_count == 0 else "FAIL",
            inf_count,
        )
    )

    audit_df = pd.DataFrame(rows)

    # Keep audit parquet schema stable. The observed column can contain counts,
    # ratios, and human-readable strings like "6467 / 6467".
    for column in ["check_name", "status", "severity", "observed", "details"]:
        if column in audit_df.columns:
            audit_df[column] = audit_df[column].astype(str)

    return audit_df


def main() -> None:
    ensure_data_dirs()

    print("=" * 80)
    print("BUILD MONEYLINE FAV/DOG MARKET FEATURE VIEW")
    print("=" * 80)
    print(f"Feature source path : {MONEYLINE_RFS_FEATURE_VIEW_PATH}")
    print(f"Market source path  : {HISTORICAL_MARKET_OUTCOMES_PATH}")
    print(f"Output path         : {MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_PATH}")
    print(f"Audit path          : {MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_AUDIT_PATH}")

    if not MONEYLINE_RFS_FEATURE_VIEW_PATH.exists():
        raise FileNotFoundError(MONEYLINE_RFS_FEATURE_VIEW_PATH)
    if not HISTORICAL_MARKET_OUTCOMES_PATH.exists():
        raise FileNotFoundError(HISTORICAL_MARKET_OUTCOMES_PATH)

    feature_df = pd.read_parquet(MONEYLINE_RFS_FEATURE_VIEW_PATH)
    market_df = pd.read_parquet(HISTORICAL_MARKET_OUTCOMES_PATH)

    print(f"Feature source shape: {feature_df.shape}")
    print(f"Market source shape : {market_df.shape}")

    output_df, summary = build_favdog_market_view(feature_df, market_df)
    audit_df = build_validation_audit(output_df=output_df, summary=summary)

    MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_parquet(MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_PATH, index=False)
    audit_df.to_parquet(MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_AUDIT_PATH, index=False)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Input feature rows       : {summary.input_feature_rows}")
    print(f"Output rows              : {summary.output_rows}")
    print(f"Clean feature count      : {summary.clean_feature_count}")
    print(f"Market moneyline rows    : {summary.market_moneyline_rows}")
    print(f"Market moneyline fights  : {summary.market_moneyline_fights}")
    print(f"Rows with both odds      : {summary.both_odds_rows}")
    print(f"Coverage pct             : {summary.coverage_pct:.4f}")
    print(f"Output columns           : {output_df.shape[1]}")

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(audit_df.to_string(index=False))

    fatal_failures = audit_df[
        audit_df["severity"].eq("fatal") & audit_df["status"].eq("FAIL")
    ]
    if len(fatal_failures):
        raise RuntimeError(
            "Fatal validation failures in moneyline fav/dog market feature view: "
            f"{fatal_failures['check_name'].tolist()}"
        )

    print()
    print("Saved feature view:", MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_PATH)
    print("Saved audit       :", MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_AUDIT_PATH)
    print("DONE")


if __name__ == "__main__":
    main()
