"""Build configured UFC feature views."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline.common.paths import ensure_data_dirs
from pipeline.features.formula_engine import apply_formula_features
from pipeline.features.registry_feature_builder import apply_registry_feature_definitions
from pipeline.features.run_build_rolling_features import prepare_master_for_rolling
from pipeline.features.views.moneyline import build_moneyline_feature_view
from ufc_feature_engineering import add_v5_engineered_features, get_engineered_feature_list

DEFAULT_CONFIG_PATH = "configs/feature_views/moneyline_base.yaml"
DEFAULT_MODEL_LAB_FEATURE_REGISTRY = "configs/features/feature_registry.yaml"
SUPPORTED_VIEW_FAMILIES = {"moneyline", "prop"}
SUPPORTED_PROP_MARKETS = {"goes_distance", "over_under_2_5"}
STYLE_SCORE_NAMES = [
    "control_wrestler",
    "ko_finisher",
    "submission_grappler",
    "decision_technician",
    "all_round_finisher",
]
STYLE_EDGE_PAIRS = {
    "style_edge_ko_finisher_vs_decision_technician": ("ko_finisher", "decision_technician"),
    "style_edge_decision_technician_vs_submission_grappler": ("decision_technician", "submission_grappler"),
    "style_edge_control_wrestler_vs_ko_finisher": ("control_wrestler", "ko_finisher"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a configured UFC feature view.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to feature-view YAML config.")
    return parser.parse_args()


def load_feature_view_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature-view config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Feature-view config must be a dictionary: {path}")
    validate_feature_view_config(config, path)
    return config


def validate_feature_view_config(config: dict[str, Any], config_path: Path) -> None:
    missing = [field for field in ["view_id", "view_family", "inputs", "output"] if field not in config]
    if missing:
        raise ValueError(f"Feature-view config missing fields in {config_path}: {missing}")
    view_family = str(config.get("view_family"))
    if view_family not in SUPPORTED_VIEW_FAMILIES:
        raise ValueError(f"Unsupported view_family in {config_path}: {view_family}")
    if view_family == "prop":
        market_key = str(config.get("market_key", ""))
        if market_key not in SUPPORTED_PROP_MARKETS:
            raise ValueError(f"Unsupported prop market_key in {config_path}: {market_key}")
        if "label" not in config:
            raise ValueError(f"Prop feature-view config missing label block: {config_path}")
    inputs = config.get("inputs", {})
    output = config.get("output", {})
    if "master_path" not in inputs:
        raise ValueError(f"Feature-view config missing inputs.master_path: {config_path}")
    if "fighter_state_history_path" not in inputs:
        raise ValueError(f"Feature-view config missing inputs.fighter_state_history_path: {config_path}")
    if "feature_view_path" not in output:
        raise ValueError(f"Feature-view config missing output.feature_view_path: {config_path}")


def build_feature_view_from_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    config_path = Path(config_path)
    config = load_feature_view_config(config_path)
    ensure_data_dirs()
    view_id = str(config["view_id"])
    view_family = str(config["view_family"])
    market_key = str(config.get("market_key", ""))

    print("=" * 80)
    print("BUILD UFC FEATURE VIEW")
    print("=" * 80)
    print(f"Config path       : {config_path}")
    print(f"View ID           : {view_id}")
    print(f"View family       : {view_family}")
    if market_key:
        print(f"Market key        : {market_key}")

    inputs = config["inputs"]
    output = config["output"]
    include = config.get("include", {})
    master_path = Path(inputs["master_path"])
    fighter_state_history_path = Path(inputs["fighter_state_history_path"])
    feature_view_path = Path(output["feature_view_path"])

    print(f"Master path       : {master_path}")
    print(f"Fighter state path: {fighter_state_history_path}")
    print(f"Output path       : {feature_view_path}")

    master_df = pd.read_parquet(master_path)
    print(f"Master shape      : {master_df.shape}")
    prepared_df = prepare_master_for_rolling(master_df)
    print(f"Prepared shape    : {prepared_df.shape}")
    fighter_state_history_df = pd.read_parquet(fighter_state_history_path)
    print(f"State shape       : {fighter_state_history_df.shape}")

    feature_view_df = build_moneyline_feature_view(
        prepared_fights_df=prepared_df,
        fighter_state_history_df=fighter_state_history_df,
    )

    if view_family == "prop":
        feature_view_df = add_prop_labels(feature_view_df=feature_view_df, config=config)
        print(f"Prop target column: {config['label']['target_column']}")
        print_target_distribution(feature_view_df, str(config["label"]["target_column"]))

    engineered_config = include.get("engineered_features", {})
    if engineered_config.get("enabled", False):
        feature_view_df = add_physical_alias_columns(feature_view_df)
        feature_view_df = add_v5_engineered_features(feature_view_df)
        feature_view_df = remove_duplicate_columns(feature_view_df, keep="last")
        engineered_features = get_engineered_feature_list()
        missing_engineered = [column for column in engineered_features if column not in feature_view_df.columns]
        if missing_engineered:
            raise ValueError(f"Missing engineered features: {missing_engineered}")
        print(f"Engineered features: {len(engineered_features)}")

    registry_result = apply_registry_feature_definitions(
        feature_view_df,
        allowed_statuses={"active", "draft"},
        overwrite_existing=True,
    )
    feature_view_df = registry_result.dataframe
    if registry_result.generated_columns:
        print(f"Registry features materialized: {len(registry_result.generated_columns)} ({registry_result.generated_columns})")
    if registry_result.missing_inputs:
        print(f"Registry features skipped for missing inputs: {registry_result.missing_inputs}")

    style_edge_config = include.get("style_matchup_edges", {}) or {}
    if style_edge_config.get("enabled", False):
        feature_view_df = add_style_matchup_edge_features(feature_view_df)
        print("Style matchup edge features enabled")

    feature_view_df = apply_model_lab_formula_features(feature_view_df=feature_view_df, config=config)
    validate_feature_view_output(feature_view_df=feature_view_df, prepared_df=prepared_df, config=config)
    feature_view_path.parent.mkdir(parents=True, exist_ok=True)
    feature_view_df.to_parquet(feature_view_path, index=False)
    print(f"Feature view shape: {feature_view_df.shape}")
    print(f"Saved feature view: {feature_view_path}")
    print("DONE")
    return feature_view_path


def add_style_matchup_edge_features(feature_view_df: pd.DataFrame) -> pd.DataFrame:
    """Add V9 style-score abs-diff and persistent matchup edge features."""

    out = feature_view_df.copy()
    new_columns: dict[str, Any] = {}

    for style_name in STYLE_SCORE_NAMES:
        diff_col = f"style_{style_name}_score_diff"
        if diff_col in out.columns:
            new_columns[f"style_{style_name}_score_abs_diff"] = out[diff_col].abs()

    for edge_name, (red_style, blue_style) in STYLE_EDGE_PAIRS.items():
        direct_r = f"r_pre_style_{red_style}_score"
        direct_b = f"b_pre_style_{blue_style}_score"
        reverse_r = f"r_pre_style_{blue_style}_score"
        reverse_b = f"b_pre_style_{red_style}_score"
        missing = [column for column in [direct_r, direct_b, reverse_r, reverse_b] if column not in out.columns]
        if missing:
            raise ValueError(f"Missing style edge inputs for {edge_name}: {missing}")
        direct = out[direct_r] * out[direct_b]
        reverse = out[reverse_r] * out[reverse_b]
        new_columns[edge_name] = direct
        new_columns[f"{edge_name}_reverse"] = reverse
        new_columns[f"{edge_name}_net"] = direct - reverse

    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)
    return out


def add_physical_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in ["r", "b"]:
        for field in ["age", "height", "reach", "weight"]:
            legacy_col = f"{side}_{field}"
            state_col = f"{side}_pre_{field}"
            if legacy_col not in out.columns and state_col in out.columns:
                out[legacy_col] = out[state_col]
    return out


def remove_duplicate_columns(df: pd.DataFrame, *, keep: str = "last") -> pd.DataFrame:
    if df.columns.has_duplicates:
        return df.loc[:, ~df.columns.duplicated(keep=keep)].copy()
    return df


def apply_model_lab_formula_features(*, feature_view_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    formula_config = config.get("model_lab_formula_features", {}) or {}
    if not formula_config.get("enabled", False):
        return feature_view_df
    registry_path = Path(formula_config.get("registry_path") or DEFAULT_MODEL_LAB_FEATURE_REGISTRY)
    if not registry_path.exists():
        raise FileNotFoundError(f"Feature registry not found: {registry_path}")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(registry, dict):
        raise ValueError(f"Feature registry must be a dictionary: {registry_path}")
    before_columns = set(feature_view_df.columns)
    output_df = apply_formula_features(
        feature_view_df,
        registry,
        selected_bundles=formula_config.get("bundles") or [],
        selected_features=formula_config.get("features") or [],
        allowed_statuses=formula_config.get("statuses") or {"active"},
    )
    added_columns = sorted(set(output_df.columns) - before_columns)
    print(f"Model Lab formula features: {len(added_columns)}")
    if added_columns:
        print(f"Formula columns added: {added_columns}")
    return output_df


def add_prop_labels(*, feature_view_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    market_key = str(config.get("market_key", ""))
    target_column = str((config.get("label", {}) or {}).get("target_column", f"target_{market_key}"))
    out = feature_view_df.copy()

    if market_key == "goes_distance":
        if "method" not in out.columns:
            raise ValueError("Cannot build goes-distance target because method column is missing.")
        decision_methods = {"decision - unanimous", "decision - split", "decision - majority"}
        method = out["method"].astype("string").fillna("").str.strip().str.lower()
        out[target_column] = method.isin(decision_methods).astype(int)
        return out

    if market_key == "over_under_2_5":
        if "match_time_sec" not in out.columns:
            raise ValueError("Cannot build over/under 2.5 target because match_time_sec column is missing.")
        elapsed_seconds = pd.to_numeric(out["match_time_sec"], errors="coerce")
        if elapsed_seconds.isna().any():
            missing = int(elapsed_seconds.isna().sum())
            raise ValueError(f"Cannot build over/under 2.5 target; match_time_sec missing in {missing} rows.")
        out[target_column] = elapsed_seconds.gt(750.0).astype(int)
        return out

    raise ValueError(f"Unsupported prop market for label generation: {market_key}")


def print_target_distribution(feature_view_df: pd.DataFrame, target_column: str) -> None:
    if target_column not in feature_view_df.columns:
        print(f"Target distribution unavailable; missing column: {target_column}")
        return
    target = pd.to_numeric(feature_view_df[target_column], errors="coerce")
    positive_count = int(target.fillna(0).sum())
    total_count = int(target.notna().sum())
    positive_rate = positive_count / total_count if total_count else 0.0
    print(f"Target distribution: positives={positive_count}, total={total_count}, positive_rate={positive_rate:.4f}")


def validate_feature_view_output(feature_view_df: pd.DataFrame, prepared_df: pd.DataFrame, config: dict[str, Any]) -> None:
    contracts = config.get("contracts", {})
    validation = config.get("validation", {})
    if contracts.get("expected_rows_match_prepared_fights", False):
        if len(feature_view_df) != len(prepared_df):
            raise ValueError(f"Feature-view row mismatch: expected {len(prepared_df)}, observed {len(feature_view_df)}")
    required_columns = contracts.get("required_columns", [])
    missing_required = [column for column in required_columns if column not in feature_view_df.columns]
    if missing_required:
        raise ValueError(f"Feature view missing required columns: {missing_required}")
    if contracts.get("require_no_missing_state_matches", False):
        state_check_columns = ["r_pre_elo", "b_pre_elo"]
        if all(column in feature_view_df.columns for column in state_check_columns):
            missing_state_count = int(feature_view_df[state_check_columns].isna().any(axis=1).sum())
            print(f"Missing state matches: {missing_state_count}")
            if missing_state_count:
                raise ValueError(f"Feature view has missing fighter-state matches. Rows affected: {missing_state_count}")
    expected_shape = validation.get("expected_feature_view_shape", {})
    expected_columns = expected_shape.get("columns_current")
    allow_extra_columns = bool(validation.get("allow_extra_columns", False))
    if expected_columns is not None:
        observed_columns = len(feature_view_df.columns)
        expected_columns = int(expected_columns)
        if observed_columns < expected_columns or (observed_columns != expected_columns and not allow_extra_columns):
            raise ValueError(f"Feature-view column count mismatch: expected {expected_columns}, observed {observed_columns}, allow_extra_columns={allow_extra_columns}")
    target_distribution = validation.get("target_distribution", {}) or {}
    if target_distribution:
        target_column = str((config.get("label", {}) or {}).get("target_column", ""))
        if target_column:
            validate_target_distribution(
                feature_view_df=feature_view_df,
                target_column=target_column,
                min_positive_rate=target_distribution.get("min_positive_rate"),
                max_positive_rate=target_distribution.get("max_positive_rate"),
            )


def validate_target_distribution(*, feature_view_df: pd.DataFrame, target_column: str, min_positive_rate: Any = None, max_positive_rate: Any = None) -> None:
    if target_column not in feature_view_df.columns:
        raise ValueError(f"Target distribution validation missing target column: {target_column}")
    target = pd.to_numeric(feature_view_df[target_column], errors="coerce")
    total_count = int(target.notna().sum())
    if not total_count:
        raise ValueError(f"Target column has no valid values: {target_column}")
    positive_rate = float(target.fillna(0).mean())
    if min_positive_rate is not None and positive_rate < float(min_positive_rate):
        raise ValueError(f"Target positive rate below minimum for {target_column}: {positive_rate:.4f} < {float(min_positive_rate):.4f}")
    if max_positive_rate is not None and positive_rate > float(max_positive_rate):
        raise ValueError(f"Target positive rate above maximum for {target_column}: {positive_rate:.4f} > {float(max_positive_rate):.4f}")


def main() -> None:
    args = parse_args()
    build_feature_view_from_config(args.config)


if __name__ == "__main__":
    main()
