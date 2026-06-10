"""Build configured UFC feature views.

This is the first adapter-style generic feature-view runner. It reads a
feature-view YAML config and dispatches to the currently validated builder for
that view. The initial goal is to reproduce the existing moneyline feature view
without changing the validated moneyline builder internals.

Run from repo root:

    python -m pipeline.features.run_build_feature_view \
        --config configs/feature_views/moneyline_base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline.common.paths import ensure_data_dirs
from pipeline.features.run_build_rolling_features import prepare_master_for_rolling
from pipeline.features.views.moneyline import build_moneyline_feature_view
from ufc_feature_engineering import add_v5_engineered_features, get_engineered_feature_list


DEFAULT_CONFIG_PATH = "configs/feature_views/moneyline_base.yaml"
SUPPORTED_VIEW_FAMILIES = {"moneyline", "prop"}
SUPPORTED_PROP_MARKETS = {"goes_distance"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build a configured UFC feature view.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to feature-view YAML config.",
    )
    return parser.parse_args()


def load_feature_view_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a feature-view config."""

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
    """Validate required feature-view config fields."""

    required_top_level = ["view_id", "view_family", "inputs", "output"]
    missing = [field for field in required_top_level if field not in config]
    if missing:
        raise ValueError(f"Feature-view config missing fields in {config_path}: {missing}")

    view_family = str(config.get("view_family"))
    if view_family not in SUPPORTED_VIEW_FAMILIES:
        raise ValueError(
            f"Unsupported view_family in {config_path}: {view_family}. "
            f"Supported: {sorted(SUPPORTED_VIEW_FAMILIES)}"
        )

    if view_family == "prop":
        market_key = str(config.get("market_key", ""))
        if market_key not in SUPPORTED_PROP_MARKETS:
            raise ValueError(
                f"Unsupported prop market_key in {config_path}: {market_key}. "
                f"Supported: {sorted(SUPPORTED_PROP_MARKETS)}"
            )
        if "label" not in config:
            raise ValueError(f"Prop feature-view config missing label block: {config_path}")

    inputs = config.get("inputs", {})
    output = config.get("output", {})

    if "master_path" not in inputs:
        raise ValueError(f"Feature-view config missing inputs.master_path: {config_path}")
    if "fighter_state_history_path" not in inputs:
        raise ValueError(
            f"Feature-view config missing inputs.fighter_state_history_path: {config_path}"
        )
    if "feature_view_path" not in output:
        raise ValueError(f"Feature-view config missing output.feature_view_path: {config_path}")


def build_feature_view_from_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Build a feature view from a YAML config and return the output path.

    This callable entry point lets training and future Model Lab workflows build
    the configured feature view before loading model features, without shelling
    out to the CLI.
    """

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

    # Current reusable base-state join. Moneyline remains the canonical validated
    # view; the first prop view reuses the same point-in-time fighter-state join
    # and then adds prop-specific labels/validation.
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
        feature_view_df = add_v5_engineered_features(feature_view_df)
        engineered_features = get_engineered_feature_list()
        missing_engineered = [
            column for column in engineered_features if column not in feature_view_df.columns
        ]
        if missing_engineered:
            raise ValueError(f"Missing engineered features: {missing_engineered}")
        print(f"Engineered features: {len(engineered_features)}")

    validate_feature_view_output(
        feature_view_df=feature_view_df,
        prepared_df=prepared_df,
        config=config,
    )

    feature_view_path.parent.mkdir(parents=True, exist_ok=True)
    feature_view_df.to_parquet(feature_view_path, index=False)

    print(f"Feature view shape: {feature_view_df.shape}")
    print(f"Saved feature view: {feature_view_path}")
    print("DONE")

    return feature_view_path


def add_prop_labels(*, feature_view_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add configured prop target labels to a feature view.

    Initial scope intentionally supports only the first V1 prop slice:
    market_key=goes_distance, target=method == Decision.
    """

    market_key = str(config.get("market_key", ""))
    if market_key != "goes_distance":
        raise ValueError(f"Unsupported prop market for label generation: {market_key}")

    label_config = config.get("label", {}) or {}
    target_column = str(label_config.get("target_column", "target_goes_distance"))

    if "method" not in feature_view_df.columns:
        raise ValueError("Cannot build goes-distance target because method column is missing.")

    out = feature_view_df.copy()
    method = out["method"].astype("string").fillna("").str.strip().str.lower()
    out[target_column] = method.eq("decision").astype(int)

    return out


def print_target_distribution(feature_view_df: pd.DataFrame, target_column: str) -> None:
    """Print positive-rate diagnostics for a target column."""

    if target_column not in feature_view_df.columns:
        print(f"Target distribution unavailable; missing column: {target_column}")
        return

    target = pd.to_numeric(feature_view_df[target_column], errors="coerce")
    positive_count = int(target.fillna(0).sum())
    total_count = int(target.notna().sum())
    positive_rate = positive_count / total_count if total_count else 0.0

    print(
        "Target distribution: "
        f"positives={positive_count}, total={total_count}, positive_rate={positive_rate:.4f}"
    )


def main() -> None:
    """Build a feature view from YAML config."""

    args = parse_args()
    build_feature_view_from_config(args.config)


def validate_feature_view_output(
    feature_view_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Validate generated feature-view dataframe against config contracts."""

    contracts = config.get("contracts", {})
    validation = config.get("validation", {})

    if contracts.get("expected_rows_match_prepared_fights", False):
        if len(feature_view_df) != len(prepared_df):
            raise ValueError(
                "Feature-view row mismatch: "
                f"expected {len(prepared_df)}, observed {len(feature_view_df)}"
            )

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
                raise ValueError(
                    "Feature view has missing fighter-state matches. "
                    f"Rows affected: {missing_state_count}"
                )

    expected_shape = validation.get("expected_feature_view_shape", {})
    expected_columns = expected_shape.get("columns_current")
    if expected_columns is not None and int(expected_columns) != len(feature_view_df.columns):
        raise ValueError(
            "Feature-view column count mismatch: "
            f"expected {expected_columns}, observed {len(feature_view_df.columns)}"
        )

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


def validate_target_distribution(
    *,
    feature_view_df: pd.DataFrame,
    target_column: str,
    min_positive_rate: Any = None,
    max_positive_rate: Any = None,
) -> None:
    """Validate a configured binary target's positive-rate range."""

    if target_column not in feature_view_df.columns:
        raise ValueError(f"Target distribution validation missing target column: {target_column}")

    target = pd.to_numeric(feature_view_df[target_column], errors="coerce")
    total_count = int(target.notna().sum())
    if not total_count:
        raise ValueError(f"Target column has no valid values: {target_column}")

    positive_rate = float(target.fillna(0).mean())

    if min_positive_rate is not None and positive_rate < float(min_positive_rate):
        raise ValueError(
            f"Target positive rate below minimum for {target_column}: "
            f"{positive_rate:.4f} < {float(min_positive_rate):.4f}"
        )

    if max_positive_rate is not None and positive_rate > float(max_positive_rate):
        raise ValueError(
            f"Target positive rate above maximum for {target_column}: "
            f"{positive_rate:.4f} > {float(max_positive_rate):.4f}"
        )


if __name__ == "__main__":
    main()
