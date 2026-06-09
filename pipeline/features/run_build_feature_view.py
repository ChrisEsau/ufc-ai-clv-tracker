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


def main() -> None:
    """Build a feature view from YAML config."""

    args = parse_args()
    config_path = Path(args.config)
    config = load_feature_view_config(config_path)

    ensure_data_dirs()

    view_id = str(config["view_id"])
    view_family = str(config["view_family"])

    print("=" * 80)
    print("BUILD UFC FEATURE VIEW")
    print("=" * 80)
    print(f"Config path       : {config_path}")
    print(f"View ID           : {view_id}")
    print(f"View family       : {view_family}")

    if view_family != "moneyline":
        raise ValueError(
            "Generic feature-view runner currently supports only view_family='moneyline'. "
            f"Observed: {view_family}"
        )

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


if __name__ == "__main__":
    main()
