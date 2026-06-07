"""Production runner for rolling UFC feature generation.

This replaces the executable role of UFC_rolling_dataset_V4_refactored.ipynb.

Run from repo root:

    python -m pipeline.features.run_build_rolling_features

Pipeline:
- Load data/master/ufc_master.parquet
- Build base rolling fighter-state features
- Add EWM/recent-form features
- Add engineered moneyline features
- Validate the 483-column rolling feature contract
- Write data/features/UFC_enhanced_rolling_features_EWM.parquet
"""

from __future__ import annotations

import pandas as pd

from pipeline.common.paths import MASTER_PATH, ROLLING_FEATURES_PATH, ensure_data_dirs
from pipeline.features.base.build_rolling_features import build_rolling_base_features
from pipeline.features.base.ewm_features import add_ewm_feature_layer
from ufc_feature_engineering import add_v5_engineered_features

EXPECTED_ROLLING_COLUMNS = 483


def prepare_master_for_rolling(master_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare master fight rows for chronological rolling feature generation."""
    df = master_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "event_id", "fight_id"]).reset_index(drop=True)

    # Preserve the notebook convention: target=1 means red fighter won.
    df["target"] = (
        df["winner"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("red")
        .astype(int)
    )

    return df


def build_full_rolling_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Build the complete 483-column rolling feature dataframe."""
    df = prepare_master_for_rolling(master_df)
    rolling_df = build_rolling_base_features(df)
    rolling_df = add_ewm_feature_layer(rolling_df)
    rolling_df = add_v5_engineered_features(rolling_df)
    return rolling_df


def validate_rolling_contract(rolling_df: pd.DataFrame) -> None:
    """Raise if the rolling feature dataframe violates the expected contract."""
    observed_columns = len(rolling_df.columns)
    if observed_columns != EXPECTED_ROLLING_COLUMNS:
        raise ValueError(
            f"Rolling feature column count mismatch: "
            f"expected {EXPECTED_ROLLING_COLUMNS}, observed {observed_columns}"
        )


def main() -> None:
    ensure_data_dirs()

    print("=" * 80)
    print("BUILD UFC ROLLING FEATURES")
    print("=" * 80)
    print(f"Master path: {MASTER_PATH}")
    print(f"Output path: {ROLLING_FEATURES_PATH}")

    master_df = pd.read_parquet(MASTER_PATH)
    print(f"Master shape: {master_df.shape}")

    rolling_df = build_full_rolling_features(master_df)
    print(f"Rolling shape: {rolling_df.shape}")

    validate_rolling_contract(rolling_df)

    rolling_df.to_parquet(ROLLING_FEATURES_PATH, index=False)
    print("Saved rolling features successfully.")
    print("DONE")


if __name__ == "__main__":
    main()
