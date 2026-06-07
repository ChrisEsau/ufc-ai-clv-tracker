"""Production runner for rolling UFC feature generation.

This replaces the executable role of UFC_rolling_dataset_V4_refactored.ipynb.

Run from repo root:

    python -m pipeline.features.run_build_rolling_features

Pipeline:
- Load data/master/ufc_master.parquet
- Build an ID-based moneyline target where target=1 means winner_id == r_id
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
TARGET_ID_COLUMNS = ["winner_id", "r_id", "b_id"]


def _normalized_id_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return normalized string IDs for target validation and comparison."""
    return df[column].astype("string").str.strip()


def add_id_based_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the canonical moneyline target from fighter IDs.

    The canonical target is:

    - ``1`` when the red fighter won: ``winner_id == r_id``
    - ``0`` when the blue fighter won: ``winner_id == b_id``

    This intentionally does not use the ``winner`` name column. In the master
    dataset, ``winner`` stores the winning fighter name, not the literal corner
    value ``red`` or ``blue``. Using names or corner text here can silently
    corrupt labels and inflate model metrics after symmetry augmentation.
    """
    missing_columns = [column for column in TARGET_ID_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Cannot build rolling features because required target ID columns are missing: "
            f"{missing_columns}"
        )

    out = df.copy()
    winner_id = _normalized_id_series(out, "winner_id")
    red_id = _normalized_id_series(out, "r_id")
    blue_id = _normalized_id_series(out, "b_id")

    missing_id_mask = winner_id.isna() | red_id.isna() | blue_id.isna()
    if missing_id_mask.any():
        bad_count = int(missing_id_mask.sum())
        raise ValueError(
            "Cannot build rolling features because target ID fields contain missing values "
            f"in {bad_count} rows. Required columns: {TARGET_ID_COLUMNS}"
        )

    winner_matches_red = winner_id.eq(red_id)
    winner_matches_blue = winner_id.eq(blue_id)
    invalid_winner_mask = ~(winner_matches_red | winner_matches_blue)

    if invalid_winner_mask.any():
        bad_count = int(invalid_winner_mask.sum())
        example_columns = [
            column
            for column in ["event_name", "date", "fight_id", "r_name", "b_name", "winner", "winner_id", "r_id", "b_id"]
            if column in out.columns
        ]
        examples = out.loc[invalid_winner_mask, example_columns].head(10).to_dict("records")
        raise ValueError(
            "Cannot build rolling features because winner_id does not match r_id or b_id "
            f"in {bad_count} rows. Example rows: {examples}"
        )

    out["target"] = winner_matches_red.astype(int)
    return out


def prepare_master_for_rolling(master_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare master fight rows for chronological rolling feature generation."""
    df = master_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "event_id", "fight_id"]).reset_index(drop=True)

    # Canonical target: 1 means the red fighter won, based on stable UFCStats IDs.
    df = add_id_based_target(df)

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
    print(f"Target positive rate: {rolling_df['target'].mean():.4f}")

    validate_rolling_contract(rolling_df)

    rolling_df.to_parquet(ROLLING_FEATURES_PATH, index=False)
    print("Saved rolling features successfully.")
    print("DONE")


if __name__ == "__main__":
    main()
