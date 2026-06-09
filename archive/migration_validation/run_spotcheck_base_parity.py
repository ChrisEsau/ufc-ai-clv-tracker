"""Spot-check base fighter-state parity by fight_id.

This one-off archived script manually compares selected high-impact base-state
columns between the legacy rolling artifact and the new moneyline feature view.
It joins by fight_id instead of relying on row order.

Run from repo root:

    python archive/migration_validation/run_spotcheck_base_parity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LEGACY_ROLLING_PATH = Path("data/features/UFC_enhanced_rolling_features_EWM.parquet")
NEW_MONEYLINE_VIEW_PATH = Path("data/features/moneyline_feature_view.parquet")

SPOTCHECK_COLUMNS = [
    "r_pre_elo",
    "b_pre_elo",
    "r_pre_avg_opponent_elo",
    "b_pre_avg_opponent_elo",
    "r_pre_avg_fight_time",
    "b_pre_avg_fight_time",
    "r_pre_days_since_last_fight",
    "b_pre_days_since_last_fight",
    "r_pre_splm",
    "b_pre_splm",
    "r_pre_td_avg",
    "b_pre_td_avg",
    "r_pre_sub_avg",
    "b_pre_sub_avg",
]

TOLERANCE = 1e-9


def compare_column(old_df: pd.DataFrame, new_df: pd.DataFrame, column: str) -> None:
    """Print value parity details for one selected column."""

    merged = old_df[["fight_id", "event_name", "date", "r_name", "b_name", column]].merge(
        new_df[["fight_id", column]],
        on="fight_id",
        how="inner",
        suffixes=("_old", "_new"),
    )

    old_col = f"{column}_old"
    new_col = f"{column}_new"
    diff_col = "abs_diff"

    merged[diff_col] = (merged[old_col] - merged[new_col]).abs()
    nonzero = merged[merged[diff_col] > TOLERANCE]

    print("-" * 80)
    print(column)
    print(f"Rows compared : {len(merged)}")
    print(f"Max abs diff  : {merged[diff_col].max()}")
    print(f"Mean abs diff : {merged[diff_col].mean()}")
    print(f"Nonzero rows  : {len(nonzero)}")

    if not nonzero.empty:
        print("Worst rows:")
        print(
            nonzero.sort_values(diff_col, ascending=False)
            .head(10)[["fight_id", "event_name", "date", "r_name", "b_name", old_col, new_col, diff_col]]
            .to_string(index=False)
        )


def main() -> None:
    """Run manual spot checks for selected base-state columns."""

    print("=" * 80)
    print("BASE STATE SPOTCHECK BY FIGHT_ID")
    print("=" * 80)
    print(f"Legacy path: {LEGACY_ROLLING_PATH}")
    print(f"New path   : {NEW_MONEYLINE_VIEW_PATH}")

    old_df = pd.read_parquet(LEGACY_ROLLING_PATH)
    new_df = pd.read_parquet(NEW_MONEYLINE_VIEW_PATH)

    print(f"Legacy shape: {old_df.shape}")
    print(f"New shape   : {new_df.shape}")

    missing_old = [column for column in SPOTCHECK_COLUMNS if column not in old_df.columns]
    missing_new = [column for column in SPOTCHECK_COLUMNS if column not in new_df.columns]
    if missing_old or missing_new:
        raise ValueError(f"Missing columns. old={missing_old}, new={missing_new}")

    for column in SPOTCHECK_COLUMNS:
        compare_column(old_df, new_df, column)

    print("DONE")


if __name__ == "__main__":
    main()
