"""Build the moneyline feature view from fighter-state artifacts.

Run from repo root:

    python -m pipeline.features.run_build_moneyline_view

This runner is for parity inspection only at this stage. It does not replace the
existing rolling feature artifact used by training or prediction.
"""

from __future__ import annotations

import pandas as pd

from pipeline.common.paths import (
    FIGHTER_STATE_HISTORY_PATH,
    MASTER_PATH,
    MONEYLINE_FEATURE_VIEW_PATH,
    ensure_data_dirs,
)
from pipeline.features.run_build_rolling_features import prepare_master_for_rolling
from pipeline.features.views.moneyline import build_moneyline_feature_view


def main() -> None:
    """Build and save the moneyline feature view artifact."""

    ensure_data_dirs()

    print("=" * 80)
    print("BUILD UFC MONEYLINE FEATURE VIEW")
    print("=" * 80)
    print(f"Master path             : {MASTER_PATH}")
    print(f"Fighter state path      : {FIGHTER_STATE_HISTORY_PATH}")
    print(f"Moneyline view path     : {MONEYLINE_FEATURE_VIEW_PATH}")

    master_df = pd.read_parquet(MASTER_PATH)
    print(f"Master shape            : {master_df.shape}")

    prepared_df = prepare_master_for_rolling(master_df)
    print(f"Prepared fight shape    : {prepared_df.shape}")

    fighter_state_history_df = pd.read_parquet(FIGHTER_STATE_HISTORY_PATH)
    print(f"Fighter state shape     : {fighter_state_history_df.shape}")

    moneyline_df = build_moneyline_feature_view(
        prepared_fights_df=prepared_df,
        fighter_state_history_df=fighter_state_history_df,
    )

    print(f"Moneyline view shape    : {moneyline_df.shape}")
    print(f"Unique fights           : {moneyline_df['fight_id'].nunique() if not moneyline_df.empty else 0}")

    if len(moneyline_df) != len(prepared_df):
        raise ValueError(
            "Moneyline feature view row mismatch: "
            f"expected {len(prepared_df)}, observed {len(moneyline_df)}"
        )

    missing_state_match_count = int(
        moneyline_df[["r_pre_elo", "b_pre_elo"]].isna().any(axis=1).sum()
        if {"r_pre_elo", "b_pre_elo"}.issubset(moneyline_df.columns)
        else len(moneyline_df)
    )
    print(f"Missing state matches   : {missing_state_match_count}")

    if missing_state_match_count:
        raise ValueError(
            "Moneyline feature view has missing fighter-state matches. "
            f"Rows affected: {missing_state_match_count}"
        )

    moneyline_df.to_parquet(MONEYLINE_FEATURE_VIEW_PATH, index=False)

    print("Saved moneyline feature view successfully.")
    print("DONE")


if __name__ == "__main__":
    main()
