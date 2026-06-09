"""Production runner for fighter-state artifact generation.

Run from repo root:

    python -m pipeline.features.run_build_fighter_state

Pipeline:
- Load data/master/ufc_master.parquet
- Reuse rolling feature preparation for date sorting and ID-based targets
- Build fighter-level prefight state history with raw fighter feature plugins
- Add whitelisted fighter-level EWM state features through the EWM plugin adapter
- Derive latest fighter state
- Write data/features/fighter_state_history.parquet
- Write data/features/latest_fighter_state.parquet
"""

from __future__ import annotations

import pandas as pd

from pipeline.common.paths import (
    FIGHTER_STATE_HISTORY_PATH,
    LATEST_FIGHTER_STATE_PATH,
    MASTER_PATH,
    ensure_data_dirs,
)
from pipeline.features.run_build_rolling_features import prepare_master_for_rolling
from pipeline.features.state.ewm_state import get_ewm_state_columns
from pipeline.features.state.history_builder import build_latest_fighter_state
from pipeline.features.state.plugin_history_builder import build_plugin_fighter_state_history


def main() -> None:
    """Build fighter-state history and latest-state artifacts."""

    ensure_data_dirs()

    print("=" * 80)
    print("BUILD UFC FIGHTER STATE ARTIFACTS")
    print("=" * 80)
    print(f"Master path        : {MASTER_PATH}")
    print(f"History output path: {FIGHTER_STATE_HISTORY_PATH}")
    print(f"Latest output path : {LATEST_FIGHTER_STATE_PATH}")
    print("Builder            : plugin_history_builder")

    master_df = pd.read_parquet(MASTER_PATH)
    print(f"Master shape       : {master_df.shape}")

    prepared_df = prepare_master_for_rolling(master_df)
    print(f"Prepared shape     : {prepared_df.shape}")

    base_history_df = build_plugin_fighter_state_history(prepared_df, add_ewm=False)
    ewm_source_columns = get_ewm_state_columns(base_history_df)
    history_df = build_plugin_fighter_state_history(prepared_df, add_ewm=True)
    latest_df = build_latest_fighter_state(history_df)

    expected_history_rows = len(prepared_df) * 2
    print(f"History shape      : {history_df.shape}")
    print(f"Expected rows      : {expected_history_rows}")
    print(f"EWM source columns : {len(ewm_source_columns)}")
    print(f"EWM added columns  : {len(ewm_source_columns) * 2}")
    print(f"Latest shape       : {latest_df.shape}")
    print(f"Unique fighters    : {history_df['fighter_id'].nunique() if not history_df.empty else 0}")

    if len(history_df) != expected_history_rows:
        raise ValueError(
            "Fighter-state history row mismatch: "
            f"expected {expected_history_rows}, observed {len(history_df)}"
        )

    history_df.to_parquet(FIGHTER_STATE_HISTORY_PATH, index=False)
    latest_df.to_parquet(LATEST_FIGHTER_STATE_PATH, index=False)

    print("Saved fighter-state artifacts successfully.")
    print("DONE")


if __name__ == "__main__":
    main()
