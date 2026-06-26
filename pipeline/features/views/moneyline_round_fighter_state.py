"""Experimental moneyline feature view with Round Fighter State features.

This module intentionally wraps the existing production moneyline feature view
instead of modifying it. Use this for offline experiments only until RFS proves
value in training/backtests.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.common.paths import ROUND_FIGHTER_STATE_HISTORY_PATH
from pipeline.features.views.moneyline import build_moneyline_feature_view
from pipeline.round_stats.join_round_fighter_state import (
    join_round_fighter_state_history,
    summarize_rfs_join,
)


def build_moneyline_feature_view_with_round_state(
    prepared_fights_df: pd.DataFrame,
    fighter_state_history_df: pd.DataFrame,
    round_state_history_path: str | Path = ROUND_FIGHTER_STATE_HISTORY_PATH,
    add_round_state_diffs: bool = True,
) -> pd.DataFrame:
    """Build baseline moneyline view, then append leakage-safe RFS history.

    Required prepared fight columns follow the production moneyline view:
    - fight_id
    - r_id
    - b_id

    RFS history is joined by fight_id + fighter_id, so historical rows receive
    only the RFS state available entering that fight.
    """
    base_view = build_moneyline_feature_view(
        prepared_fights_df=prepared_fights_df,
        fighter_state_history_df=fighter_state_history_df,
    )

    return join_round_fighter_state_history(
        base_view,
        history_path=round_state_history_path,
        red_fighter_id_col="r_id",
        blue_fighter_id_col="b_id",
        fight_id_col="fight_id",
        add_diffs=add_round_state_diffs,
    )


def summarize_moneyline_round_state_view(
    feature_view_df: pd.DataFrame,
) -> dict[str, int | float]:
    """Return row/column counts plus RFS completeness summary."""
    summary = summarize_rfs_join(feature_view_df)
    summary["row_count"] = int(len(feature_view_df))
    summary["column_count"] = int(len(feature_view_df.columns))
    return summary
