"""Experimental moneyline feature view with full-family Round Fighter State.

This module intentionally wraps the existing production moneyline feature view
instead of modifying it. Use this for offline experiments only until RFS proves
value in training/backtests.

Current behavior:
- Builds the existing moneyline feature view.
- Appends point-in-time RFS history families.
- Excludes current-fight RFS observations by default.
- Emits RFS matchup diffs plus availability flags by default.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.features.views.moneyline import build_moneyline_feature_view
from pipeline.round_stats.join_round_fighter_state_families import (
    DEFAULT_RFS_FAMILY_CONFIGS,
    RfsFamilyConfig,
    join_round_fighter_state_families_history,
    summarize_rfs_family_join,
)


def build_moneyline_feature_view_with_round_state(
    prepared_fights_df: pd.DataFrame,
    fighter_state_history_df: pd.DataFrame,
    family_configs: tuple[RfsFamilyConfig, ...] = DEFAULT_RFS_FAMILY_CONFIGS,
    add_round_state_diffs: bool = True,
    include_fight_observations: bool = False,
    keep_side_features: bool = False,
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

    # For symmetric flipped rows, fight_id becomes "<base_fight_id>__flip" for
    # row uniqueness and market/output traceability. RFS history is keyed to the
    # original fight id, so use state_fight_id when present.
    rfs_join_fight_id_col = "state_fight_id" if "state_fight_id" in base_view.columns else "fight_id"

    return join_round_fighter_state_families_history(
        base_view,
        family_configs=family_configs,
        red_fighter_id_col="r_id",
        blue_fighter_id_col="b_id",
        fight_id_col=rfs_join_fight_id_col,
        add_diffs=add_round_state_diffs,
        include_fight_observations=include_fight_observations,
        keep_side_features=keep_side_features,
    )


def summarize_moneyline_round_state_view(
    feature_view_df: pd.DataFrame,
) -> dict[str, int | float]:
    """Return row/column counts plus RFS completeness summary."""
    summary = summarize_rfs_family_join(feature_view_df)
    summary["row_count"] = int(len(feature_view_df))
    summary["column_count"] = int(len(feature_view_df.columns))
    return summary
