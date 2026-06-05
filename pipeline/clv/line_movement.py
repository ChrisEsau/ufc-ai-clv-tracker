"""Line-movement artifact builder."""

from __future__ import annotations

import pandas as pd

from pipeline.clv.market_normalization import NORMALIZED_MARKET_COLUMNS
from pipeline.clv.utils import empty_frame

LINE_MOVEMENT_COLUMNS = [
    "event_name",
    "fight_id",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "sportsbook",
    "market_type",
    "opening_timestamp",
    "latest_timestamp",
    "opening_odds",
    "latest_odds",
    "opening_implied_prob",
    "latest_implied_prob",
    "odds_movement",
    "implied_prob_movement",
    "is_steam_move",
    "steam_direction",
]

STEAM_THRESHOLD = 0.05


def build_line_movement(normalized_snapshots: pd.DataFrame | None) -> pd.DataFrame:
    """Build opening/latest line movement from side-level market snapshots."""

    if normalized_snapshots is None or normalized_snapshots.empty:
        return empty_frame(LINE_MOVEMENT_COLUMNS)

    snapshots = normalized_snapshots.copy()
    for column in NORMALIZED_MARKET_COLUMNS:
        if column not in snapshots.columns:
            snapshots[column] = pd.NA
    snapshots["snapshot_timestamp"] = pd.to_datetime(snapshots["snapshot_timestamp"], utc=True, errors="coerce")
    snapshots = snapshots.dropna(subset=["snapshot_timestamp", "fight_id", "fighter_id", "american_odds"]).copy()
    if snapshots.empty:
        return empty_frame(LINE_MOVEMENT_COLUMNS)

    group_cols = ["fight_id", "fighter_id", "sportsbook", "market_type"]
    ordered = snapshots.sort_values("snapshot_timestamp")
    opening = ordered.groupby(group_cols, dropna=False).head(1).copy()
    latest = ordered.groupby(group_cols, dropna=False).tail(1).copy()

    opening = opening.rename(
        columns={
            "snapshot_timestamp": "opening_timestamp",
            "american_odds": "opening_odds",
            "implied_prob": "opening_implied_prob",
        }
    )
    latest = latest.rename(
        columns={
            "snapshot_timestamp": "latest_timestamp",
            "american_odds": "latest_odds",
            "implied_prob": "latest_implied_prob",
        }
    )

    keep_latest = [*group_cols, "latest_timestamp", "latest_odds", "latest_implied_prob"]
    movement = opening.merge(latest[keep_latest], how="inner", on=group_cols)
    movement["odds_movement"] = movement["latest_odds"] - movement["opening_odds"]
    movement["implied_prob_movement"] = movement["latest_implied_prob"] - movement["opening_implied_prob"]
    movement["is_steam_move"] = movement["implied_prob_movement"].abs() >= STEAM_THRESHOLD
    movement["steam_direction"] = "neutral"
    movement.loc[movement["implied_prob_movement"] > 0, "steam_direction"] = "toward_fighter"
    movement.loc[movement["implied_prob_movement"] < 0, "steam_direction"] = "against_fighter"

    for column in LINE_MOVEMENT_COLUMNS:
        if column not in movement.columns:
            movement[column] = pd.NA
    return movement[LINE_MOVEMENT_COLUMNS].sort_values(["event_name", "fight_id", "fighter_name", "sportsbook"]).reset_index(drop=True)
