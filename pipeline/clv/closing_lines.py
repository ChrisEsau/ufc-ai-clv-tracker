"""Closing-line artifact builder."""

from __future__ import annotations

import pandas as pd

from pipeline.clv.market_normalization import NORMALIZED_MARKET_COLUMNS
from pipeline.clv.utils import empty_frame

CLOSING_LINE_COLUMNS = [
    "event_name",
    "fight_id",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "sportsbook",
    "market_type",
    "commence_time",
    "closing_timestamp",
    "closing_odds",
    "closing_implied_prob",
    "minutes_before_fight",
    "is_true_closing_window",
    "closing_line_status",
]


def build_closing_lines(normalized_snapshots: pd.DataFrame | None) -> pd.DataFrame:
    """Extract latest available pre-fight line for each fighter/book/market."""

    if normalized_snapshots is None or normalized_snapshots.empty:
        return empty_frame(CLOSING_LINE_COLUMNS)

    snapshots = normalized_snapshots.copy()
    for column in NORMALIZED_MARKET_COLUMNS:
        if column not in snapshots.columns:
            snapshots[column] = pd.NA

    snapshots["snapshot_timestamp"] = pd.to_datetime(snapshots["snapshot_timestamp"], utc=True, errors="coerce")
    snapshots["commence_time"] = pd.to_datetime(snapshots["commence_time"], utc=True, errors="coerce")
    snapshots = snapshots.dropna(subset=["snapshot_timestamp", "fight_id", "fighter_id", "american_odds"]).copy()
    if snapshots.empty:
        return empty_frame(CLOSING_LINE_COLUMNS)

    pre_fight = snapshots[
        snapshots["commence_time"].isna() | (snapshots["snapshot_timestamp"] <= snapshots["commence_time"])
    ].copy()
    if pre_fight.empty:
        pre_fight = snapshots.copy()

    group_cols = ["fight_id", "fighter_id", "sportsbook", "market_type"]
    closing = (
        pre_fight.sort_values("snapshot_timestamp")
        .groupby(group_cols, dropna=False)
        .tail(1)
        .copy()
    )
    closing = closing.rename(
        columns={
            "snapshot_timestamp": "closing_timestamp",
            "american_odds": "closing_odds",
            "implied_prob": "closing_implied_prob",
        }
    )

    closing["minutes_before_fight"] = (
        closing["commence_time"] - closing["closing_timestamp"]
    ).dt.total_seconds() / 60
    closing["is_true_closing_window"] = closing["minutes_before_fight"].le(60).fillna(False)
    closing["closing_line_status"] = closing["is_true_closing_window"].map(
        {True: "true_close_window", False: "latest_pre_fight_snapshot"}
    )

    for column in CLOSING_LINE_COLUMNS:
        if column not in closing.columns:
            closing[column] = pd.NA
    return closing[CLOSING_LINE_COLUMNS].sort_values(["event_name", "fight_id", "fighter_name", "sportsbook"]).reset_index(drop=True)
