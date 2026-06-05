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


def _select_closing_row(group: pd.DataFrame) -> pd.Series:
    """Select the most reliable closing row for one fighter/book/market group."""

    ordered = group.sort_values("snapshot_timestamp").copy()
    has_commence = ordered["commence_time"].notna()
    if has_commence.any():
        pre_fight = ordered[ordered["snapshot_timestamp"] <= ordered["commence_time"]]
        if not pre_fight.empty:
            selected = pre_fight.iloc[-1].copy()
            minutes = (selected["commence_time"] - selected["snapshot_timestamp"]).total_seconds() / 60
            selected["minutes_before_fight"] = minutes
            selected["is_true_closing_window"] = bool(0 <= minutes <= 60)
            selected["closing_line_status"] = "true_close_window" if selected["is_true_closing_window"] else "latest_pre_fight_snapshot"
            return selected

        selected = ordered.iloc[-1].copy()
        commence_time = ordered["commence_time"].dropna().iloc[-1]
        selected["commence_time"] = commence_time
        selected["minutes_before_fight"] = (commence_time - selected["snapshot_timestamp"]).total_seconds() / 60
        selected["is_true_closing_window"] = False
        selected["closing_line_status"] = "post_fight_fallback"
        return selected

    selected = ordered.iloc[-1].copy()
    selected["minutes_before_fight"] = pd.NA
    selected["is_true_closing_window"] = False
    selected["closing_line_status"] = "missing_commence_time"
    return selected


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

    group_cols = ["fight_id", "fighter_id", "sportsbook", "market_type"]
    closing = pd.DataFrame([_select_closing_row(group) for _, group in snapshots.groupby(group_cols, dropna=False)])
    closing = closing.rename(
        columns={
            "snapshot_timestamp": "closing_timestamp",
            "american_odds": "closing_odds",
            "implied_prob": "closing_implied_prob",
        }
    )

    for column in CLOSING_LINE_COLUMNS:
        if column not in closing.columns:
            closing[column] = pd.NA
    return closing[CLOSING_LINE_COLUMNS].sort_values(["event_name", "fight_id", "fighter_name", "sportsbook"]).reset_index(drop=True)
