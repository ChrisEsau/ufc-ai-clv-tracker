"""Normalize Betting Board market snapshots into side-level CLV rows."""

from __future__ import annotations

import pandas as pd

from pipeline.clv.utils import MONEYLINE, american_to_implied_prob, empty_frame, normalize_market_type

NORMALIZED_MARKET_COLUMNS = [
    "snapshot_timestamp",
    "event_name",
    "commence_time",
    "fight_id",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "sportsbook",
    "market_type",
    "american_odds",
    "implied_prob",
    "source_snapshot_run_id",
    "odds_match_type",
    "odds_match_score",
]


def _series(df: pd.DataFrame, column: str, default=""):
    if column in df.columns:
        return df[column]
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series([default] * len(df), index=df.index)


def _side_rows(snapshots: pd.DataFrame, side: str) -> pd.DataFrame:
    other = "blue" if side == "red" else "red"
    odds_col = f"{side}_american_odds"
    implied_col = f"{side}_implied_prob"
    fighter_col = f"{side}_fighter"
    opponent_col = f"{other}_fighter"
    fighter_id_col = f"{side}_fighter_id"
    opponent_id_col = f"{other}_fighter_id"

    rows = pd.DataFrame(index=snapshots.index)
    rows["snapshot_timestamp"] = _series(snapshots, "snapshot_timestamp")
    rows["event_name"] = _series(snapshots, "event_name")
    rows["commence_time"] = _series(snapshots, "commence_time")
    rows["fight_id"] = _series(snapshots, "fight_id")
    rows["fighter_id"] = _series(snapshots, fighter_id_col)
    rows["fighter_name"] = _series(snapshots, fighter_col)
    rows["opponent_id"] = _series(snapshots, opponent_id_col)
    rows["opponent_name"] = _series(snapshots, opponent_col)
    rows["sportsbook"] = _series(snapshots, "sportsbook", _series(snapshots, "bookmaker"))
    rows["market_type"] = _series(snapshots, "market_type", MONEYLINE).apply(normalize_market_type)
    rows["american_odds"] = pd.to_numeric(_series(snapshots, odds_col, pd.NA), errors="coerce")
    if implied_col in snapshots.columns:
        rows["implied_prob"] = pd.to_numeric(snapshots[implied_col], errors="coerce")
    else:
        rows["implied_prob"] = rows["american_odds"].apply(american_to_implied_prob)
    rows["source_snapshot_run_id"] = _series(snapshots, "snapshot_run_id", _series(snapshots, "prediction_run_id"))
    rows["odds_match_type"] = _series(snapshots, "odds_match_type")
    rows["odds_match_score"] = pd.to_numeric(_series(snapshots, "odds_match_score", pd.NA), errors="coerce")
    return rows


def normalize_market_snapshots(snapshots: pd.DataFrame | None) -> pd.DataFrame:
    """Return one side-level row per fighter/market snapshot.

    The active market updater writes a wide fight-level snapshot with red/blue
    odds.  CLV calculations need side-level rows keyed by fight, fighter,
    sportsbook, and market type.
    """

    if snapshots is None or snapshots.empty:
        return empty_frame(NORMALIZED_MARKET_COLUMNS)

    work = snapshots.copy()
    if {"red_american_odds", "blue_american_odds"}.issubset(work.columns):
        normalized = pd.concat([_side_rows(work, "red"), _side_rows(work, "blue")], ignore_index=True)
    elif {"fighter_id", "american_odds"}.issubset(work.columns):
        normalized = work.rename(columns={"bookmaker": "sportsbook"}).copy()
        for column in NORMALIZED_MARKET_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = pd.NA
        normalized = normalized[NORMALIZED_MARKET_COLUMNS]
    else:
        return empty_frame(NORMALIZED_MARKET_COLUMNS)

    normalized["snapshot_timestamp"] = pd.to_datetime(normalized["snapshot_timestamp"], utc=True, errors="coerce")
    normalized["commence_time"] = pd.to_datetime(normalized["commence_time"], utc=True, errors="coerce")
    normalized["american_odds"] = pd.to_numeric(normalized["american_odds"], errors="coerce")
    normalized["implied_prob"] = pd.to_numeric(normalized["implied_prob"], errors="coerce")
    normalized.loc[normalized["implied_prob"].isna(), "implied_prob"] = normalized.loc[
        normalized["implied_prob"].isna(), "american_odds"
    ].apply(american_to_implied_prob)
    normalized["market_type"] = normalized["market_type"].apply(normalize_market_type)

    required = ["snapshot_timestamp", "fight_id", "fighter_id", "american_odds"]
    normalized = normalized.dropna(subset=required).copy()
    normalized = normalized[NORMALIZED_MARKET_COLUMNS]
    return normalized.sort_values(["fight_id", "fighter_id", "sportsbook", "market_type", "snapshot_timestamp"]).reset_index(drop=True)
