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
    normalized["sportsbook"] = normalized["sportsbook"].replace("", pd.NA)

    required = ["snapshot_timestamp", "fight_id", "fighter_id", "american_odds"]
    normalized = normalized.dropna(subset=required).copy()
    normalized = normalized[NORMALIZED_MARKET_COLUMNS]
    return normalized.sort_values(["fight_id", "fighter_id", "sportsbook", "market_type", "snapshot_timestamp"]).reset_index(drop=True)


MARKET_NORMALIZATION_AUDIT_COLUMNS = [
    "input_rows",
    "normalized_rows",
    "dropped_rows",
    "unique_events",
    "unique_fights",
    "unique_fighters",
    "unique_sportsbooks",
    "missing_snapshot_timestamp",
    "missing_commence_time",
    "missing_fight_id",
    "missing_fighter_id",
    "missing_sportsbook",
    "missing_american_odds",
    "invalid_american_odds",
    "duplicate_side_snapshots",
    "post_commence_snapshots",
    "audit_status",
]


def build_market_normalization_audit(raw_snapshots: pd.DataFrame | None, normalized_snapshots: pd.DataFrame | None) -> pd.DataFrame:
    """Summarize CLV market-normalization coverage and quality.

    The audit is intentionally one row so the dashboard/workflows can quickly
    tell whether CLV inputs are usable without scanning large parquet files.
    """

    raw_rows = 0 if raw_snapshots is None else len(raw_snapshots)
    normalized = empty_frame(NORMALIZED_MARKET_COLUMNS) if normalized_snapshots is None else normalized_snapshots.copy()
    normalized_rows = len(normalized)

    if normalized.empty:
        row = {
            "input_rows": raw_rows,
            "normalized_rows": 0,
            "dropped_rows": raw_rows,
            "unique_events": 0,
            "unique_fights": 0,
            "unique_fighters": 0,
            "unique_sportsbooks": 0,
            "missing_snapshot_timestamp": None,
            "missing_commence_time": None,
            "missing_fight_id": None,
            "missing_fighter_id": None,
            "missing_sportsbook": None,
            "missing_american_odds": None,
            "invalid_american_odds": None,
            "duplicate_side_snapshots": None,
            "post_commence_snapshots": None,
            "audit_status": "empty",
        }
        return pd.DataFrame([row], columns=MARKET_NORMALIZATION_AUDIT_COLUMNS)

    work = normalized.copy()
    work["snapshot_timestamp"] = pd.to_datetime(work["snapshot_timestamp"], utc=True, errors="coerce")
    work["commence_time"] = pd.to_datetime(work["commence_time"], utc=True, errors="coerce")
    work["american_odds"] = pd.to_numeric(work["american_odds"], errors="coerce")

    duplicate_keys = ["snapshot_timestamp", "fight_id", "fighter_id", "sportsbook", "market_type"]
    duplicate_side_snapshots = int(work.duplicated(subset=duplicate_keys, keep=False).sum())
    post_commence = int(((work["commence_time"].notna()) & (work["snapshot_timestamp"] > work["commence_time"])).sum())
    missing_required = int(
        work[["snapshot_timestamp", "fight_id", "fighter_id", "sportsbook", "american_odds"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    audit_status = "ready" if missing_required == 0 and normalized_rows > 0 else "warning"

    expected_rows = raw_rows * 2 if raw_snapshots is not None and {"red_american_odds", "blue_american_odds"}.issubset(raw_snapshots.columns) else raw_rows

    row = {
        "input_rows": raw_rows,
        "normalized_rows": normalized_rows,
        "dropped_rows": max(expected_rows - normalized_rows, 0) if raw_rows else 0,
        "unique_events": int(work["event_name"].dropna().nunique()) if "event_name" in work else 0,
        "unique_fights": int(work["fight_id"].dropna().nunique()),
        "unique_fighters": int(work["fighter_id"].dropna().nunique()),
        "unique_sportsbooks": int(work["sportsbook"].dropna().nunique()),
        "missing_snapshot_timestamp": int(work["snapshot_timestamp"].isna().sum()),
        "missing_commence_time": int(work["commence_time"].isna().sum()),
        "missing_fight_id": int(work["fight_id"].isna().sum()),
        "missing_fighter_id": int(work["fighter_id"].isna().sum()),
        "missing_sportsbook": int(work["sportsbook"].isna().sum()),
        "missing_american_odds": int(work["american_odds"].isna().sum()),
        "invalid_american_odds": int((work["american_odds"] == 0).sum()),
        "duplicate_side_snapshots": duplicate_side_snapshots,
        "post_commence_snapshots": post_commence,
        "audit_status": audit_status,
    }
    return pd.DataFrame([row], columns=MARKET_NORMALIZATION_AUDIT_COLUMNS)
