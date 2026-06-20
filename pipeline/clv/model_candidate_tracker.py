from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


CANDIDATE_RULE_VERSION = "v1_is_bet_candidate"

CANDIDATE_TRACKER_COLUMNS = [
    "candidate_id",
    "candidate_rule_version",
    "candidate_status",
    "candidate_timestamp",
    "candidate_source",
    "capture_run_id",
    "snapshot_model_mode",
    "model_id",
    "model_family",
    "model_stage",
    "algorithm",
    "prediction_type",
    "event_id",
    "event_name",
    "commence_time",
    "fight_id",
    "fight_display",
    "red_fighter",
    "blue_fighter",
    "market_key",
    "market_display",
    "bookmaker",
    "source",
    "outcome_label",
    "outcome_display",
    "outcome_fighter_id",
    "outcome_join_key",
    "outcome_side",
    "candidate_odds",
    "candidate_decimal_odds",
    "candidate_implied_probability",
    "candidate_model_probability",
    "candidate_model_pick_probability",
    "candidate_is_model_pick",
    "candidate_model_pick",
    "candidate_confidence",
    "candidate_confidence_score",
    "candidate_confidence_pct",
    "candidate_confidence_tier",
    "candidate_edge",
    "candidate_edge_pct",
    "candidate_ev",
    "candidate_ev_pct",
    "candidate_ev_dollars_at_100",
    "candidate_full_kelly_fraction",
    "candidate_fractional_kelly_fraction",
    "candidate_recommended_stake",
    "candidate_max_stake",
    "passes_edge_filter",
    "passes_confidence_filter",
    "passes_odds_filter",
    "passes_market_data_filter",
    "is_bet_candidate",
    "bet_status",
]


@dataclass(frozen=True)
class CandidateRule:
    rule_version: str = CANDIDATE_RULE_VERSION
    min_edge: float | None = None
    min_confidence_pct: float | None = None
    require_market_filter: bool = True
    use_is_bet_candidate: bool = True


def empty_candidate_tracker() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_TRACKER_COLUMNS)


def _series(df: pd.DataFrame, column: str, default=pd.NA) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _numeric(df: pd.DataFrame, column: str, default=pd.NA) -> pd.Series:
    return pd.to_numeric(_series(df, column, default), errors="coerce")


def _bool_series(df: pd.DataFrame, column: str, default=False) -> pd.Series:
    values = _series(df, column, default)
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _stable_candidate_id(row: pd.Series) -> str:
    parts = [
        row.get("candidate_rule_version", ""),
        row.get("model_id", ""),
        row.get("market_key", ""),
        row.get("fight_id", ""),
        row.get("outcome_join_key", ""),
        row.get("outcome_fighter_id", ""),
        row.get("bookmaker", ""),
    ]
    raw = "|".join("" if pd.isna(part) else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _candidate_mask(snapshots: pd.DataFrame, rule: CandidateRule) -> pd.Series:
    mask = pd.Series(True, index=snapshots.index)

    if rule.use_is_bet_candidate and "is_bet_candidate" in snapshots.columns:
        mask &= _bool_series(snapshots, "is_bet_candidate")
    else:
        if rule.min_edge is not None:
            mask &= _numeric(snapshots, "edge") >= float(rule.min_edge)
        if rule.min_confidence_pct is not None:
            confidence = _numeric(snapshots, "confidence_pct")
            mask &= confidence >= float(rule.min_confidence_pct)
        if rule.require_market_filter and "passes_market_data_filter" in snapshots.columns:
            mask &= _bool_series(snapshots, "passes_market_data_filter")

    required = ["model_id", "fight_id", "market_key", "american_odds"]
    for column in required:
        if column in snapshots.columns:
            mask &= _series(snapshots, column).notna()
        else:
            mask &= False
    return mask.fillna(False)


def _candidate_rows(snapshots: pd.DataFrame, rule: CandidateRule) -> pd.DataFrame:
    if snapshots is None or snapshots.empty:
        return empty_candidate_tracker()

    work = snapshots.copy()
    work = work[_candidate_mask(work, rule)].copy()
    if work.empty:
        return empty_candidate_tracker()

    rows = pd.DataFrame(index=work.index)
    rows["candidate_rule_version"] = rule.rule_version
    rows["candidate_status"] = "active"
    rows["candidate_timestamp"] = _series(work, "market_snapshot_timestamp", _series(work, "capture_timestamp"))
    rows["candidate_source"] = _series(work, "snapshot_source", "market_refresh_orchestrator")
    rows["capture_run_id"] = _series(work, "capture_run_id")
    rows["snapshot_model_mode"] = _series(work, "snapshot_model_mode")
    rows["model_id"] = _series(work, "model_id")
    rows["model_family"] = _series(work, "model_family")
    rows["model_stage"] = _series(work, "model_stage", _series(work, "model_registry_status"))
    rows["algorithm"] = _series(work, "algorithm")
    rows["prediction_type"] = _series(work, "prediction_type")
    rows["event_id"] = _series(work, "event_id")
    rows["event_name"] = _series(work, "event_name")
    rows["commence_time"] = _series(work, "commence_time")
    rows["fight_id"] = _series(work, "fight_id")
    rows["fight_display"] = _series(work, "fight_display")
    rows["red_fighter"] = _series(work, "red_fighter")
    rows["blue_fighter"] = _series(work, "blue_fighter")
    rows["market_key"] = _series(work, "market_key")
    rows["market_display"] = _series(work, "market_display")
    rows["bookmaker"] = _series(work, "bookmaker")
    rows["source"] = _series(work, "source")
    rows["outcome_label"] = _series(work, "outcome_label")
    rows["outcome_display"] = _series(work, "outcome_display")
    rows["outcome_fighter_id"] = _series(work, "outcome_fighter_id")
    rows["outcome_join_key"] = _series(work, "outcome_join_key")
    rows["outcome_side"] = _series(work, "outcome_side")
    rows["candidate_odds"] = _numeric(work, "american_odds")
    rows["candidate_decimal_odds"] = _numeric(work, "decimal_odds")
    rows["candidate_implied_probability"] = _numeric(work, "implied_probability")
    rows["candidate_model_probability"] = _numeric(work, "model_probability")
    rows["candidate_model_pick_probability"] = _numeric(work, "model_pick_probability")
    rows["candidate_is_model_pick"] = _bool_series(work, "is_model_pick")
    rows["candidate_model_pick"] = _series(work, "model_pick")
    rows["candidate_confidence"] = _numeric(work, "model_confidence")
    rows["candidate_confidence_score"] = _numeric(work, "confidence_score")
    rows["candidate_confidence_pct"] = _numeric(work, "confidence_pct")
    rows["candidate_confidence_tier"] = _series(work, "confidence_tier")
    rows["candidate_edge"] = _numeric(work, "edge")
    rows["candidate_edge_pct"] = _numeric(work, "edge_pct")
    rows["candidate_ev"] = _numeric(work, "ev")
    rows["candidate_ev_pct"] = _numeric(work, "ev_pct")
    rows["candidate_ev_dollars_at_100"] = _numeric(work, "ev_dollars_at_100")
    rows["candidate_full_kelly_fraction"] = _numeric(work, "full_kelly_fraction")
    rows["candidate_fractional_kelly_fraction"] = _numeric(work, "fractional_kelly_fraction")
    rows["candidate_recommended_stake"] = _numeric(work, "recommended_stake")
    rows["candidate_max_stake"] = _numeric(work, "max_stake")
    rows["passes_edge_filter"] = _bool_series(work, "passes_edge_filter")
    rows["passes_confidence_filter"] = _bool_series(work, "passes_confidence_filter")
    rows["passes_odds_filter"] = _bool_series(work, "passes_odds_filter")
    rows["passes_market_data_filter"] = _bool_series(work, "passes_market_data_filter")
    rows["is_bet_candidate"] = _bool_series(work, "is_bet_candidate")
    rows["bet_status"] = _series(work, "bet_status")

    rows["candidate_timestamp"] = pd.to_datetime(rows["candidate_timestamp"], utc=True, errors="coerce")
    rows["commence_time"] = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["candidate_timestamp", "model_id", "fight_id", "market_key", "candidate_odds"]).copy()
    if rows.empty:
        return empty_candidate_tracker()

    rows["candidate_id"] = rows.apply(_stable_candidate_id, axis=1)
    return rows[CANDIDATE_TRACKER_COLUMNS]


def build_model_candidate_tracker(
    snapshots: pd.DataFrame | None,
    existing_candidates: pd.DataFrame | None = None,
    rule: CandidateRule | None = None,
) -> pd.DataFrame:
    """Append first-qualifying model candidates while preserving original odds/time.

    First signal wins per candidate_id. Later snapshots can create new candidates
    only when model, market, fight, outcome/book, or rule version differ.
    """

    rule = rule or CandidateRule()
    existing = empty_candidate_tracker() if existing_candidates is None or existing_candidates.empty else existing_candidates.copy()
    for column in CANDIDATE_TRACKER_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA
    existing = existing[CANDIDATE_TRACKER_COLUMNS]

    new_rows = _candidate_rows(snapshots if snapshots is not None else pd.DataFrame(), rule)
    if new_rows.empty:
        return existing.reset_index(drop=True)

    existing_ids = set(existing["candidate_id"].dropna().astype(str)) if not existing.empty else set()
    append_rows = new_rows[~new_rows["candidate_id"].astype(str).isin(existing_ids)].copy()
    combined = pd.concat([existing, append_rows], ignore_index=True, sort=False) if not append_rows.empty else existing
    combined["candidate_timestamp"] = pd.to_datetime(combined["candidate_timestamp"], utc=True, errors="coerce")
    combined = combined.sort_values(["candidate_timestamp", "model_id", "fight_id"], na_position="last")
    return combined[CANDIDATE_TRACKER_COLUMNS].reset_index(drop=True)


def count_new_candidates(before: pd.DataFrame | None, after: pd.DataFrame | None) -> int:
    before_ids = set() if before is None or before.empty else set(before.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str))
    after_ids = set() if after is None or after.empty else set(after.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str))
    return len(after_ids - before_ids)
