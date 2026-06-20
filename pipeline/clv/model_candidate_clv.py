from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.clv.closing_lines import CLOSING_LINE_COLUMNS
from pipeline.clv.model_candidate_tracker import CANDIDATE_TRACKER_COLUMNS, empty_candidate_tracker
from pipeline.clv.utils import clv_pct


MODEL_CANDIDATE_CLV_COLUMNS = [
    "candidate_id",
    "candidate_rule_version",
    "candidate_status",
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
    "market_key",
    "market_display",
    "bookmaker",
    "outcome_label",
    "outcome_display",
    "outcome_fighter_id",
    "outcome_join_key",
    "outcome_side",
    "candidate_timestamp",
    "candidate_odds",
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
    "candidate_recommended_stake",
    "closing_timestamp",
    "closing_odds",
    "closing_implied_prob",
    "closing_line_status",
    "hours_before_fight",
    "hours_before_close",
    "clv_pct",
    "clv_implied_prob_delta",
    "beat_closing_line",
    "candidate_clv_status",
]


def empty_candidate_clv() -> pd.DataFrame:
    return pd.DataFrame(columns=MODEL_CANDIDATE_CLV_COLUMNS)


def _prepare_candidates(candidates: pd.DataFrame | None) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return empty_candidate_tracker()
    out = candidates.copy()
    for column in CANDIDATE_TRACKER_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["candidate_timestamp"] = pd.to_datetime(out["candidate_timestamp"], utc=True, errors="coerce")
    out["commence_time"] = pd.to_datetime(out["commence_time"], utc=True, errors="coerce")
    out["candidate_odds"] = pd.to_numeric(out["candidate_odds"], errors="coerce")
    out["candidate_implied_probability"] = pd.to_numeric(out["candidate_implied_probability"], errors="coerce")
    return out[CANDIDATE_TRACKER_COLUMNS]


def _prepare_closing(closing_lines: pd.DataFrame | None) -> pd.DataFrame:
    if closing_lines is None or closing_lines.empty:
        return pd.DataFrame()
    closing = closing_lines.copy()
    for column in CLOSING_LINE_COLUMNS:
        if column not in closing.columns:
            closing[column] = pd.NA
    closing["closing_timestamp"] = pd.to_datetime(closing["closing_timestamp"], utc=True, errors="coerce")
    closing["commence_time"] = pd.to_datetime(closing["commence_time"], utc=True, errors="coerce")
    closing["closing_odds"] = pd.to_numeric(closing["closing_odds"], errors="coerce")
    closing["closing_implied_prob"] = pd.to_numeric(closing["closing_implied_prob"], errors="coerce")
    return closing


def _candidate_market_type(candidates: pd.DataFrame) -> pd.Series:
    return candidates.get("market_key", pd.Series(index=candidates.index, dtype=object)).astype(str).str.lower().replace({"": pd.NA})


def _closing_market_type(closing: pd.DataFrame) -> pd.Series:
    return closing.get("market_type", pd.Series(index=closing.index, dtype=object)).astype(str).str.lower().replace({"": pd.NA})


def build_model_candidate_clv(candidates: pd.DataFrame | None, closing_lines: pd.DataFrame | None) -> pd.DataFrame:
    """Join frozen model candidates to closing lines and calculate candidate CLV."""

    candidate_rows = _prepare_candidates(candidates)
    if candidate_rows.empty:
        return empty_candidate_clv()

    closing = _prepare_closing(closing_lines)
    if closing.empty:
        results = candidate_rows.copy()
        for column in ["closing_timestamp", "closing_odds", "closing_implied_prob", "closing_line_status"]:
            results[column] = pd.NA
    else:
        candidates_work = candidate_rows.copy()
        candidates_work["join_market_type"] = _candidate_market_type(candidates_work)
        candidates_work["join_fighter_id"] = candidates_work["outcome_fighter_id"].astype(str)
        candidates_work["join_sportsbook"] = candidates_work["bookmaker"].astype(str)

        closing_work = closing.copy()
        closing_work["join_market_type"] = _closing_market_type(closing_work)
        closing_work["join_fighter_id"] = closing_work["fighter_id"].astype(str)
        closing_work["join_sportsbook"] = closing_work["sportsbook"].astype(str)

        exact_cols = [
            "fight_id",
            "join_fighter_id",
            "join_market_type",
            "join_sportsbook",
            "closing_timestamp",
            "closing_odds",
            "closing_implied_prob",
            "closing_line_status",
        ]
        exact = candidates_work.merge(
            closing_work[exact_cols],
            how="left",
            on=["fight_id", "join_fighter_id", "join_market_type", "join_sportsbook"],
        )

        fallback_closing = (
            closing_work.sort_values("closing_timestamp")
            .groupby(["fight_id", "join_fighter_id", "join_market_type"], dropna=False)
            .tail(1)
        )
        fallback_cols = [
            "fight_id",
            "join_fighter_id",
            "join_market_type",
            "closing_timestamp",
            "closing_odds",
            "closing_implied_prob",
            "closing_line_status",
        ]
        unmatched = exact["closing_odds"].isna()
        if unmatched.any():
            fallback_values = exact.loc[unmatched].drop(
                columns=["closing_timestamp", "closing_odds", "closing_implied_prob", "closing_line_status"],
                errors="ignore",
            ).merge(
                fallback_closing[fallback_cols],
                how="left",
                on=["fight_id", "join_fighter_id", "join_market_type"],
            )
            for column in ["closing_timestamp", "closing_odds", "closing_implied_prob", "closing_line_status"]:
                if column == "closing_timestamp":
                    exact[column] = pd.to_datetime(exact[column], utc=True, errors="coerce").astype("object")
                    values = pd.to_datetime(fallback_values[column], utc=True, errors="coerce").astype("object")
                    exact.loc[unmatched, column] = values.to_numpy()
                else:
                    exact.loc[unmatched, column] = fallback_values[column].to_numpy()
        results = exact.drop(columns=["join_market_type", "join_fighter_id", "join_sportsbook"], errors="ignore")

    results["candidate_timestamp"] = pd.to_datetime(results["candidate_timestamp"], utc=True, errors="coerce")
    results["commence_time"] = pd.to_datetime(results["commence_time"], utc=True, errors="coerce")
    results["closing_timestamp"] = pd.to_datetime(results.get("closing_timestamp"), utc=True, errors="coerce")
    results["candidate_odds"] = pd.to_numeric(results["candidate_odds"], errors="coerce")
    results["closing_odds"] = pd.to_numeric(results.get("closing_odds"), errors="coerce")
    results["closing_implied_prob"] = pd.to_numeric(results.get("closing_implied_prob"), errors="coerce")
    results["clv_pct"] = results.apply(lambda row: clv_pct(row.get("candidate_odds"), row.get("closing_odds")), axis=1)
    results["clv_implied_prob_delta"] = results["closing_implied_prob"] - pd.to_numeric(
        results.get("candidate_implied_probability"), errors="coerce"
    )
    results["beat_closing_line"] = (results["clv_pct"] >= 0).astype("boolean")
    results.loc[results["clv_pct"].isna(), "beat_closing_line"] = pd.NA
    results["hours_before_fight"] = (
        results["commence_time"] - results["candidate_timestamp"]
    ).dt.total_seconds() / 3600
    results["hours_before_close"] = (
        results["closing_timestamp"] - results["candidate_timestamp"]
    ).dt.total_seconds() / 3600
    results["candidate_clv_status"] = np.where(results["closing_odds"].notna(), "priced", "missing_close")

    for column in MODEL_CANDIDATE_CLV_COLUMNS:
        if column not in results.columns:
            results[column] = pd.NA
    return results[MODEL_CANDIDATE_CLV_COLUMNS].sort_values(
        ["candidate_timestamp", "model_id", "fight_id"], ascending=[False, True, True]
    ).reset_index(drop=True)
