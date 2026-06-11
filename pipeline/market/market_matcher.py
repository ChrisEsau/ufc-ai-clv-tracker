# ============================================================
# pipeline/market/market_matcher.py
# ============================================================

"""Generic canonical sportsbook market matcher.

This module consumes sportsbook-agnostic canonical market catalog rows and
matches them to the UFC live card. It intentionally contains no provider-specific
logic. Provider quirks must be resolved before this layer in provider normalizers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline.common.outcome_join import build_outcome_join_key
from ufc_odds_utils import composite_name_score


MARKET_OUTCOME_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "provider_event_id",
    "provider_market_id",
    "provider_selection_id",
    "provider_market_name",
    "provider_selection_name",
    "event_id",
    "event_name",
    "commence_time",
    "fight_id",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "market_key",
    "blue_last_name_token",
    "red_last_name_token",
    "provider_matchup_text",
    "matchup_secondary_confirmed",
    "matching_strategy",
    "outcome_label",
    "outcome_key",
    "outcome_type",
    "outcome_fighter_id",
    "outcome_join_key",
    "outcome_fighter_name",
    "side",
    "line",
    "american_odds",
    "decimal_odds",
    "implied_probability",
    "odds_match_type",
    "odds_match_score",
    "odds_min_single_score",
    "market_family",
    "is_conditional_no_action",
    "condition_key",
    "round_number",
    "method_key",
]

MARKET_MATCH_AUDIT_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "provider_event_id",
    "provider_market_id",
    "provider_selection_id",
    "provider_market_name",
    "provider_selection_name",
    "market_key",
    "matchup_secondary_confirmed",
    "provider_matchup_text",
    "red_last_name_token",
    "blue_last_name_token",
    "matching_strategy",
    "outcome_key",
    "fighter_name",
    "matched_fight_id",
    "event_id",
    "event_name",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "odds_match_type",
    "odds_match_score",
    "odds_min_single_score",
    "red_score",
    "blue_score",
    "event_score",
    "is_matched",
]


def _safe_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _last_name_token(value: Any) -> str:
    """Return a simple last-name token for secondary matchup confirmation."""

    text = _safe_str(value).lower().replace(".", "").replace(",", "")
    parts = [part for part in text.split() if part]
    return parts[-1] if parts else ""


def _provider_matchup_text(catalog_row: pd.Series) -> str:
    """Provider text used for matchup-level confirmation."""

    return " ".join(
        part
        for part in [
            _safe_str(catalog_row.get("event_name")),
            _safe_str(catalog_row.get("provider_market_name")),
            _safe_str(catalog_row.get("provider_selection_name")),
        ]
        if part
    ).lower()


def _both_fighters_present_in_provider_text(catalog_row: pd.Series, live_row: pd.Series) -> bool:
    """Confirm both live-card fighter last names appear in provider matchup text."""

    provider_text = _provider_matchup_text(catalog_row)
    red_last = _last_name_token(live_row.get("red_fighter"))
    blue_last = _last_name_token(live_row.get("blue_fighter"))

    return bool(red_last and blue_last and red_last in provider_text and blue_last in provider_text)


def _matching_strategy(catalog_row: pd.Series) -> str:
    """Return the configured market matching strategy."""

    strategy = _safe_str(catalog_row.get("matching_strategy")).lower()
    return strategy or "fighter_name"


def _last_name_token(value: Any) -> str:
    """Return a simple last-name token for secondary matchup confirmation."""

    text = _safe_str(value).lower().replace(".", "").replace(",", "")
    parts = [part for part in text.split() if part]
    return parts[-1] if parts else ""


def _provider_matchup_text(catalog_row: pd.Series) -> str:
    """Provider text used for matchup-level confirmation."""

    return " ".join(
        part
        for part in [
            _safe_str(catalog_row.get("event_name")),
            _safe_str(catalog_row.get("provider_market_name")),
            _safe_str(catalog_row.get("provider_selection_name")),
        ]
        if part
    ).lower()


def _matchup_confirmation_payload(catalog_row: pd.Series, live_row: pd.Series) -> dict[str, Any]:
    """Return secondary matchup confirmation details for auditability."""

    provider_text = _provider_matchup_text(catalog_row)
    red_last = _last_name_token(live_row.get("red_fighter"))
    blue_last = _last_name_token(live_row.get("blue_fighter"))
    confirmed = bool(red_last and blue_last and red_last in provider_text and blue_last in provider_text)

    return {
        "matchup_secondary_confirmed": confirmed,
        "provider_matchup_text": provider_text,
        "red_last_name_token": red_last,
        "blue_last_name_token": blue_last,
    }


def _event_score(catalog_row: pd.Series, live_row: pd.Series) -> float:
    """Score provider event/market text against the red-blue live matchup."""

    provider_text = " ".join(
        part
        for part in [
            _safe_str(catalog_row.get("event_name")),
            _safe_str(catalog_row.get("provider_market_name")),
        ]
        if part
    )
    if not provider_text:
        return 0.0

    red_blue = f"{live_row.get('red_fighter', '')} {live_row.get('blue_fighter', '')}"
    blue_red = f"{live_row.get('blue_fighter', '')} {live_row.get('red_fighter', '')}"

    return max(
        composite_name_score(provider_text, red_blue),
        composite_name_score(provider_text, blue_red),
    )


def _fighter_scores(catalog_row: pd.Series, live_row: pd.Series) -> tuple[float, float]:
    fighter_name = _safe_str(catalog_row.get("fighter_name"))
    if not fighter_name:
        return np.nan, np.nan

    return (
        composite_name_score(fighter_name, live_row.get("red_fighter")),
        composite_name_score(fighter_name, live_row.get("blue_fighter")),
    )


def _score_catalog_row_to_live_fight(catalog_row: pd.Series, live_row: pd.Series) -> dict[str, Any]:
    event_score = _event_score(catalog_row, live_row)
    red_score, blue_score = _fighter_scores(catalog_row, live_row)

    strategy = _matching_strategy(catalog_row)
    matchup_payload = _matchup_confirmation_payload(catalog_row, live_row)
    has_fighter = not np.isnan(red_score) and not np.isnan(blue_score)

    if strategy == "matchup_name":
        # Fight-level markets such as goes_distance and total_rounds usually do
        # not have a fighter-specific selection. Use fuzzy score, but require
        # secondary confirmation that both live-card fighter last names appear
        # in provider text before accepting the candidate.
        if not matchup_payload["matchup_secondary_confirmed"]:
            match_score = 0.0
            min_single_score = 0.0
        else:
            match_score = event_score
            min_single_score = event_score
    elif strategy == "event_name":
        match_score = event_score
        min_single_score = event_score
    elif has_fighter:
        best_fighter_score = max(red_score, blue_score)
        match_score = (event_score + best_fighter_score) / 2 if event_score > 0 else best_fighter_score
        min_single_score = min(event_score if event_score > 0 else best_fighter_score, best_fighter_score)
    else:
        match_score = event_score
        min_single_score = event_score

    return {
        "live_row": live_row,
        "odds_match_score": float(match_score),
        "odds_min_single_score": float(min_single_score),
        "red_score": red_score,
        "blue_score": blue_score,
        "event_score": event_score,
        "matching_strategy": strategy,
        **matchup_payload,
    }


def match_canonical_market_row_to_live_card(
    catalog_row: pd.Series,
    live_card_df: pd.DataFrame,
    min_match_score: float = 80,
) -> dict[str, Any] | None:
    """Return the best live-card fight match for one canonical market row."""

    candidates = [
        _score_catalog_row_to_live_fight(catalog_row, live_row)
        for _, live_row in live_card_df.iterrows()
    ]
    if not candidates:
        return None

    best = sorted(candidates, key=lambda item: item["odds_match_score"], reverse=True)[0]
    if best["odds_min_single_score"] < min_match_score:
        return None

    live_row = best["live_row"]
    return {
        "fight_id": live_row.get("fight_id"),
        "event_id": live_row.get("event_id"),
        "event_name": live_row.get("event_name"),
        "commence_time": live_row.get("event_date"),
        "red_fighter": live_row.get("red_fighter"),
        "blue_fighter": live_row.get("blue_fighter"),
        "red_fighter_id": live_row.get("red_fighter_id"),
        "blue_fighter_id": live_row.get("blue_fighter_id"),
        "odds_match_type": "matched",
        "odds_match_score": best["odds_match_score"],
        "odds_min_single_score": best["odds_min_single_score"],
        "red_score": best["red_score"],
        "blue_score": best["blue_score"],
        "event_score": best["event_score"],
        "blue_last_name_token": best.get("blue_last_name_token"),
        "red_last_name_token": best.get("red_last_name_token"),
        "provider_matchup_text": best.get("provider_matchup_text"),
        "matchup_secondary_confirmed": best.get("matchup_secondary_confirmed"),
        "matching_strategy": best.get("matching_strategy"),
    }


def _outcome_fighter_id(catalog_row: pd.Series, match: dict[str, Any] | None) -> Any:
    if match is None:
        return pd.NA

    fighter_name = _safe_str(catalog_row.get("fighter_name"))
    if not fighter_name:
        return pd.NA

    red_score = composite_name_score(fighter_name, match.get("red_fighter"))
    blue_score = composite_name_score(fighter_name, match.get("blue_fighter"))

    if red_score >= blue_score:
        return match.get("red_fighter_id")
    return match.get("blue_fighter_id")


def _outcome_label(catalog_row: pd.Series) -> Any:
    market_key = _safe_str(catalog_row.get("market_key"))
    side = _safe_str(catalog_row.get("side")).lower()
    outcome_key = _safe_str(catalog_row.get("outcome_key"))
    fighter_name = _safe_str(catalog_row.get("fighter_name"))
    line = catalog_row.get("line")

    if market_key == "goes_distance":
        if side == "yes":
            return "goes_distance"
        if side == "no":
            return "inside_distance"

    if market_key == "total_rounds" and side in {"over", "under"} and not pd.isna(line):
        normalized_line = str(line).replace(".", "_")
        return f"{side}_{normalized_line}"

    if fighter_name:
        return fighter_name

    return outcome_key or side or pd.NA


def build_market_outcome_row(catalog_row: pd.Series, match: dict[str, Any]) -> dict[str, Any]:
    """Build one production market outcome row from a matched canonical row."""

    outcome_label = _outcome_label(catalog_row)
    outcome_fighter_id = _outcome_fighter_id(catalog_row, match)
    outcome_join_key = build_outcome_join_key(
        market_key=catalog_row.get("market_key"),
        outcome_label=outcome_label,
        outcome_fighter_id=outcome_fighter_id,
        outcome_key=catalog_row.get("outcome_key"),
        side=catalog_row.get("side"),
        line=catalog_row.get("line"),
    )

    return {
        "snapshot_run_id": catalog_row.get("snapshot_run_id"),
        "snapshot_timestamp": catalog_row.get("snapshot_timestamp"),
        "source": catalog_row.get("source"),
        "bookmaker": catalog_row.get("bookmaker"),
        "provider_event_id": catalog_row.get("provider_event_id"),
        "provider_market_id": catalog_row.get("provider_market_id"),
        "provider_selection_id": catalog_row.get("provider_selection_id"),
        "provider_market_name": catalog_row.get("provider_market_name"),
        "provider_selection_name": catalog_row.get("provider_selection_name"),
        "event_id": match.get("event_id"),
        "event_name": match.get("event_name"),
        "commence_time": match.get("commence_time"),
        "fight_id": match.get("fight_id"),
        "red_fighter": match.get("red_fighter"),
        "blue_fighter": match.get("blue_fighter"),
        "red_fighter_id": match.get("red_fighter_id"),
        "blue_fighter_id": match.get("blue_fighter_id"),
        "market_key": catalog_row.get("market_key"),
        "matching_strategy": match.get("matching_strategy", catalog_row.get("matching_strategy")),
        "matching_strategy": catalog_row.get("matching_strategy"),
        "outcome_label": outcome_label,
        "outcome_key": catalog_row.get("outcome_key"),
        "outcome_type": catalog_row.get("outcome_type"),
        "outcome_fighter_id": outcome_fighter_id,
        "outcome_join_key": outcome_join_key,
        "outcome_fighter_name": catalog_row.get("fighter_name"),
        "side": catalog_row.get("side"),
        "line": catalog_row.get("line"),
        "american_odds": catalog_row.get("american_odds"),
        "decimal_odds": catalog_row.get("decimal_odds"),
        "implied_probability": catalog_row.get("implied_probability"),
        "odds_match_type": match.get("odds_match_type"),
        "odds_match_score": match.get("odds_match_score"),
        "odds_min_single_score": match.get("odds_min_single_score"),
        "market_family": catalog_row.get("market_family"),
        "is_conditional_no_action": catalog_row.get("is_conditional_no_action"),
        "condition_key": catalog_row.get("condition_key"),
        "round_number": catalog_row.get("round_number"),
        "method_key": catalog_row.get("method_key"),
    }


def build_market_match_audit_row(catalog_row: pd.Series, match: dict[str, Any] | None) -> dict[str, Any]:
    """Build one audit row for canonical market matching."""

    row = {
        "snapshot_run_id": catalog_row.get("snapshot_run_id"),
        "snapshot_timestamp": catalog_row.get("snapshot_timestamp"),
        "source": catalog_row.get("source"),
        "bookmaker": catalog_row.get("bookmaker"),
        "provider_event_id": catalog_row.get("provider_event_id"),
        "provider_market_id": catalog_row.get("provider_market_id"),
        "provider_selection_id": catalog_row.get("provider_selection_id"),
        "provider_market_name": catalog_row.get("provider_market_name"),
        "provider_selection_name": catalog_row.get("provider_selection_name"),
        "market_key": catalog_row.get("market_key"),
        "matching_strategy": catalog_row.get("matching_strategy"),
        "outcome_key": catalog_row.get("outcome_key"),
        "fighter_name": catalog_row.get("fighter_name"),
        "matched_fight_id": None,
        "event_id": None,
        "event_name": catalog_row.get("event_name"),
        "red_fighter": None,
        "blue_fighter": None,
        "red_fighter_id": None,
        "blue_fighter_id": None,
        "odds_match_type": "unmatched",
        "odds_match_score": np.nan,
        "odds_min_single_score": np.nan,
        "red_score": np.nan,
        "blue_score": np.nan,
        "event_score": np.nan,
        "is_matched": False,
    }

    if match is not None:
        row.update(
            {
                "matched_fight_id": match.get("fight_id"),
                "event_id": match.get("event_id"),
                "event_name": match.get("event_name"),
                "red_fighter": match.get("red_fighter"),
                "blue_fighter": match.get("blue_fighter"),
                "red_fighter_id": match.get("red_fighter_id"),
                "blue_fighter_id": match.get("blue_fighter_id"),
                "odds_match_type": match.get("odds_match_type"),
                "odds_match_score": match.get("odds_match_score"),
                "odds_min_single_score": match.get("odds_min_single_score"),
                "red_score": match.get("red_score"),
                "blue_score": match.get("blue_score"),
                "event_score": match.get("event_score"),
                "matching_strategy": match.get("matching_strategy"),
                "matchup_secondary_confirmed": match.get("matchup_secondary_confirmed"),
                "provider_matchup_text": match.get("provider_matchup_text"),
                "red_last_name_token": match.get("red_last_name_token"),
                "blue_last_name_token": match.get("blue_last_name_token"),
                "is_matched": True,
            }
        )

    return row


def ensure_market_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in MARKET_OUTCOME_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[MARKET_OUTCOME_COLUMNS]


def ensure_market_match_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in MARKET_MATCH_AUDIT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[MARKET_MATCH_AUDIT_COLUMNS]
