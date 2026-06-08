# ============================================================
# pipeline/market/normalizers/moneyline.py
# ============================================================

"""Moneyline normalizer for Market Pipeline V2.

The normalizer converts one matched provider/bookmaker/fight row into two
canonical outcome rows that can join directly to Prediction V2 on:

    fight_id + market_key + outcome_label

For moneyline, outcome_label must match the fighter display name emitted by
Prediction V2.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


MARKET_OUTCOME_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "provider_bookmaker_key",
    "provider_event_id",
    "provider_market_key",
    "provider_outcome_label",
    "matched_market_name",
    "matched_outcome_name",
    "event_id",
    "event_name",
    "commence_time",
    "fight_id",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "market_key",
    "outcome_label",
    "outcome_fighter_id",
    "american_odds",
    "decimal_odds",
    "implied_probability",
    "odds_match_type",
    "odds_match_order",
    "odds_match_score",
    "odds_min_single_score",
    "provider_market_last_update",
]


def _provider_side_value(provider_row: pd.Series, match: dict[str, Any], side: str, field_suffix: str):
    """Return a provider fighter value mapped to UFCStats red/blue side."""

    if match.get("fighter_1_side") == side:
        return provider_row.get(f"fighter_1_{field_suffix}")

    if match.get("fighter_2_side") == side:
        return provider_row.get(f"fighter_2_{field_suffix}")

    return None


def _provider_outcome_label(provider_row: pd.Series, match: dict[str, Any], side: str):
    """Return raw provider outcome label for the requested UFCStats side."""

    if match.get("fighter_1_side") == side:
        return provider_row.get("fighter_1_provider_outcome_label")

    if match.get("fighter_2_side") == side:
        return provider_row.get("fighter_2_provider_outcome_label")

    return None


def _build_side_outcome_row(
    *,
    provider_row: pd.Series,
    match: dict[str, Any],
    snapshot_run_id: str,
    snapshot_timestamp: str,
    side: str,
) -> dict[str, Any]:
    """Build one canonical moneyline market outcome row for red or blue."""

    if side == "red":
        outcome_label = match.get("red_fighter")
        outcome_fighter_id = match.get("red_fighter_id")
    elif side == "blue":
        outcome_label = match.get("blue_fighter")
        outcome_fighter_id = match.get("blue_fighter_id")
    else:
        raise ValueError(f"Unsupported moneyline side: {side}")

    return {
        "snapshot_run_id": snapshot_run_id,
        "snapshot_timestamp": snapshot_timestamp,
        "source": provider_row.get("source"),
        "bookmaker": provider_row.get("bookmaker"),
        "provider_bookmaker_key": provider_row.get("provider_bookmaker_key"),
        "provider_event_id": provider_row.get("provider_event_id"),
        "provider_market_key": provider_row.get("provider_market_key"),
        "provider_outcome_label": _provider_outcome_label(provider_row, match, side),
        "matched_market_name": "moneyline",
        "matched_outcome_name": outcome_label,
        "event_id": match.get("event_id"),
        "event_name": match.get("event_name"),
        "commence_time": provider_row.get("commence_time"),
        "fight_id": match.get("fight_id"),
        "red_fighter": match.get("red_fighter"),
        "blue_fighter": match.get("blue_fighter"),
        "red_fighter_id": match.get("red_fighter_id"),
        "blue_fighter_id": match.get("blue_fighter_id"),
        "market_key": "moneyline",
        "outcome_label": outcome_label,
        "outcome_fighter_id": outcome_fighter_id,
        "american_odds": _provider_side_value(provider_row, match, side, "american_odds"),
        "decimal_odds": _provider_side_value(provider_row, match, side, "decimal_odds"),
        "implied_probability": _provider_side_value(provider_row, match, side, "implied_prob"),
        "odds_match_type": match.get("odds_match_type"),
        "odds_match_order": match.get("match_type"),
        "odds_match_score": match.get("odds_match_score"),
        "odds_min_single_score": match.get("odds_min_single_score"),
        "provider_market_last_update": provider_row.get("provider_market_last_update"),
    }


def normalize_moneyline_provider_row(
    *,
    provider_row: pd.Series,
    match: dict[str, Any],
    snapshot_run_id: str,
    snapshot_timestamp: str,
) -> list[dict[str, Any]]:
    """Convert one matched provider moneyline row into red and blue outcomes."""

    return [
        _build_side_outcome_row(
            provider_row=provider_row,
            match=match,
            snapshot_run_id=snapshot_run_id,
            snapshot_timestamp=snapshot_timestamp,
            side="red",
        ),
        _build_side_outcome_row(
            provider_row=provider_row,
            match=match,
            snapshot_run_id=snapshot_run_id,
            snapshot_timestamp=snapshot_timestamp,
            side="blue",
        ),
    ]


def ensure_market_outcome_columns(outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stable Market V2 moneyline schema even when output is empty."""

    out = outcomes_df.copy()
    for column in MARKET_OUTCOME_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")

    return out[MARKET_OUTCOME_COLUMNS]
