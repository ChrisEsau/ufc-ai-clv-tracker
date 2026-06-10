# ============================================================
# pipeline/market/providers/the_odds_api.py
# ============================================================

"""The Odds API provider adapter for Market Pipeline V2.

This module is intentionally provider-specific and does not perform UFCStats
fight matching, betting decisions, EV calculations, or CLV calculations.
It only fetches and flattens provider odds into a raw provider dataframe that
normalizers and matchers can consume.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ufc_odds_utils import fetch_the_odds_api_events
from ufc_pipeline_utils import (
    american_to_decimal,
    american_to_implied_prob,
    normalize_name,
)


def _as_list(value) -> list:
    """Normalize scalar/list config values into a plain list."""

    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return list(value)

    return [value]


def provider_market_string(config: dict) -> str:
    """Build the comma-separated provider market request string.

    The config stores canonical market keys and provider-specific market keys.
    For Phase 10A, moneyline maps to the Odds API h2h market.
    """

    configured_markets = _as_list(config.get("markets"))
    provider_map = config.get("market_provider_keys", {}) or {}

    provider_markets = []
    for market_key in configured_markets:
        provider_key = provider_map.get(market_key)
        if provider_key and provider_key not in provider_markets:
            provider_markets.append(provider_key)

    if not provider_markets:
        provider_markets = ["h2h"]

    return ",".join(provider_markets)


def fetch_odds(api_key: str, config: dict) -> list[dict]:
    """Fetch raw provider odds JSON from The Odds API."""

    return fetch_the_odds_api_events(
        api_key=api_key,
        sport=config.get("sport", "mma_mixed_martial_arts"),
        regions=config.get("regions", "us"),
        markets=provider_market_string(config),
        odds_format=config.get("odds_format", "american"),
    )


def flatten_provider_market_diagnostics(
    odds_json: Iterable[dict],
    bookmakers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Flatten every returned provider market/outcome for inspection.

    This diagnostic artifact is intentionally raw. It is used to verify what The
    Odds API returns for configured non-moneyline markets before those markets
    are mapped into canonical UFC market_outcomes rows.
    """

    bookmaker_filter = set(bookmakers or [])
    rows = []

    for event in odds_json:
        provider_event_id = event.get("id")
        event_name = f"{event.get('home_team', '')} vs {event.get('away_team', '')}".strip()
        commence_time = event.get("commence_time")

        for bookmaker in event.get("bookmakers", []):
            bookmaker_title = bookmaker.get("title")
            bookmaker_key = bookmaker.get("key")

            if bookmaker_filter and bookmaker_title not in bookmaker_filter:
                continue

            for market in bookmaker.get("markets", []):
                provider_market_key = market.get("key")
                provider_market_last_update = market.get("last_update")

                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    rows.append(
                        {
                            "source": "the_odds_api",
                            "provider_event_id": provider_event_id,
                            "event_name": event_name,
                            "commence_time": commence_time,
                            "provider_bookmaker_key": bookmaker_key,
                            "bookmaker": bookmaker_title,
                            "provider_market_key": provider_market_key,
                            "provider_market_last_update": provider_market_last_update,
                            "provider_outcome_label": outcome.get("name"),
                            "provider_outcome_description": outcome.get("description"),
                            "price": price,
                            "decimal_odds": american_to_decimal(price),
                            "implied_probability": american_to_implied_prob(price),
                            "point": outcome.get("point"),
                        }
                    )

    return pd.DataFrame(rows)


def flatten_moneyline_odds(
    odds_json: Iterable[dict],
    bookmakers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Flatten Odds API h2h markets into provider-level fight rows.

    One output row represents one provider event/bookmaker/moneyline market.
    It still contains provider fighter order. UFCStats side mapping is handled
    later by the Market V2 outcome matcher.
    """

    bookmaker_filter = set(bookmakers or [])
    rows = []

    for event in odds_json:
        provider_event_id = event.get("id")
        event_name = f"{event.get('home_team', '')} vs {event.get('away_team', '')}".strip()
        commence_time = event.get("commence_time")

        for bookmaker in event.get("bookmakers", []):
            bookmaker_title = bookmaker.get("title")
            bookmaker_key = bookmaker.get("key")

            if bookmaker_filter and bookmaker_title not in bookmaker_filter:
                continue

            for market in bookmaker.get("markets", []):
                provider_market_key = market.get("key")
                if provider_market_key != "h2h":
                    continue

                outcomes = market.get("outcomes", [])
                if len(outcomes) < 2:
                    continue

                first = outcomes[0]
                second = outcomes[1]

                fighter_1 = first.get("name")
                fighter_2 = second.get("name")
                odds_1 = first.get("price")
                odds_2 = second.get("price")

                rows.append({
                    "source": "the_odds_api",
                    "provider_event_id": provider_event_id,
                    "provider_bookmaker_key": bookmaker_key,
                    "bookmaker": bookmaker_title,
                    "provider_market_key": provider_market_key,
                    "matched_market_name": "moneyline",
                    "event_name": event_name,
                    "commence_time": commence_time,
                    "provider_market_last_update": market.get("last_update"),
                    "fighter_1": fighter_1,
                    "fighter_2": fighter_2,
                    "fighter_1_norm": normalize_name(fighter_1),
                    "fighter_2_norm": normalize_name(fighter_2),
                    "fighter_1_american_odds": odds_1,
                    "fighter_2_american_odds": odds_2,
                    "fighter_1_decimal_odds": american_to_decimal(odds_1),
                    "fighter_2_decimal_odds": american_to_decimal(odds_2),
                    "fighter_1_implied_prob": american_to_implied_prob(odds_1),
                    "fighter_2_implied_prob": american_to_implied_prob(odds_2),
                    "fighter_1_provider_outcome_label": fighter_1,
                    "fighter_2_provider_outcome_label": fighter_2,
                })

    return pd.DataFrame(rows)
