from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline.common.risk_settings import (
    MarketRiskFilter,
    RiskSettings,
    default_market_filters,
    normalize_market_key,
)


MARKET_LABELS = {
    "moneyline": "Moneyline",
    "goes_distance": "Goes Distance",
    "ko_tko": "KO/TKO",
    "submission": "Submission",
    "decision": "Decision",
    "over_under": "Over / Under",
}


def market_label(market_key: str) -> str:
    key = normalize_market_key(market_key)
    return MARKET_LABELS.get(key, key.replace("_", " ").title())


def default_editable_markets(settings: RiskSettings) -> list[str]:
    keys = list(default_market_filters().keys())
    for key in settings.market_filters:
        normalized = normalize_market_key(key)
        if normalized not in keys:
            keys.append(normalized)
    for key in ["ko_tko", "submission", "decision", "over_under"]:
        if key not in keys:
            keys.append(key)
    return keys


def _fallback_filter(settings: RiskSettings, market_key: str) -> MarketRiskFilter:
    defaults = default_market_filters()
    key = normalize_market_key(market_key)
    if key in settings.market_filters:
        return settings.market_filters[key]
    if key in defaults:
        return defaults[key]
    if "moneyline" in settings.market_filters:
        return settings.market_filters["moneyline"]
    return MarketRiskFilter(settings.min_edge, settings.min_confidence, settings.min_odds, settings.max_odds)


def render_market_risk_controls(settings: RiskSettings) -> dict[str, MarketRiskFilter]:
    """Render editable per-market filters and return updated filters.

    This component is intentionally independent from the Bankroll page so the
    Bankroll tab can remain focused on layout while the market-key risk controls
    scale as new prop markets are added.
    """

    market_filters: dict[str, MarketRiskFilter] = {}
    markets = default_editable_markets(settings)

    st.markdown("#### Market Filters")
    st.caption("These thresholds apply per market key. Bankroll, Kelly, max stake, and event exposure remain global.")

    header = st.columns([1.55, 1, 1, 1, 1])
    header[0].markdown("**Market**")
    header[1].markdown("**Min Edge %**")
    header[2].markdown("**Min Confidence %**")
    header[3].markdown("**Min Odds**")
    header[4].markdown("**Max Odds**")

    for market_key in markets:
        current = _fallback_filter(settings, market_key)
        columns = st.columns([1.55, 1, 1, 1, 1])
        columns[0].markdown(f"**{market_label(market_key)}**  ")
        min_edge_pct = columns[1].number_input(
            "Min edge %",
            min_value=-100.0,
            max_value=100.0,
            value=float(current.min_edge * 100),
            step=0.5,
            key=f"risk_{market_key}_min_edge_pct",
            label_visibility="collapsed",
        )
        min_confidence = columns[2].number_input(
            "Min confidence %",
            min_value=0.0,
            max_value=100.0,
            value=float(current.min_confidence),
            step=1.0,
            key=f"risk_{market_key}_min_confidence",
            label_visibility="collapsed",
        )
        min_odds = columns[3].number_input(
            "Min odds",
            min_value=-2000,
            max_value=3000,
            value=int(current.min_odds),
            step=5,
            key=f"risk_{market_key}_min_odds",
            label_visibility="collapsed",
        )
        max_odds = columns[4].number_input(
            "Max odds",
            min_value=-2000,
            max_value=3000,
            value=int(current.max_odds),
            step=5,
            key=f"risk_{market_key}_max_odds",
            label_visibility="collapsed",
        )
        market_filters[normalize_market_key(market_key)] = MarketRiskFilter(
            min_edge=float(min_edge_pct) / 100.0,
            min_confidence=float(min_confidence),
            min_odds=int(min_odds),
            max_odds=int(max_odds),
        )

    return market_filters


def market_filters_to_payload(market_filters: dict[str, MarketRiskFilter]) -> dict[str, dict[str, float | int]]:
    return {
        normalize_market_key(market_key): {
            "min_edge": market_filter.min_edge,
            "min_confidence": market_filter.min_confidence,
            "min_odds": market_filter.min_odds,
            "max_odds": market_filter.max_odds,
        }
        for market_key, market_filter in market_filters.items()
    }


def market_filters_summary_frame(settings: RiskSettings) -> pd.DataFrame:
    rows = []
    for market_key in default_editable_markets(settings):
        market_filter = _fallback_filter(settings, market_key)
        rows.append(
            {
                "Market": market_label(market_key),
                "Min Edge": f"{market_filter.min_edge * 100:.1f}%",
                "Min Confidence": f"{market_filter.min_confidence:.1f}%",
                "Odds Range": f"{market_filter.min_odds} to {market_filter.max_odds:+d}",
            }
        )
    return pd.DataFrame(rows)
