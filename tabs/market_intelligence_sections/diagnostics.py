from __future__ import annotations

import streamlit as st

from tabs.market_intelligence_sections.data import MarketIntelligenceData


def render_diagnostics(data: MarketIntelligenceData) -> None:
    history = data.history
    signals = data.signals

    books = signals["bookmaker"].dropna().astype(str).nunique() if not signals.empty and "bookmaker" in signals.columns else 0
    markets = signals["market_key"].dropna().astype(str).nunique() if not signals.empty and "market_key" in signals.columns else 0
    rows = len(history)

    st.html(
        '<div class="mi-status-bar">'
        f'<span><b>Market Summary</b></span>'
        f'<span>Books in feed: {books}</span>'
        f'<span>Markets: {markets}</span>'
        f'<span>History rows: {rows:,}</span>'
        f'<span>Data source: market_signals.parquet</span>'
        '</div>'
    )
