from __future__ import annotations

import streamlit as st

from tabs.market_intelligence_sections.best_prices import render_best_prices
from tabs.market_intelligence_sections.book_comparison import render_book_comparison
from tabs.market_intelligence_sections.data import load_market_intelligence_data
from tabs.market_intelligence_sections.diagnostics import render_diagnostics
from tabs.market_intelligence_sections.header import render_header
from tabs.market_intelligence_sections.model_vs_market import render_model_vs_market
from tabs.market_intelligence_sections.overview import render_overview
from tabs.market_intelligence_sections.signal_feed import render_signal_feed
from tabs.market_intelligence_sections.steam import render_steam
from tabs.market_intelligence_sections.styles import inject_market_intelligence_css


def render_market_intelligence() -> None:
    inject_market_intelligence_css()
    data = load_market_intelligence_data()

    render_header()
    render_overview(data)

    left, right = st.columns([1.18, .92], gap="medium")
    with left:
        render_signal_feed(data)
    with right:
        render_best_prices()

    b1, b2 = st.columns([1, 1.05], gap="medium")
    with b1:
        render_book_comparison()
        render_steam(data)
    with b2:
        render_model_vs_market()

    render_diagnostics(data)
