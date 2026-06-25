from __future__ import annotations

import streamlit as st


def render_book_comparison() -> None:
    st.html(
        '<div class="mi-card mi-panel mi-small-panel">'
        '<div class="mi-panel-head"><div><div class="mi-section-title">Book Comparison</div>'
        '<div class="mi-section-subtitle">Cross-book prices for the selected high-signal market.</div></div>'
        '<div class="mi-panel-link">View full table →</div></div>'
        '<div class="mi-placeholder">'
        '<div class="mi-placeholder-icon">▦</div>'
        '<div><b>No data to display</b><br><span>Select a fight and market with active signals to view cross-book comparison.</span></div>'
        '</div>'
        '</div>'
    )
