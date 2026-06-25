from __future__ import annotations

import streamlit as st


def render_model_vs_market() -> None:
    st.html(
        '<div class="mi-card mi-panel mi-small-panel">'
        '<div class="mi-panel-head"><div><div class="mi-section-title">Model vs Market</div>'
        '<div class="mi-section-subtitle">Compare model probability vs market consensus.</div></div>'
        '<div class="mi-panel-link">View details →</div></div>'
        '<div class="mi-placeholder">'
        '<div class="mi-placeholder-icon">↗</div>'
        '<div><b>No model data available</b><br><span>Run the model pipeline to generate model vs market consensus.</span></div>'
        '</div>'
        '</div>'
    )
