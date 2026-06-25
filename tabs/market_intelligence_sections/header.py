from __future__ import annotations

import streamlit as st

from utils.operations_status import latest_update_label


def render_header() -> None:
    left, right = st.columns([1.8, 1])
    with left:
        st.html(
            '<div class="mi-title">Market Intelligence</div>'
            '<div class="mi-subtitle">Sportsbook signal feed, consensus gaps, movement, and market diagnostics</div>'
        )
    with right:
        st.html(f"<div class='mi-actions'><span>{latest_update_label()}</span></div>")
