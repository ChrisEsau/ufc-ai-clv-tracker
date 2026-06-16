from __future__ import annotations

import streamlit as st

from utils.operations_status import latest_update_label


def render_header() -> None:
    left, right = st.columns([1.8, 1])
    with left:
        st.html(
            '<div class="ops-title">Operations Center</div>'
            '<div class="ops-subtitle">Run and monitor data, model, market, and betting operations</div>'
        )
    with right:
        st.html(f"<div class='ops-actions'><span>{latest_update_label()}</span></div>")
