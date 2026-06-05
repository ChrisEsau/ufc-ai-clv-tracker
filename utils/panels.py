"""Panel helpers retained for existing tab imports."""

import streamlit as st

from utils.ui.sections import section_heading


def render_section_header(title, caption=None):
    section_heading(title, caption=caption)


def render_panel_open():
    st.markdown('<div class="panel">', unsafe_allow_html=True)


def render_panel_close():
    st.markdown('</div>', unsafe_allow_html=True)


def render_status_pill(status):
    status_classes = {
        "OFFICIAL BET": "status-success",
        "WATCHLIST": "status-warning",
        "INVALID MODEL DATA": "status-danger",
        "LOW ODDS MATCH": "status-danger",
        "SPARSE FEATURES": "status-danger",
        "STRONG BET": "status-success",
        "LEAN BET": "status-info",
        "PASS": "status-neutral",
    }
    css_class = status_classes.get(status, "status-info")
    return f'<span class="status-pill {css_class}">{status}</span>'
