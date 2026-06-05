"""Card components for dashboard KPIs and compact stats."""

from __future__ import annotations

import html
from textwrap import dedent

import streamlit as st

ACCENT_COLORS = {
    "success": "#35d96b",
    "green": "#35d96b",
    "danger": "#ef4444",
    "red": "#ef4444",
    "info": "#3b82f6",
    "blue": "#3b82f6",
    "warning": "#facc15",
    "amber": "#facc15",
    "purple": "#a855f7",
    "neutral": "#f5f7fb",
}


def metric_card(
    label: str,
    value,
    delta: str | None = None,
    status: str = "neutral",
    caption: str | None = None,
) -> None:
    """Render a high-contrast KPI card."""

    color = ACCENT_COLORS.get(status, ACCENT_COLORS["neutral"])
    delta_html = (
        f'<div class="metric-delta" style="color:{color};">{html.escape(str(delta))}</div>'
        if delta
        else ""
    )
    caption_html = (
        f'<div class="metric-subtext">{html.escape(str(caption))}</div>'
        if caption
        else ""
    )
    card_html = dedent(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value" style="color:{color};">{html.escape(str(value))}</div>
            {delta_html}
            {caption_html}
        </div>
        """
    ).strip()
    st.markdown(card_html, unsafe_allow_html=True)


def stat_row(label: str, value, status: str = "neutral") -> str:
    """Return HTML for a labeled stat row used inside markdown cards."""

    color = ACCENT_COLORS.get(status, ACCENT_COLORS["neutral"])
    return (
        '<div style="display:flex;justify-content:space-between;gap:.75rem;'
        'padding:.42rem 0;border-bottom:1px solid rgba(38,54,74,.65);">'
        f'<span style="color:#9aa8bd;">{html.escape(str(label))}</span>'
        f'<strong style="color:{color};">{html.escape(str(value))}</strong>'
        "</div>"
    )
