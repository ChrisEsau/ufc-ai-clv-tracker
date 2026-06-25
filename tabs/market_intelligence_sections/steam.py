from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from tabs.market_intelligence_sections.data import MarketIntelligenceData


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _pct(value) -> str:
    try:
        return f"{float(value):.0%}"
    except Exception:
        return "—"


def _fmt_cents(value) -> str:
    try:
        return f"{float(value):+.0f}¢"
    except Exception:
        return "—"


def _steam_row(row: pd.Series) -> str:
    return (
        '<div class="mi-steam-row">'
        '<div>'
        f'<div class="mi-steam-title">{_escape(row.get("fight_display"))}</div>'
        f'<div class="mi-steam-subtitle">{_escape(row.get("market_display"))} · {_escape(row.get("outcome_display"))}</div>'
        '</div>'
        '<div>'
        f'<div class="mi-steam-value">{_fmt_cents(row.get("line_move_cents"))}</div>'
        f'<div class="mi-steam-subtitle">{_escape(row.get("bookmakers_involved"))}</div>'
        '</div>'
        '<div>'
        f'<div class="mi-steam-confidence">{_pct(row.get("confidence_score"))}</div>'
        '<div class="mi-steam-subtitle">confidence</div>'
        '</div>'
        '</div>'
    )


def render_steam(data: MarketIntelligenceData) -> None:
    signals = data.signals
    steam = pd.DataFrame()
    if not signals.empty and "signal_type" in signals.columns:
        steam = signals[signals["signal_type"].astype(str) == "steam_move"].copy()

    if not steam.empty and "confidence_score" in steam.columns:
        steam = steam.sort_values("confidence_score", ascending=False, na_position="last")

    rows = "".join(_steam_row(row) for _, row in steam.head(8).iterrows())

    body = (
        '<div class="mi-placeholder">'
        '<div class="mi-placeholder-icon">⇄</div>'
        '<div><b>No steam moves detected</b><br><span>Coordinated multi-book movement will appear here after multiple books move the same market.</span></div>'
        '</div>'
        if steam.empty
        else f'<div class="mi-steam-list">{rows}</div>'
    )

    st.html(
        '<div class="mi-card mi-panel mi-small-panel">'
        '<div class="mi-panel-head"><div><div class="mi-section-title">Steam Moves</div>'
        '<div class="mi-section-subtitle">Coordinated same-direction movement across sportsbooks.</div></div>'
        '<div class="mi-panel-link">View steam →</div></div>'
        f'{body}'
        '</div>'
    )
