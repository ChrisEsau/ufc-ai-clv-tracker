from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from pipeline.common.paths import MARKET_SIGNALS_PATH
from tabs.market_intelligence_sections.data import MarketIntelligenceData


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _pct(value) -> str:
    try:
        return f"{float(value):.0%}"
    except Exception:
        return "—"


def _moneyline(value) -> str:
    try:
        return f"{int(float(value)):+d}"
    except Exception:
        return "—"


def _severity_class(value) -> str:
    value = str(value or "").lower()
    if value == "opportunity":
        return "opportunity"
    if value == "watch":
        return "watch"
    return "info"


def _signal_card(row: pd.Series) -> str:
    severity = str(row.get("severity") or "info").upper()
    signal_type = str(row.get("signal_type") or "Signal").replace("_", " ").title()
    confidence = _pct(row.get("confidence_score"))
    spread = row.get("spread_cents")
    move = row.get("line_move_cents")
    consensus = row.get("consensus_implied_probability")

    metrics = []
    if pd.notna(spread):
        metrics.append(f"<span>Spread <b>{float(spread):.0f}¢</b></span>")
    if pd.notna(move):
        metrics.append(f"<span>Move <b>{float(move):+.0f}¢</b></span>")
    if pd.notna(consensus):
        metrics.append(f"<span>Consensus <b>{_pct(consensus)}</b></span>")

    best = row.get("best_american_odds")
    worst = row.get("worst_american_odds")
    if pd.notna(best) or pd.notna(worst):
        metrics.append(f"<span>Best/Worst <b>{_moneyline(best)} / {_moneyline(worst)}</b></span>")

    return (
        f'<div class="mi-signal-card {_severity_class(row.get("severity"))}">'
        '<div class="mi-signal-top">'
        f'<span class="mi-signal-badge {_severity_class(row.get("severity"))}">{_escape(severity)}</span>'
        f'<span class="mi-signal-type">{_escape(signal_type)}</span>'
        f'<span class="mi-signal-confidence">{confidence}</span>'
        '</div>'
        f'<div class="mi-signal-title">{_escape(row.get("fight_display"))}</div>'
        f'<div class="mi-signal-subtitle">{_escape(row.get("market_display"))} · {_escape(row.get("outcome_display"))} · {_escape(row.get("bookmaker"))}</div>'
        f'<div class="mi-signal-metrics">{"".join(metrics)}</div>'
        f'<div class="mi-signal-explanation">{_escape(row.get("explanation"))}</div>'
        '</div>'
    )


def render_signal_feed(data: MarketIntelligenceData) -> None:
    signals = data.signals

    st.html(
        '<div class="mi-signal-shell">'
        '<div class="mi-signal-header">'
        '<div class="mi-section-title">Market Signals</div>'
        '<div class="mi-section-subtitle">Scan current sportsbook intelligence signals by severity, type, and market.</div>'
        '</div>'
        '</div>'
    )

    if signals.empty:
        st.info(f"No market signals found at `{MARKET_SIGNALS_PATH}`. Run `python -m pipeline.market.run_build_market_signals`.")
        return

    st.html('<div class="mi-signal-filter-row">')
    filters = st.columns(4)
    with filters[0]:
        signal_types = ["All"] + sorted(signals["signal_type"].dropna().astype(str).unique().tolist()) if "signal_type" in signals else ["All"]
        selected_type = st.selectbox("Signal Type", signal_types, key="mi_signal_type")
    with filters[1]:
        severities = ["All"] + sorted(signals["severity"].dropna().astype(str).unique().tolist()) if "severity" in signals else ["All"]
        selected_severity = st.selectbox("Severity", severities, key="mi_severity")
    with filters[2]:
        markets = ["All"] + sorted(signals["market_display"].dropna().astype(str).unique().tolist()) if "market_display" in signals else ["All"]
        selected_market = st.selectbox("Market", markets, key="mi_market")
    with filters[3]:
        actionable_only = st.checkbox("Actionable only", value=False, key="mi_actionable_only")

    out = signals.copy()
    st.html('</div>')

    if selected_type != "All" and "signal_type" in out:
        out = out[out["signal_type"].astype(str) == selected_type]
    if selected_severity != "All" and "severity" in out:
        out = out[out["severity"].astype(str) == selected_severity]
    if selected_market != "All" and "market_display" in out:
        out = out[out["market_display"].astype(str) == selected_market]
    if actionable_only and "is_actionable" in out:
        out = out[out["is_actionable"].fillna(False).astype(bool)]

    if "confidence_score" in out.columns:
        out = out.sort_values("confidence_score", ascending=False, na_position="last")

    cards = "".join(_signal_card(row) for _, row in out.iterrows())
    st.html(f'<div class="mi-signal-scroll"><div class="mi-signal-list">{cards}</div></div><div class="mi-signal-footer">Showing {len(out):,} signals</div>')
