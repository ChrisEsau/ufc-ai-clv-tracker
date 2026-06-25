from __future__ import annotations

import pandas as pd
import streamlit as st

from tabs.market_intelligence_sections.data import MarketIntelligenceData


def _kpi(icon: str, label: str, value: str, caption: str = "") -> str:
    return (
        '<div class="mi-card mi-kpi">'
        f'<div class="mi-kpi-icon">{icon}</div>'
        '<div>'
        f'<div class="mi-kpi-label">{label}</div>'
        f'<div class="mi-kpi-value">{value}</div>'
        f'<div class="mi-kpi-caption">{caption}</div>'
        '</div>'
        '</div>'
    )


def render_overview(data: MarketIntelligenceData) -> None:
    signals = data.signals
    history = data.history

    active = len(signals)
    actionable = int(signals.get("is_actionable", pd.Series(dtype=bool)).fillna(False).sum()) if not signals.empty else 0
    steam = int((signals.get("signal_type", pd.Series(dtype=str)) == "steam_move").sum()) if not signals.empty else 0
    consensus = int((signals.get("signal_type", pd.Series(dtype=str)) == "market_consensus_gap").sum()) if not signals.empty else 0
    snapshots = history["refresh_id"].nunique() if not history.empty and "refresh_id" in history.columns else 0

    st.html(
        '<div class="mi-kpis">'
        + _kpi("〽", "Signals", f"{active:,}", "active feed")
        + _kpi("◎", "Actionable", f"{actionable:,}", "opportunities")
        + _kpi("⇄", "Steam", f"{steam:,}", "coordinated moves")
        + _kpi("◔", "Consensus", f"{consensus:,}", "gaps")
        + _kpi("◷", "Snapshots", f"{snapshots:,}", "history")
        + '</div>'
    )
