from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline.common.paths import MARKET_SIGNALS_PATH, MARKET_INTELLIGENCE_HISTORY_PATH
from utils.ui import page_header
from utils.data_loader import load_parquet


def _metric(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div style="background:#101c2d;border:1px solid #26364a;border-radius:10px;padding:.85rem;text-align:center;">
          <div style="color:#dbe7f5;font-size:.75rem;font-weight:800;text-transform:uppercase;">{label}</div>
          <div style="color:#35d96b;font-size:1.45rem;font-weight:900;margin-top:.25rem;">{value}</div>
          <div style="color:#8fb6df;font-size:.75rem;margin-top:.25rem;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_intelligence():
    page_header("Market Intelligence", "Sportsbook signal feed, consensus gaps, movement, and market diagnostics")

    signals = load_parquet(MARKET_SIGNALS_PATH)
    history = load_parquet(MARKET_INTELLIGENCE_HISTORY_PATH)

    if signals is None:
        signals = pd.DataFrame()
    if history is None:
        history = pd.DataFrame()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _metric("Signals", f"{len(signals):,}", "active feed")
    with c2:
        _metric("Actionable", f"{int(signals.get('is_actionable', pd.Series(dtype=bool)).fillna(False).sum()) if not signals.empty else 0:,}", "opportunities")
    with c3:
        _metric("Steam", f"{int((signals.get('signal_type', pd.Series(dtype=str)) == 'steam_move').sum()) if not signals.empty else 0:,}", "coordinated moves")
    with c4:
        _metric("Consensus", f"{int((signals.get('signal_type', pd.Series(dtype=str)) == 'market_consensus_gap').sum()) if not signals.empty else 0:,}", "gaps")
    with c5:
        _metric("Snapshots", f"{history['refresh_id'].nunique():,}" if not history.empty and "refresh_id" in history.columns else "0", "history")

    st.markdown("### Signal Feed")
    if signals.empty:
        st.info(f"No market signals found at `{MARKET_SIGNALS_PATH}`. Run `python -m pipeline.market.run_build_market_signals`.")
        return

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
    if selected_type != "All" and "signal_type" in out:
        out = out[out["signal_type"].astype(str) == selected_type]
    if selected_severity != "All" and "severity" in out:
        out = out[out["severity"].astype(str) == selected_severity]
    if selected_market != "All" and "market_display" in out:
        out = out[out["market_display"].astype(str) == selected_market]
    if actionable_only and "is_actionable" in out:
        out = out[out["is_actionable"].fillna(False).astype(bool)]

    display_cols = [
        "severity",
        "signal_type",
        "confidence_score",
        "fight_display",
        "market_display",
        "outcome_display",
        "bookmaker",
        "bookmakers_involved",
        "spread_cents",
        "line_move_cents",
        "consensus_implied_probability",
        "explanation",
    ]
    display_cols = [col for col in display_cols if col in out.columns]

    st.dataframe(out[display_cols], use_container_width=True, hide_index=True)
