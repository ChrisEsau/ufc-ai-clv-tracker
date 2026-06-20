from __future__ import annotations

import pandas as pd
import streamlit as st


def render_official_bets(official_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">Official Bet CLV</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">Actual logged wagers remain separate from model candidate validation.</div>', unsafe_allow_html=True)
    if official_clv is None or official_clv.empty:
        st.info("No official CLV rows available yet.")
        return
    cols = [
        "placed_timestamp",
        "event_name",
        "fighter",
        "market_type",
        "sportsbook",
        "odds_taken",
        "closing_odds",
        "clv_pct",
        "beat_closing_line",
        "result",
        "profit_loss",
    ]
    show = official_clv[[col for col in cols if col in official_clv.columns]].copy()
    st.dataframe(show, use_container_width=True, hide_index=True)
