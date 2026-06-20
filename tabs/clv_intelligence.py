from __future__ import annotations

import streamlit as st

from tabs.clv_intelligence_sections.buckets import (
    render_confidence_chart,
    render_edge_bucket_chart,
    render_timing_chart,
)
from tabs.clv_intelligence_sections.data import (
    filter_candidates,
    load_clv_intelligence_data,
    prepare_candidate_clv,
    prepare_official_clv,
)
from tabs.clv_intelligence_sections.header import render_header
from tabs.clv_intelligence_sections.official_bets import render_official_bets
from tabs.clv_intelligence_sections.overview import (
    render_candidate_explorer,
    render_edge_scatter,
    render_filters,
    render_kpis,
    render_market_validation_matrix,
    render_model_leaderboard,
)
from tabs.clv_intelligence_sections.styles import inject_styles


def render_clv_intelligence() -> None:
    """Render the candidate-based CLV Intelligence workspace."""

    inject_styles()
    data = load_clv_intelligence_data()
    candidate_clv = prepare_candidate_clv(data.candidate_clv)
    official_clv = prepare_official_clv(data.official_clv)

    render_header(candidate_clv, official_clv)

    if candidate_clv.empty:
        st.warning(
            "No model candidate CLV artifact found yet. Run the UFC CLV Tracker after Market Refresh to populate candidate CLV."
        )
        return

    with st.container(border=True):
        render_filters(candidate_clv)

    filtered_candidates = filter_candidates(candidate_clv)

    render_kpis(filtered_candidates, official_clv)

    main_col, side_col = st.columns([1.55, 1.0], gap="medium")
    with main_col:
        with st.container(border=True):
            render_edge_scatter(filtered_candidates)
    with side_col:
        with st.container(border=True):
            render_model_leaderboard(filtered_candidates)

    row2_col1, row2_col2 = st.columns(2, gap="medium")
    with row2_col1:
        with st.container(border=True):
            render_edge_bucket_chart(filtered_candidates)
    with row2_col2:
        with st.container(border=True):
            render_confidence_chart(filtered_candidates)

    row3_col1, row3_col2 = st.columns(2, gap="medium")
    with row3_col1:
        with st.container(border=True):
            render_timing_chart(filtered_candidates)
    with row3_col2:
        with st.container(border=True):
            render_market_validation_matrix(filtered_candidates)

    with st.container(border=True):
        render_official_bets(official_clv)

    with st.container(border=True):
        render_candidate_explorer(filtered_candidates)
