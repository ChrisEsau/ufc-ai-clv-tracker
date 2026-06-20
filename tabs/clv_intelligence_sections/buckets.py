from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def _bar_chart(df: pd.DataFrame, x: str, title: str) -> None:
    if df.empty or x not in df.columns or "clv_pct" not in df.columns:
        st.info(f"No data available for {title}.")
        return
    work = df.dropna(subset=[x, "clv_pct"]).copy()
    if work.empty:
        st.info(f"No populated rows available for {title}.")
        return
    summary = (
        work.groupby(x, dropna=False)
        .agg(candidates=("candidate_id", "count"), avg_clv=("clv_pct", "mean"), beat_close=("beat_closing_line", "mean"))
        .reset_index()
    )
    fig = px.bar(summary, x=x, y="avg_clv", text="candidates", hover_data=["beat_close"])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title="Avg CLV", tickformat=".0%", gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(255,255,255,.28)")
    fig.update_xaxes(title="")
    st.plotly_chart(fig, use_container_width=True)


def _edge_bucket(value) -> str:
    if pd.isna(value):
        return "Unknown"
    value = float(value)
    if value < 0.02:
        return "0-2%"
    if value < 0.05:
        return "2-5%"
    if value < 0.10:
        return "5-10%"
    return "10%+"


def _timing_bucket(value) -> str:
    if pd.isna(value):
        return "Unknown"
    value = float(value)
    if value < 6:
        return "0-6h"
    if value < 12:
        return "6-12h"
    if value < 24:
        return "12-24h"
    if value < 48:
        return "24-48h"
    return "48h+"


def render_edge_bucket_chart(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">CLV by Edge Bucket</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">Does higher model edge produce better market validation?</div>', unsafe_allow_html=True)
    work = candidate_clv.copy()
    if "candidate_edge" in work.columns:
        work["edge_bucket"] = work["candidate_edge"].apply(_edge_bucket)
    _bar_chart(work, "edge_bucket", "edge bucket")


def render_confidence_chart(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">CLV by Confidence Tier</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">Checks whether confidence tiers are market validated.</div>', unsafe_allow_html=True)
    work = candidate_clv.copy()
    if "candidate_confidence_tier" not in work.columns or work["candidate_confidence_tier"].isna().all():
        work["candidate_confidence_tier"] = pd.cut(
            pd.to_numeric(work.get("candidate_confidence_pct", pd.Series(dtype=float)), errors="coerce"),
            bins=[-1, 60, 65, 70, 100],
            labels=["<60%", "60-65%", "65-70%", "70%+"],
        ).astype(str)
    _bar_chart(work, "candidate_confidence_tier", "confidence tier")


def render_timing_chart(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">Timing Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">How candidate timing relates to CLV.</div>', unsafe_allow_html=True)
    work = candidate_clv.copy()
    if "hours_before_fight" in work.columns:
        work["timing_bucket"] = work["hours_before_fight"].apply(_timing_bucket)
    _bar_chart(work, "timing_bucket", "timing")
