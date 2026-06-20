from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def _bar_chart(df: pd.DataFrame, x: str, title: str) -> None:
    if df.empty or x not in df.columns or "clv_pct" not in df.columns:
        st.info(f"No data available for {title}.")
        return
    work = df.dropna(subset=[x, "clv_pct"]).copy()
    work = work[work[x].astype(str).str.lower() != "unknown"]
    if work.empty:
        st.markdown('<div class="clvi-empty-note">Not enough usable rows yet.</div>', unsafe_allow_html=True)
        return
    summary = (
        work.groupby(x, dropna=False)
        .agg(candidates=("candidate_id", "count"), avg_clv=("clv_pct", "mean"), beat_close=("beat_closing_line", "mean"))
        .reset_index()
    )
    fig = px.bar(summary, x=x, y="avg_clv", text="candidates", hover_data=["beat_close"])
    fig.update_traces(textposition="inside", cliponaxis=False)
    fig.update_layout(
        height=245,
        margin=dict(l=8, r=8, t=4, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_yaxes(title="Avg CLV", tickformat=".0%", gridcolor="rgba(148,163,184,.14)", zerolinecolor="rgba(255,255,255,.28)")
    fig.update_xaxes(title="")
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


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
    if value < 0:
        return "Unknown"
    if value < 6:
        return "0-6h"
    if value < 12:
        return "6-12h"
    if value < 24:
        return "12-24h"
    if value < 48:
        return "24-48h"
    return "48h+"


def _american_label(value) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return "—"
    rounded = int(round(float(parsed)))
    return f"+{rounded}" if rounded > 0 else str(rounded)


def _signed_pct(value) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return "—"
    return f"{float(parsed) * 100:+.1f}%"


def render_edge_bucket_chart(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">CLV by Edge Bucket</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">Does higher model edge produce better market validation?</div>', unsafe_allow_html=True)
    work = candidate_clv.copy()
    if "candidate_edge" in work.columns:
        work["edge_bucket"] = work["candidate_edge"].apply(_edge_bucket)
    _bar_chart(work, "edge_bucket", "edge bucket")


def render_steam_chart(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">Steam Moves</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">Largest moves toward a model candidate from first signal to close.</div>', unsafe_allow_html=True)
    if candidate_clv.empty or "clv_pct" not in candidate_clv.columns:
        st.markdown('<div class="clvi-empty-note">No candidate CLV rows available.</div>', unsafe_allow_html=True)
        return

    work = candidate_clv.dropna(subset=["clv_pct", "candidate_odds", "closing_odds"]).copy()
    work = work[work["clv_pct"] > 0].sort_values("clv_pct", ascending=False).head(6)
    if work.empty:
        st.markdown('<div class="clvi-empty-note">No positive steam moves found yet.</div>', unsafe_allow_html=True)
        return

    for _, row in work.iterrows():
        outcome = str(row.get("outcome_display") or "Unknown side")
        fight = str(row.get("fight_display") or "Unknown fight")
        model = str(row.get("model_id") or "Unknown model")
        odds_move = f"{_american_label(row.get('candidate_odds'))} → {_american_label(row.get('closing_odds'))}"
        clv = _signed_pct(row.get("clv_pct"))
        st.markdown(
            f"""
            <div class="clvi-steam-row">
                <div class="clvi-steam-main">
                    <div class="clvi-steam-side">{outcome}</div>
                    <div class="clvi-steam-meta">{fight} · {model}</div>
                </div>
                <div class="clvi-steam-odds">{odds_move}</div>
                <div class="clvi-steam-clv">{clv}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_timing_chart(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">Timing Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">How candidate timing relates to CLV.</div>', unsafe_allow_html=True)
    work = candidate_clv.copy()
    if "hours_before_fight" in work.columns:
        usable = pd.to_numeric(work["hours_before_fight"], errors="coerce")
        if usable.notna().sum() == 0:
            st.markdown('<div class="clvi-empty-note">Waiting for usable fight-time timestamps.</div>', unsafe_allow_html=True)
            return
        work["timing_bucket"] = usable.apply(_timing_bucket)
    _bar_chart(work, "timing_bucket", "timing")
