from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from tabs.clv_intelligence_sections.data import american, number, pct, signed_pct


def _positive_class(value) -> str:
    if pd.isna(value):
        return "clvi-neutral"
    return "clvi-positive" if float(value) >= 0 else "clvi-negative"


def render_filters(candidate_clv: pd.DataFrame) -> None:
    if candidate_clv.empty:
        return
    model_ids = ["All Models"]
    if "model_id" in candidate_clv.columns:
        model_ids.extend(sorted({str(v) for v in candidate_clv["model_id"].dropna() if str(v).strip()}))
    markets = ["All Markets"]
    if "market_key" in candidate_clv.columns:
        markets.extend(sorted({str(v) for v in candidate_clv["market_key"].dropna() if str(v).strip()}))
    books = ["All Books"]
    if "bookmaker" in candidate_clv.columns:
        books.extend(sorted({str(v) for v in candidate_clv["bookmaker"].dropna() if str(v).strip()}))

    col1, col2, col3, col4 = st.columns([1.15, 1.0, 1.0, .8], gap="medium")
    with col1:
        st.selectbox("Model", model_ids, key="clvi_filter_model")
    with col2:
        st.selectbox("Market", markets, key="clvi_filter_market")
    with col3:
        st.selectbox("Book", books, key="clvi_filter_book")
    with col4:
        st.selectbox("Close Status", ["All", "Priced", "Missing Close"], key="clvi_filter_status")


def render_kpis(candidate_clv: pd.DataFrame, official_clv: pd.DataFrame) -> None:
    priced = candidate_clv[candidate_clv.get("closing_odds", pd.Series(dtype=float)).notna()].copy() if not candidate_clv.empty else pd.DataFrame()
    candidates = len(candidate_clv)
    priced_count = len(priced)
    models = candidate_clv["model_id"].nunique() if not candidate_clv.empty and "model_id" in candidate_clv.columns else 0
    beat_close = float(priced["beat_closing_line"].dropna().mean()) if not priced.empty and "beat_closing_line" in priced.columns else float("nan")
    avg_clv = float(priced["clv_pct"].mean()) if not priced.empty and "clv_pct" in priced.columns else float("nan")
    official_count = 0 if official_clv is None or official_clv.empty else len(official_clv)
    tracked_rate = priced_count / candidates if candidates else float("nan")

    cols = st.columns(6, gap="medium")
    values = [
        ("Beat Close", pct(beat_close), None),
        ("Avg Candidate CLV", signed_pct(avg_clv), None),
        ("Candidates", f"{candidates:,}", None),
        ("Priced", f"{priced_count:,}", None),
        ("Models", f"{models:,}", None),
        ("Official Bets", f"{official_count:,}", f"Tracked {pct(tracked_rate)}"),
    ]
    for col, (label, value, help_text) in zip(cols, values):
        with col:
            st.metric(label, value, help=help_text)


def render_model_leaderboard(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">Model Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">Candidate CLV performance by model.</div>', unsafe_allow_html=True)
    if candidate_clv.empty or "model_id" not in candidate_clv.columns:
        st.info("No candidate CLV rows available.")
        return
    priced = candidate_clv[candidate_clv.get("closing_odds", pd.Series(dtype=float)).notna()].copy()
    if priced.empty:
        st.info("No priced candidate rows yet.")
        return
    summary = (
        priced.groupby("model_id", dropna=False)
        .agg(
            candidates=("candidate_id", "count"),
            avg_clv=("clv_pct", "mean"),
            beat_close=("beat_closing_line", "mean"),
        )
        .sort_values(["avg_clv", "beat_close"], ascending=False)
        .reset_index()
    )
    st.markdown(
        '<div class="clvi-leader-row clvi-leader-head"><div>Model</div><div>Cand.</div><div>Avg CLV</div><div>Beat</div></div>',
        unsafe_allow_html=True,
    )
    for _, row in summary.head(8).iterrows():
        clv_class = _positive_class(row["avg_clv"])
        st.markdown(
            f"""
            <div class="clvi-leader-row">
                <div>{row['model_id']}</div>
                <div>{int(row['candidates']):,}</div>
                <div class="{clv_class}">{signed_pct(row['avg_clv'])}</div>
                <div>{pct(row['beat_close'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_edge_scatter(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">Model Edge vs Market Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">Each point is a frozen model candidate. Positive Y means the market moved toward the model side.</div>', unsafe_allow_html=True)
    if candidate_clv.empty:
        st.info("No candidate rows available.")
        return
    plot_df = candidate_clv.dropna(subset=["candidate_edge", "clv_pct"]).copy()
    if plot_df.empty:
        st.info("Candidate edge or CLV is not populated yet.")
        return
    fig = px.scatter(
        plot_df,
        x="candidate_edge",
        y="clv_pct",
        color="model_id" if "model_id" in plot_df.columns else None,
        hover_data=[col for col in ["fight_display", "outcome_display", "candidate_odds", "closing_odds", "bookmaker"] if col in plot_df.columns],
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Candidate Edge", tickformat=".0%", gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(255,255,255,.28)")
    fig.update_yaxes(title="CLV", tickformat=".0%", gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(255,255,255,.28)")
    st.plotly_chart(fig, use_container_width=True)


def render_market_validation_matrix(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">Market Validation Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">Quick read on whether high-edge candidates are being confirmed by the market.</div>', unsafe_allow_html=True)
    if candidate_clv.empty or "candidate_edge" not in candidate_clv.columns or "clv_pct" not in candidate_clv.columns:
        st.info("Need edge and CLV fields for matrix.")
        return
    priced = candidate_clv.dropna(subset=["candidate_edge", "clv_pct"]).copy()
    if priced.empty:
        st.info("No priced candidates available.")
        return
    edge_cut = priced["candidate_edge"].median()
    cells = {
        "High Edge + Positive CLV": len(priced[(priced["candidate_edge"] >= edge_cut) & (priced["clv_pct"] >= 0)]),
        "High Edge + Negative CLV": len(priced[(priced["candidate_edge"] >= edge_cut) & (priced["clv_pct"] < 0)]),
        "Low Edge + Positive CLV": len(priced[(priced["candidate_edge"] < edge_cut) & (priced["clv_pct"] >= 0)]),
        "Low Edge + Negative CLV": len(priced[(priced["candidate_edge"] < edge_cut) & (priced["clv_pct"] < 0)]),
    }
    c1, c2 = st.columns(2)
    with c1:
        st.metric("High Edge + Positive CLV", f"{cells['High Edge + Positive CLV']:,}")
        st.metric("Low Edge + Positive CLV", f"{cells['Low Edge + Positive CLV']:,}")
    with c2:
        st.metric("High Edge + Negative CLV", f"{cells['High Edge + Negative CLV']:,}")
        st.metric("Low Edge + Negative CLV", f"{cells['Low Edge + Negative CLV']:,}")


def render_candidate_explorer(candidate_clv: pd.DataFrame) -> None:
    st.markdown('<div class="clvi-card-title">Candidate Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="clvi-card-caption">Frozen candidate odds compared with closing odds.</div>', unsafe_allow_html=True)
    if candidate_clv.empty:
        st.info("No candidate rows available.")
        return
    cols = [
        "candidate_timestamp",
        "model_id",
        "fight_display",
        "outcome_display",
        "market_key",
        "bookmaker",
        "candidate_odds",
        "closing_odds",
        "candidate_edge",
        "candidate_confidence_pct",
        "clv_pct",
        "beat_closing_line",
        "hours_before_fight",
    ]
    show = candidate_clv[[col for col in cols if col in candidate_clv.columns]].copy()
    st.dataframe(show, use_container_width=True, hide_index=True)
