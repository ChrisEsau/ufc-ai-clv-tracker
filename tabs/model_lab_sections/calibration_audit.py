from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from tabs.model_lab_sections import backtest as base


PROBABILITY_BUCKET_BINS = [i / 100 for i in range(0, 105, 5)]
PROBABILITY_BUCKET_LABELS = [f"{i}-{i + 5}%" for i in range(0, 100, 5)]
DISAGREEMENT_BUCKET_BINS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 1.01]
DISAGREEMENT_BUCKET_LABELS = ["0-2%", "2-5%", "5-10%", "10-15%", "15-20%", "20%+"]


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _prepare_calibration_frame(bets: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if bets.empty:
        return pd.DataFrame(), None

    probability_col = _first_existing_column(
        bets,
        [
            "model_probability",
            "predicted_probability",
            "probability",
            "model_prob",
            "best_prob",
            "pick_probability",
            "confidence_score",
        ],
    )
    if probability_col is None:
        return pd.DataFrame(), None

    required = {"won", "implied_probability"}
    if not required.issubset(bets.columns):
        return pd.DataFrame(), probability_col

    temp = bets.copy()
    temp["model_probability"] = pd.to_numeric(temp[probability_col], errors="coerce")
    temp["market_probability"] = pd.to_numeric(temp["implied_probability"], errors="coerce")
    temp["won_numeric"] = pd.to_numeric(temp["won"], errors="coerce")
    temp["edge"] = pd.to_numeric(temp.get("edge"), errors="coerce")
    temp["flat_profit"] = pd.to_numeric(temp.get("flat_profit"), errors="coerce")
    temp["flat_stake"] = pd.to_numeric(temp.get("flat_stake"), errors="coerce")

    temp = temp[
        temp["model_probability"].between(0, 1, inclusive="both")
        & temp["market_probability"].between(0, 1, inclusive="both")
        & temp["won_numeric"].notna()
    ].copy()
    if temp.empty:
        return temp, probability_col

    temp["Model Probability Bucket"] = pd.cut(
        temp["model_probability"],
        bins=PROBABILITY_BUCKET_BINS,
        labels=PROBABILITY_BUCKET_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)
    temp["Market Probability Bucket"] = pd.cut(
        temp["market_probability"],
        bins=PROBABILITY_BUCKET_BINS,
        labels=PROBABILITY_BUCKET_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)
    temp["model_market_disagreement"] = (temp["model_probability"] - temp["market_probability"]).abs()
    temp["Disagreement Bucket"] = pd.cut(
        temp["model_market_disagreement"],
        bins=DISAGREEMENT_BUCKET_BINS,
        labels=DISAGREEMENT_BUCKET_LABELS,
        include_lowest=True,
    ).astype(str)
    return temp, probability_col


def _model_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("Model Probability Bucket", dropna=False).agg(
        Bets=("won_numeric", "count"),
        Avg_Model_Prob=("model_probability", "mean"),
        Avg_Market_Prob=("market_probability", "mean"),
        Actual_Win_Rate=("won_numeric", "mean"),
        Avg_Disagreement=("model_market_disagreement", "mean"),
        Avg_Edge=("edge", "mean"),
        Flat_Profit=("flat_profit", "sum"),
        Flat_Risked=("flat_stake", "sum"),
    ).reset_index()
    grouped["Model_Error"] = grouped["Avg_Model_Prob"] - grouped["Actual_Win_Rate"]
    grouped["Market_Error"] = grouped["Avg_Market_Prob"] - grouped["Actual_Win_Rate"]
    grouped["Model_Advantage"] = grouped["Market_Error"].abs() - grouped["Model_Error"].abs()
    grouped["Flat_ROI"] = grouped["Flat_Profit"] / grouped["Flat_Risked"].replace({0: pd.NA})
    return grouped


def _market_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("Market Probability Bucket", dropna=False).agg(
        Bets=("won_numeric", "count"),
        Avg_Model_Prob=("model_probability", "mean"),
        Avg_Market_Prob=("market_probability", "mean"),
        Actual_Win_Rate=("won_numeric", "mean"),
        Avg_Disagreement=("model_market_disagreement", "mean"),
        Avg_Edge=("edge", "mean"),
        Flat_Profit=("flat_profit", "sum"),
        Flat_Risked=("flat_stake", "sum"),
    ).reset_index()
    grouped["Model_Error"] = grouped["Avg_Model_Prob"] - grouped["Actual_Win_Rate"]
    grouped["Market_Error"] = grouped["Avg_Market_Prob"] - grouped["Actual_Win_Rate"]
    grouped["Model_Advantage"] = grouped["Market_Error"].abs() - grouped["Model_Error"].abs()
    grouped["Flat_ROI"] = grouped["Flat_Profit"] / grouped["Flat_Risked"].replace({0: pd.NA})
    return grouped


def _disagreement_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("Disagreement Bucket", dropna=False).agg(
        Bets=("won_numeric", "count"),
        Avg_Model_Prob=("model_probability", "mean"),
        Avg_Market_Prob=("market_probability", "mean"),
        Actual_Win_Rate=("won_numeric", "mean"),
        Avg_Disagreement=("model_market_disagreement", "mean"),
        Avg_Edge=("edge", "mean"),
        Flat_Profit=("flat_profit", "sum"),
        Flat_Risked=("flat_stake", "sum"),
    ).reset_index()
    grouped["Model_Error"] = grouped["Avg_Model_Prob"] - grouped["Actual_Win_Rate"]
    grouped["Market_Error"] = grouped["Avg_Market_Prob"] - grouped["Actual_Win_Rate"]
    grouped["Model_Advantage"] = grouped["Market_Error"].abs() - grouped["Model_Error"].abs()
    grouped["Flat_ROI"] = grouped["Flat_Profit"] / grouped["Flat_Risked"].replace({0: pd.NA})
    return grouped


def _weighted_abs_error(df: pd.DataFrame, prob_col: str) -> float | None:
    if df.empty:
        return None
    grouped = df.groupby(pd.cut(df[prob_col], bins=PROBABILITY_BUCKET_BINS, include_lowest=True, right=False), observed=False).agg(
        Bets=("won_numeric", "count"),
        Avg_Prob=(prob_col, "mean"),
        Actual_Win_Rate=("won_numeric", "mean"),
    )
    grouped = grouped[grouped["Bets"] > 0]
    if grouped.empty:
        return None
    grouped["Abs_Error"] = (grouped["Avg_Prob"] - grouped["Actual_Win_Rate"]).abs()
    return float((grouped["Bets"] * grouped["Abs_Error"]).sum() / grouped["Bets"].sum())


def _style_calibration_table(df: pd.DataFrame):
    if df.empty:
        return df
    formatters = {
        "Avg_Model_Prob": "{:.1%}",
        "Avg_Market_Prob": "{:.1%}",
        "Actual_Win_Rate": "{:.1%}",
        "Avg_Disagreement": "{:.1%}",
        "Avg_Edge": "{:+.1%}",
        "Model_Error": "{:+.1%}",
        "Market_Error": "{:+.1%}",
        "Model_Advantage": "{:+.1%}",
        "Flat_Profit": "${:,.0f}",
        "Flat_Risked": "${:,.0f}",
        "Flat_ROI": "{:.1%}",
    }
    return df.style.format({key: value for key, value in formatters.items() if key in df.columns})


def _render_table(title: str, df: pd.DataFrame, caption: str | None = None) -> None:
    st.markdown(f"###### {title}")
    if caption:
        st.caption(caption)
    if df.empty:
        st.info(f"No data available for {title.lower()}.")
        return
    st.dataframe(_style_calibration_table(df), use_container_width=True, hide_index=True)


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def render_calibration_audit(artifact_dir: Path) -> None:
    st.markdown("##### Model vs Market Calibration")
    st.caption("Compares model probability, market implied probability, and realized win rate across the full 0-100% probability range.")

    bets = base._load_bet_level_table(artifact_dir)
    temp, probability_col = _prepare_calibration_frame(bets)
    if probability_col is None:
        st.warning("Calibration Audit could not find a model probability column. Expected one of: model_probability, predicted_probability, probability, model_prob, best_prob, pick_probability, confidence_score.")
        return
    if temp.empty:
        st.warning(f"Calibration Audit could not use `{probability_col}` because required rows or columns are missing.")
        return

    if probability_col == "confidence_score":
        st.warning("Using `confidence_score` as a fallback. This is pick-level confidence and may not equal outcome-level probability.")
    else:
        st.caption(f"Using `{probability_col}` as outcome-level model probability and `implied_probability` as market probability.")

    model_ece = _weighted_abs_error(temp, "model_probability")
    market_ece = _weighted_abs_error(temp, "market_probability")
    avg_disagreement = float(temp["model_market_disagreement"].mean()) if not temp.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(temp):,}")
    c2.metric("Model ECE", _fmt_pct(model_ece))
    c3.metric("Market ECE", _fmt_pct(market_ece))
    c4.metric("Avg Disagreement", _fmt_pct(avg_disagreement))

    _render_table(
        "Model Probability Buckets",
        _model_bucket_summary(temp),
        "Bucketed by model probability. Positive Model Advantage means model is closer to actual win rate than the market in that bucket.",
    )
    _render_table(
        "Market Probability Buckets",
        _market_bucket_summary(temp),
        "Bucketed by market implied probability. Useful for seeing whether the market or model is better calibrated across price ranges.",
    )
    _render_table(
        "Model vs Market Disagreement",
        _disagreement_summary(temp),
        "Bucketed by absolute model-market probability gap. This shows whether large disagreements are productive or dangerous.",
    )

    with st.expander("How to read this", expanded=False):
        st.write(
            "Model Error = Avg Model Prob - Actual Win Rate. Market Error = Avg Market Prob - Actual Win Rate. Model Advantage = |Market Error| - |Model Error|, so positive values mean the model was closer to reality than the market. ECE is the bet-count weighted average absolute calibration error across 5% probability buckets."
        )
