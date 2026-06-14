from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from tabs.model_lab_sections import backtest as base


PROBABILITY_BUCKET_BINS = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
PROBABILITY_BUCKET_LABELS = ["≤50%", "50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-85%", "85-90%", "90%+"]


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

    required = {"won", "american_odds"}
    if not required.issubset(bets.columns):
        return pd.DataFrame(), probability_col

    temp = bets.copy()
    temp["model_probability"] = pd.to_numeric(temp[probability_col], errors="coerce")
    temp["won_numeric"] = pd.to_numeric(temp["won"], errors="coerce")
    temp["american_odds"] = pd.to_numeric(temp["american_odds"], errors="coerce")
    temp["edge"] = pd.to_numeric(temp.get("edge"), errors="coerce")
    temp["flat_profit"] = pd.to_numeric(temp.get("flat_profit"), errors="coerce")
    temp["flat_stake"] = pd.to_numeric(temp.get("flat_stake"), errors="coerce")

    temp = temp[temp["model_probability"].notna() & temp["won_numeric"].notna() & temp["american_odds"].notna()].copy()
    if temp.empty:
        return temp, probability_col

    temp["Favorite/Underdog"] = temp["american_odds"].apply(lambda odds: "Favorite" if odds < 0 else "Underdog")
    temp["Probability Bucket"] = pd.cut(
        temp["model_probability"],
        bins=PROBABILITY_BUCKET_BINS,
        labels=PROBABILITY_BUCKET_LABELS,
        include_lowest=True,
    ).astype(str)
    return temp, probability_col


def _calibration_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(group_cols, dropna=False).agg(
        Bets=("won_numeric", "count"),
        Avg_Model_Prob=("model_probability", "mean"),
        Actual_Win_Rate=("won_numeric", "mean"),
        Avg_Edge=("edge", "mean"),
        Flat_Profit=("flat_profit", "sum"),
        Flat_Risked=("flat_stake", "sum"),
    ).reset_index()
    grouped["Calibration_Error"] = grouped["Avg_Model_Prob"] - grouped["Actual_Win_Rate"]
    grouped["Abs_Calibration_Error"] = grouped["Calibration_Error"].abs()
    grouped["Flat_ROI"] = grouped["Flat_Profit"] / grouped["Flat_Risked"].replace({0: pd.NA})
    return grouped


def _expected_calibration_error(summary: pd.DataFrame) -> float | None:
    if summary.empty or not {"Bets", "Abs_Calibration_Error"}.issubset(summary.columns):
        return None
    total = pd.to_numeric(summary["Bets"], errors="coerce").sum()
    if not total:
        return None
    return float((summary["Bets"] * summary["Abs_Calibration_Error"]).sum() / total)


def _style_calibration_table(df: pd.DataFrame):
    if df.empty:
        return df
    formatters = {
        "Avg_Model_Prob": "{:.1%}",
        "Actual_Win_Rate": "{:.1%}",
        "Calibration_Error": "{:+.1%}",
        "Abs_Calibration_Error": "{:.1%}",
        "Avg_Edge": "{:.1%}",
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


def render_calibration_audit(artifact_dir: Path) -> None:
    st.markdown("##### Calibration Audit")
    st.caption("Checks whether model probabilities match realized win rates, especially split by favorite versus underdog.")

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
        st.caption(f"Using `{probability_col}` as outcome-level model probability.")

    overall = _calibration_summary(temp, ["Probability Bucket"])
    by_side = _calibration_summary(temp, ["Favorite/Underdog", "Probability Bucket"])
    side_summary = _calibration_summary(temp, ["Favorite/Underdog"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Bets", f"{len(temp):,}")
    overall_ece = _expected_calibration_error(overall)
    side_ece = _expected_calibration_error(by_side)
    c2.metric("Overall ECE", "—" if overall_ece is None else f"{overall_ece:.1%}")
    c3.metric("Side-Bucket ECE", "—" if side_ece is None else f"{side_ece:.1%}")

    _render_table(
        "Favorite vs Underdog Calibration",
        side_summary,
        "Positive calibration error means model probability is higher than actual win rate.",
    )
    _render_table("Calibration by Probability Bucket", overall)
    _render_table(
        "Calibration by Side and Probability Bucket",
        by_side,
        "This is the key table for diagnosing dog overconfidence.",
    )

    with st.expander("How to read this", expanded=False):
        st.write(
            "Calibration Error = Avg Model Probability - Actual Win Rate. A positive value means the model is overconfident in that bucket; a negative value means it is underconfident. ECE is the bet-count weighted average absolute calibration error."
        )
