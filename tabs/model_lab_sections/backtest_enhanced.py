from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from tabs.model_lab_sections import backtest as base
import utils.model_lab_workflows as mlw


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def _split_summary(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    """Summarize profitability by bucket, split by favorite versus underdog."""

    required = {bucket_col, "Favorite/Underdog", "fight_id", "won", "flat_profit", "flat_stake"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    temp = df.copy()
    temp["won_numeric"] = pd.to_numeric(temp["won"], errors="coerce")
    grouped = temp.groupby([bucket_col, "Favorite/Underdog"], dropna=False).agg(
        Bets=("fight_id", "count"),
        Win_Rate=("won_numeric", "mean"),
        Flat_Profit=("flat_profit", "sum"),
        Flat_Risked=("flat_stake", "sum"),
    ).reset_index()
    grouped["Flat_ROI"] = grouped["Flat_Profit"] / grouped["Flat_Risked"].replace({0: pd.NA})
    grouped = grouped.rename(columns={bucket_col: "Bucket", "Favorite/Underdog": "Side"})
    return grouped[["Bucket", "Side", "Bets", "Win_Rate", "Flat_Profit", "Flat_ROI"]]


def _style_table(df: pd.DataFrame):
    if df.empty:
        return df
    return df.style.format({
        "Win_Rate": "{:.1%}",
        "Flat_Profit": "${:,.0f}",
        "Flat_ROI": "{:.1%}",
    })


def _render_split_table(title: str, df: pd.DataFrame) -> None:
    st.markdown(f"##### {title}")
    if df.empty:
        st.info(f"No data available for {title.lower()}.")
        return
    st.dataframe(_style_table(df), use_container_width=True, hide_index=True)


def _render_favorite_underdog_splits(artifact_dir: Path) -> None:
    bets = base._load_bet_level_table(artifact_dir)
    if bets.empty:
        return

    temp = bets.copy()
    for column in ["edge", "confidence_score", "american_odds", "flat_profit", "flat_stake"]:
        if column in temp.columns:
            temp[column] = pd.to_numeric(temp[column], errors="coerce")

    if "american_odds" not in temp.columns:
        return

    temp["Favorite/Underdog"] = temp["american_odds"].apply(
        lambda odds: "Favorite" if pd.notna(odds) and odds < 0 else "Underdog"
    )

    st.markdown("##### Favorite/Underdog Split Diagnostics")
    st.caption("These tables show whether confidence and edge behave differently for favorites versus underdogs.")

    if "confidence_score" in temp.columns:
        temp["Confidence Bucket"] = pd.cut(
            temp["confidence_score"],
            bins=[0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01],
            labels=["≤55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"],
            include_lowest=True,
        ).astype(str)
        _render_split_table(
            "Favorite/Underdog ROI by Confidence Bucket",
            _split_summary(temp, "Confidence Bucket"),
        )

    if "edge" in temp.columns:
        temp["Edge Bucket"] = pd.cut(
            temp["edge"],
            bins=[-10, 0.05, 0.10, 0.15, 0.20, 10],
            labels=["≤5%", "5-10%", "10-15%", "15-20%", "20%+"],
            include_lowest=True,
        ).astype(str)
        _render_split_table(
            "Favorite/Underdog ROI by Edge Bucket",
            _split_summary(temp, "Edge Bucket"),
        )


def _render_enhanced_diagnostics(artifact_dir: Path) -> None:
    base._render_diagnostics(artifact_dir)
    _render_favorite_underdog_splits(artifact_dir)


def _render_latest_results(context: dict[str, Any]) -> None:
    model_id = str(context.get("model_id") or "")
    market_key = str(context.get("market_key") or "moneyline")
    artifact_dir = base._latest_backtest_dir(model_id, market_key)

    st.markdown("#### Latest Backtest Results")
    if artifact_dir is None:
        st.info("No backtest artifacts found yet for this model. Run a backtest, then refresh after GitHub Actions commits the artifacts.")
        return

    summary = base._read_json(artifact_dir / "backtest_summary.json")
    config = base._read_json(artifact_dir / "backtest_config.json")
    if not summary:
        st.warning(f"Found `{artifact_dir}` but no readable `backtest_summary.json`.")
        return

    st.caption(f"Artifact folder: `{artifact_dir}`")
    base._render_summary_cards(summary)

    tabs = st.tabs(["Summary", "Diagnostics", "Bet Preview", "Config"])
    with tabs[0]:
        base._render_summary_details(summary, artifact_dir)
    with tabs[1]:
        _render_enhanced_diagnostics(artifact_dir)
    with tabs[2]:
        base._render_artifact_tables(artifact_dir)
    with tabs[3]:
        if config:
            st.json(config)
        else:
            st.info("No readable backtest_config.json found for this run.")


def render_backtest(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    existing_model_selector: ExistingModelSelector,
) -> None:
    st.markdown("## Backtest")
    context = existing_model_selector(registry, rows, row_by_id)
    mlw._render_model_bar(context, registry)

    _render_latest_results(context)
    st.divider()
    base._render_run_controls(context)

    with st.expander("GitHub workflow status", expanded=False):
        base._render_latest_run_status()
