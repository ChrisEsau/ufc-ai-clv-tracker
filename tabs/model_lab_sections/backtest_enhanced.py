from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from tabs.model_lab_sections import backtest as base
from tabs.model_lab_sections.dog_audit import render_dog_audit
import utils.model_lab_workflows as mlw
from utils.workflow_status import launch_workflow_with_status, workflow_status_label


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def _split_summary(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
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
    return df.style.format({"Win_Rate": "{:.1%}", "Flat_Profit": "${:,.0f}", "Flat_ROI": "{:.1%}"})


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
    temp["Favorite/Underdog"] = temp["american_odds"].apply(lambda odds: "Favorite" if pd.notna(odds) and odds < 0 else "Underdog")
    st.markdown("##### Favorite/Underdog Split Diagnostics")
    st.caption("These tables show whether confidence and edge behave differently for favorites versus underdogs.")
    if "confidence_score" in temp.columns:
        temp["Confidence Bucket"] = pd.cut(
            temp["confidence_score"],
            bins=[0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01],
            labels=["≤55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"],
            include_lowest=True,
        ).astype(str)
        _render_split_table("Favorite/Underdog ROI by Confidence Bucket", _split_summary(temp, "Confidence Bucket"))
    if "edge" in temp.columns:
        temp["Edge Bucket"] = pd.cut(
            temp["edge"],
            bins=[-10, 0.05, 0.10, 0.15, 0.20, 10],
            labels=["≤5%", "5-10%", "10-15%", "15-20%", "20%+"],
            include_lowest=True,
        ).astype(str)
        _render_split_table("Favorite/Underdog ROI by Edge Bucket", _split_summary(temp, "Edge Bucket"))


def _render_enhanced_diagnostics(artifact_dir: Path) -> None:
    base._render_diagnostics(artifact_dir)
    _render_favorite_underdog_splits(artifact_dir)


def _render_lazy_dog_audit(artifact_dir: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    st.markdown("##### Dog Audit")
    st.info("Dog Audit can be slow because it loads the model and runs SHAP. Click the button below when you want to run it.")
    button_key = f"run_dog_audit_button_{artifact_dir.name}"
    if st.button("Run Dog Audit", type="primary", use_container_width=True, key=button_key):
        render_dog_audit(artifact_dir, summary, config)


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
    tabs = st.tabs(["Summary", "Diagnostics", "Dog Audit", "Bet Preview", "Config"])
    with tabs[0]:
        base._render_summary_details(summary, artifact_dir)
    with tabs[1]:
        _render_enhanced_diagnostics(artifact_dir)
    with tabs[2]:
        _render_lazy_dog_audit(artifact_dir, summary, config)
    with tabs[3]:
        base._render_artifact_tables(artifact_dir)
    with tabs[4]:
        if config:
            st.json(config)
        else:
            st.info("No readable backtest_config.json found for this run.")


def _render_run_controls_with_status(context: dict[str, Any]) -> None:
    model_config_path = str(context.get("config_path") or "")
    feature_view_path = base._default_feature_view_path(context)
    historical_market_path = base.DEFAULT_HISTORICAL_MARKET_PATH
    market_key = str(context.get("market_key") or "moneyline")
    status_key = f"mlab_backtest_{context['model_id']}"
    st.markdown("#### Run New Backtest")
    st.caption("Uses the selected model automatically. Technical paths are hidden unless you open Advanced paths.")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.text_input("Start Date", value="", placeholder="YYYY-MM-DD optional", key=f"mlab_backtest_start_date_{context['model_id']}")
    with c2:
        end_date = st.text_input("End Date", value="", placeholder="YYYY-MM-DD optional", key=f"mlab_backtest_end_date_{context['model_id']}")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        min_edge = st.number_input("Min Edge", value=0.00, step=0.01, min_value=0.0, max_value=1.0, key=f"mlab_backtest_min_edge_{context['model_id']}")
    with p2:
        min_confidence = st.number_input("Min Confidence", value=0.00, step=0.01, min_value=0.0, max_value=1.0, key=f"mlab_backtest_min_confidence_{context['model_id']}")
    with p3:
        flat_stake = st.number_input("Flat Stake", value=100, step=25, min_value=1, key=f"mlab_backtest_flat_stake_{context['model_id']}")
    with p4:
        starting_bankroll = st.number_input("Starting Bankroll", value=10000, step=500, min_value=1, key=f"mlab_backtest_starting_bankroll_{context['model_id']}")
    with st.expander("Advanced paths", expanded=False):
        model_config_path = st.text_input("Model Config Path", value=model_config_path, key=f"mlab_backtest_config_path_{context['model_id']}")
        feature_view_path = st.text_input("Historical Feature View Path", value=feature_view_path, key=f"mlab_backtest_feature_view_path_{context['model_id']}")
        historical_market_path = st.text_input("Historical Market Outcomes Path", value=historical_market_path, key=f"mlab_backtest_market_path_{context['model_id']}")
        market_key = st.text_input("Market Key", value=market_key, key=f"mlab_backtest_market_key_{context['model_id']}")
    missing = [label for label, value in {"model config path": model_config_path, "feature view path": feature_view_path, "historical market path": historical_market_path, "market key": market_key}.items() if not value]
    required_ready = not missing
    st.caption(f"Status: {workflow_status_label(base.WORKFLOW_FILE, status_key, idle_label='Ready')}")
    if missing:
        st.warning("Backtest cannot launch until these fields are populated: " + ", ".join(missing))
    if st.button("Run Full Backtest", type="primary", disabled=not required_ready, use_container_width=True, key=f"{status_key}_button"):
        inputs = {
            "model_config_path": str(model_config_path),
            "feature_view_path": str(feature_view_path),
            "historical_market_path": str(historical_market_path),
            "market_key": str(market_key),
            "start_date": str(start_date or ""),
            "end_date": str(end_date or ""),
            "min_edge": f"{float(min_edge):.4f}",
            "min_confidence": f"{float(min_confidence):.4f}",
            "flat_stake": str(int(flat_stake)),
            "starting_bankroll": str(int(starting_bankroll)),
        }
        ok, message = launch_workflow_with_status(base.WORKFLOW_FILE, status_key, inputs=inputs)
        if ok:
            st.success(message)
            st.rerun()
        else:
            st.error(message)


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
    _render_run_controls_with_status(context)
    with st.expander("GitHub workflow status", expanded=False):
        base._render_latest_run_status()
