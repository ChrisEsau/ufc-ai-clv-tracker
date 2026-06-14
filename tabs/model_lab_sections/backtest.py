from __future__ import annotations

from typing import Any, Callable

import streamlit as st

import utils.model_lab_workflows as mlw
from utils.github_actions import get_latest_workflow_run, trigger_workflow


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]

WORKFLOW_FILE = "run-full-model-backtest-v2.yml"
DEFAULT_HISTORICAL_MARKET_PATH = "data/market/historical_market_outcomes.parquet"


def _default_feature_view_path(context: dict[str, Any]) -> str:
    """Resolve the best default historical feature view path for the selected model."""

    return str(
        context.get("feature_view_output_path")
        or (context.get("config") or {}).get("feature_view_path")
        or ((context.get("config") or {}).get("data") or {}).get("rolling_features_path")
        or "data/features/moneyline_feature_view.parquet"
    )


def _render_latest_run_status() -> None:
    """Display the latest dispatched full-model-backtest workflow status when available."""

    ok, message, run = get_latest_workflow_run(WORKFLOW_FILE)
    if not ok:
        st.warning(message)
        return
    if not run:
        st.caption("No prior full model backtest workflow runs found.")
        return

    status = run.get("status", "unknown")
    conclusion = run.get("conclusion") or "pending"
    created_at = run.get("created_at") or ""
    html_url = run.get("html_url") or ""

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Status", str(status).title())
    c2.metric("Conclusion", str(conclusion).title())
    c3.metric("Created", created_at[:10] if created_at else "—")
    if html_url:
        st.link_button("Open Latest GitHub Run", html_url, use_container_width=True)


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

    st.html("<div class='mlab-card'><div class='mlab-section'><div class='mlab-section-title'>Full Model Backtest</div>")
    st.caption("Launches the GitHub Actions full-model backtest workflow using the selected model configuration.")

    model_config_path = st.text_input(
        "Model Config Path",
        value=str(context.get("config_path") or ""),
        key=f"mlab_backtest_config_path_{context['model_id']}",
    )
    feature_view_path = st.text_input(
        "Historical Feature View Path",
        value=_default_feature_view_path(context),
        key=f"mlab_backtest_feature_view_path_{context['model_id']}",
    )
    historical_market_path = st.text_input(
        "Historical Market Outcomes Path",
        value=DEFAULT_HISTORICAL_MARKET_PATH,
        key=f"mlab_backtest_market_path_{context['model_id']}",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        market_key = st.text_input(
            "Market Key",
            value=str(context.get("market_key") or "moneyline"),
            key=f"mlab_backtest_market_key_{context['model_id']}",
        )
    with c2:
        start_date = st.text_input(
            "Start Date",
            value="",
            placeholder="YYYY-MM-DD optional",
            key=f"mlab_backtest_start_date_{context['model_id']}",
        )
    with c3:
        end_date = st.text_input(
            "End Date",
            value="",
            placeholder="YYYY-MM-DD optional",
            key=f"mlab_backtest_end_date_{context['model_id']}",
        )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        min_edge = st.number_input(
            "Min Edge",
            value=0.00,
            step=0.01,
            min_value=0.0,
            max_value=1.0,
            key=f"mlab_backtest_min_edge_{context['model_id']}",
        )
    with p2:
        min_confidence = st.number_input(
            "Min Confidence",
            value=0.00,
            step=0.01,
            min_value=0.0,
            max_value=1.0,
            key=f"mlab_backtest_min_confidence_{context['model_id']}",
        )
    with p3:
        flat_stake = st.number_input(
            "Flat Stake",
            value=100,
            step=25,
            min_value=1,
            key=f"mlab_backtest_flat_stake_{context['model_id']}",
        )
    with p4:
        starting_bankroll = st.number_input(
            "Starting Bankroll",
            value=10000,
            step=500,
            min_value=1,
            key=f"mlab_backtest_starting_bankroll_{context['model_id']}",
        )

    required_ready = all([model_config_path, feature_view_path, historical_market_path, market_key])
    if st.button("Run Full Backtest", type="primary", disabled=not required_ready, use_container_width=True):
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
        ok, message = trigger_workflow(WORKFLOW_FILE, inputs=inputs)
        st.success(message) if ok else st.error(message)

    st.html("</div></div>")

    st.markdown("#### Latest Backtest Workflow Run")
    _render_latest_run_status()
