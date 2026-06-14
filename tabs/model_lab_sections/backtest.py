from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

import utils.model_lab_workflows as mlw
from utils.github_actions import get_latest_workflow_run, trigger_workflow


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]

WORKFLOW_FILE = "run-full-model-backtest-v2.yml"
BACKTEST_ROOT = Path("data/model_lab/backtests")
DEFAULT_HISTORICAL_MARKET_PATH = "data/market/historical_market_outcomes.parquet"
DEFAULT_FEATURE_VIEW_PATH = "data/features/moneyline_feature_view.parquet"


def _default_feature_view_path(context: dict[str, Any]) -> str:
    """Resolve the best default historical feature view path for the selected model."""

    return str(
        context.get("feature_view_output_path")
        or (context.get("config") or {}).get("feature_view_path")
        or ((context.get("config") or {}).get("data") or {}).get("rolling_features_path")
        or DEFAULT_FEATURE_VIEW_PATH
    )


def _fmt_pct(value: Any, decimals: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(number) <= 1:
        number *= 100
    return f"{number:.{decimals}f}%"


def _fmt_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"${number:,.0f}"


def _fmt_num(value: Any, decimals: int = 0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:,.{decimals}f}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _latest_backtest_dir(model_id: str, market_key: str) -> Path | None:
    """Find the newest committed backtest artifact folder for this model/market."""

    if not BACKTEST_ROOT.exists():
        return None
    prefix = f"{model_id}_{market_key}"
    candidates = [path for path in BACKTEST_ROOT.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    if not candidates:
        candidates = [path for path in BACKTEST_ROOT.iterdir() if path.is_dir() and model_id in path.name]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[-1]


def _render_summary_cards(summary: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bets", _fmt_num(summary.get("total_bets")))
    c2.metric("Win Rate", _fmt_pct(summary.get("win_rate")))
    c3.metric("Flat Profit", _fmt_money(summary.get("flat_profit")))
    c4.metric("Flat ROI", _fmt_pct(summary.get("flat_roi")))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Kelly Profit", _fmt_money(summary.get("kelly_profit")))
    k2.metric("Kelly ROI", _fmt_pct(summary.get("kelly_roi")))
    k3.metric("Ending Bankroll", _fmt_money(summary.get("ending_kelly_bankroll")))
    k4.metric("Joined Rows", _fmt_num(summary.get("joined_rows")))


def _render_summary_details(summary: dict[str, Any], artifact_dir: Path) -> None:
    detail = {
        "Backtest ID": summary.get("backtest_id") or artifact_dir.name,
        "Model ID": summary.get("model_id"),
        "Market": summary.get("market_key"),
        "Mode": summary.get("mode"),
        "Date Range": f"{summary.get('start_date') or 'Beginning'} → {summary.get('end_date') or 'Latest'}",
        "Min Edge": _fmt_pct(summary.get("min_edge"), decimals=2),
        "Min Confidence": _fmt_pct(summary.get("min_confidence"), decimals=2),
        "Feature Rows": _fmt_num(summary.get("feature_rows")),
        "Historical Outcome Rows": _fmt_num(summary.get("historical_model_outcome_rows")),
    }
    st.dataframe(pd.DataFrame([detail]).T.rename(columns={0: "Value"}), use_container_width=True)


def _render_artifact_tables(artifact_dir: Path) -> None:
    candidate_files = [
        "backtest_results.parquet",
        "backtest_bets.parquet",
        "bet_results.parquet",
        "backtest_results.csv",
        "backtest_bets.csv",
        "bet_results.csv",
    ]
    for filename in candidate_files:
        path = artifact_dir / filename
        table = _read_table(path)
        if not table.empty:
            st.markdown(f"##### Bet-Level Preview: `{filename}`")
            st.dataframe(table.head(250), use_container_width=True, hide_index=True)
            if len(table) > 250:
                st.caption(f"Showing first 250 of {len(table):,} rows.")
            return
    st.info("No bet-level result table was found in the latest backtest artifact folder. The summary/config artifacts are available.")


def _render_latest_results(context: dict[str, Any]) -> None:
    model_id = str(context.get("model_id") or "")
    market_key = str(context.get("market_key") or "moneyline")
    artifact_dir = _latest_backtest_dir(model_id, market_key)

    st.markdown("#### Latest Backtest Results")
    if artifact_dir is None:
        st.info("No backtest artifacts found yet for this model. Run a backtest, then refresh after GitHub Actions commits the artifacts.")
        return

    summary = _read_json(artifact_dir / "backtest_summary.json")
    config = _read_json(artifact_dir / "backtest_config.json")
    if not summary:
        st.warning(f"Found `{artifact_dir}` but no readable `backtest_summary.json`.")
        return

    st.caption(f"Artifact folder: `{artifact_dir}`")
    _render_summary_cards(summary)

    tabs = st.tabs(["Summary", "Bet Preview", "Config"])
    with tabs[0]:
        _render_summary_details(summary, artifact_dir)
    with tabs[1]:
        _render_artifact_tables(artifact_dir)
    with tabs[2]:
        if config:
            st.json(config)
        else:
            st.info("No readable backtest_config.json found for this run.")


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


def _render_run_controls(context: dict[str, Any]) -> None:
    model_config_path = str(context.get("config_path") or "")
    feature_view_path = _default_feature_view_path(context)
    historical_market_path = DEFAULT_HISTORICAL_MARKET_PATH
    market_key = str(context.get("market_key") or "moneyline")

    st.markdown("#### Run New Backtest")
    st.caption("Uses the selected model automatically. Technical paths are hidden unless you open Advanced paths.")

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.text_input(
            "Start Date",
            value="",
            placeholder="YYYY-MM-DD optional",
            key=f"mlab_backtest_start_date_{context['model_id']}",
        )
    with c2:
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

    with st.expander("Advanced paths", expanded=False):
        model_config_path = st.text_input(
            "Model Config Path",
            value=model_config_path,
            key=f"mlab_backtest_config_path_{context['model_id']}",
        )
        feature_view_path = st.text_input(
            "Historical Feature View Path",
            value=feature_view_path,
            key=f"mlab_backtest_feature_view_path_{context['model_id']}",
        )
        historical_market_path = st.text_input(
            "Historical Market Outcomes Path",
            value=historical_market_path,
            key=f"mlab_backtest_market_path_{context['model_id']}",
        )
        market_key = st.text_input(
            "Market Key",
            value=market_key,
            key=f"mlab_backtest_market_key_{context['model_id']}",
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
    _render_run_controls(context)

    with st.expander("GitHub workflow status", expanded=False):
        _render_latest_run_status()
