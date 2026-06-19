from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

from tabs import model_lab as legacy_model_lab
from tabs.model_lab_sections.actions import render_actions
from tabs.model_lab_sections.backtest_enhanced import render_backtest
from tabs.model_lab_sections.compare import render_compare
from tabs.model_lab_sections.features import render_features
from tabs.model_lab_sections.model_setup.page import render_page as render_model_setup_page
from tabs.model_lab_sections.overview import render_overview
from tabs.model_lab_sections.performance import render_performance
from tabs.model_lab_sections.styles import inject_model_lab_control_css
import utils.model_lab_workflows as mlw


WORKSPACES = ["Overview", "Configuration", "Model Setup", "Features", "Performance", "Compare", "Backtest", "Actions"]
DEFAULT_WORKSPACE = "Configuration"
WORKSPACE_DESCRIPTIONS = {
    "Overview": "Review registered models, status, and high-level model health.",
    "Configuration": "Configure and maintain legacy model config settings.",
    "Model Setup": "Configure model identity, training setup, behavior, hyperparameters, and feature selection.",
    "Features": "Review feature registry coverage, bundles, and model feature inputs.",
    "Performance": "Inspect model metrics, calibration, and training artifacts.",
    "Compare": "Compare model configurations and performance side by side.",
    "Backtest": "Run and review model backtest outputs.",
    "Actions": "Trigger model workflows and lifecycle operations.",
}
LEGACY_WORKSPACE_MAP = {
    "Overview": "Overview",
    "Configuration": "Configuration",
    "Model Setup": "Model Setup",
    "Features": "Features",
    "Performance": "Performance",
    "Comparison": "Compare",
    "Compare": "Compare",
    "Backtest": "Backtest",
    "Actions": "Actions",
}


def _select_existing_model_with_key(key: str):
    """Adapt the legacy selector to the section-module callable signature."""

    def selector(
        registry: dict[str, Any],
        rows: list[dict[str, Any]],
        row_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return legacy_model_lab._existing_model_selector(
            registry,
            rows,
            row_by_id,
            key=key,
        )

    return selector


def _sync_workspace_from_legacy_sidebar() -> None:
    """Read the current legacy sidebar workspace without letting it own routing forever."""

    legacy_workspace = st.session_state.get("sidebar_model_lab_workspace")
    previous_legacy = st.session_state.get("mlab_last_sidebar_workspace_seen")
    if legacy_workspace == previous_legacy:
        return

    st.session_state["mlab_last_sidebar_workspace_seen"] = legacy_workspace

    if legacy_workspace in LEGACY_WORKSPACE_MAP:
        st.session_state["mlab_active_workspace"] = LEGACY_WORKSPACE_MAP[legacy_workspace]
    elif st.session_state.get("mlab_active_workspace") not in WORKSPACES:
        st.session_state["mlab_active_workspace"] = DEFAULT_WORKSPACE


def _active_workspace() -> str:
    _sync_workspace_from_legacy_sidebar()
    active = st.session_state.get("mlab_active_workspace", DEFAULT_WORKSPACE)
    if active not in WORKSPACES:
        active = DEFAULT_WORKSPACE
        st.session_state["mlab_active_workspace"] = active
    return str(active)


def _render_workspace_header(active: str) -> None:
    """Render the active workspace as the page header."""

    now = datetime.now(timezone.utc).strftime("%b %-d, %Y %I:%M %p UTC")
    description = WORKSPACE_DESCRIPTIONS.get(active, "Build, tune, compare, and promote predictive models.")
    st.markdown(
        f"""
        <div class="mlab-active-header">
            <div>
                <div class="mlab-active-title">{active}</div>
                <div class="mlab-active-subtitle">{description}</div>
            </div>
            <div class="mlab-active-loaded">Last Loaded: {now}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workspace_strip(active: str) -> str:
    """Render the internal Model Lab workspace navigation strip."""

    st.markdown(
        """
        <style>
        .mlab-active-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin: .1rem 0 .85rem;
            padding-bottom: .85rem;
            border-bottom: 1px solid rgba(43,60,82,.92);
        }
        .mlab-active-title {
            color: #f8fbff;
            font-size: 1.72rem;
            font-weight: 950;
            letter-spacing: -.045em;
            line-height: 1.05;
        }
        .mlab-active-subtitle {
            color: #dbe7f5;
            font-size: .9rem;
            margin-top: .32rem;
        }
        .mlab-active-loaded {
            color: #dbe7f5;
            font-size: .75rem;
            font-weight: 760;
            white-space: nowrap;
            padding-top: .1rem;
        }
        .mlab-workspace-strip-caption {
            color: #8fb3db;
            font-size: .68rem;
            margin: 0 0 .35rem;
            text-transform: uppercase;
            letter-spacing: .06em;
            font-weight: 900;
        }
        div[data-testid="stHorizontalBlock"] button {
            min-height: 2.38rem !important;
            border-radius: 8px !important;
            font-weight: 790 !important;
            letter-spacing: -.01em !important;
            border: 1px solid rgba(45, 72, 108, .92) !important;
            box-shadow: none !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background: linear-gradient(180deg, rgba(15, 36, 68, .96), rgba(8, 25, 48, .98)) !important;
            color: #5fb7ff !important;
            border-color: rgba(59, 130, 246, .78) !important;
            box-shadow: inset 0 -2px 0 #2f9bff !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: linear-gradient(180deg, rgba(8, 22, 41, .72), rgba(6, 17, 31, .84)) !important;
            color: #dbeafe !important;
        }
        div[data-testid="stHorizontalBlock"] button:hover {
            border-color: rgba(96, 165, 250, .9) !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='mlab-workspace-strip-caption'>Workspace</div>", unsafe_allow_html=True)

    columns = st.columns(len(WORKSPACES), gap="small")
    for column, workspace in zip(columns, WORKSPACES):
        with column:
            if st.button(
                workspace,
                use_container_width=True,
                type="primary" if workspace == active else "secondary",
                key=f"mlab_workspace_{workspace}",
            ):
                st.session_state["mlab_active_workspace"] = workspace
                st.rerun()

    return str(st.session_state.get("mlab_active_workspace", DEFAULT_WORKSPACE))


def render_model_lab() -> None:
    """Render Model Lab through the internal workspace router."""

    mlw._inject_css()
    inject_model_lab_control_css()
    active = _active_workspace()
    _render_workspace_header(active)

    try:
        registry, rows, row_by_id = legacy_model_lab._load_registry_rows()
        if not rows:
            st.info("No models are registered in configs/models/model_registry.yaml.")
            return
        workspace = _render_workspace_strip(active)

        if workspace == "Overview":
            render_overview(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_overview_model"),
            )
        elif workspace == "Configuration":
            legacy_model_lab._render_configuration(registry, rows, row_by_id)
        elif workspace == "Model Setup":
            render_model_setup_page()
        elif workspace == "Features":
            render_features(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_features_model"),
            )
        elif workspace == "Performance":
            render_performance(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_performance_model"),
            )
        elif workspace == "Compare":
            render_compare(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_compare_model"),
            )
        elif workspace == "Backtest":
            render_backtest(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_backtest_model"),
            )
        elif workspace == "Actions":
            render_actions(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_actions_model"),
            )
    except Exception as exc:
        st.error(f"Unable to render Model Lab: {exc}")
