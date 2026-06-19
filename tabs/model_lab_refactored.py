from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, unquote

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


def _query_param_workspace() -> str | None:
    """Resolve the workspace from the URL query parameter, if present."""

    value = st.query_params.get("mlab_workspace")
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    workspace = unquote(str(value))
    return workspace if workspace in WORKSPACES else None


def _sync_workspace_from_legacy_sidebar() -> None:
    """Read the current legacy sidebar workspace without letting it own routing forever."""

    query_workspace = _query_param_workspace()
    if query_workspace:
        st.session_state["mlab_active_workspace"] = query_workspace
        return

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
            <div class="mlab-active-header-icon">▱</div>
            <div class="mlab-active-header-copy">
                <div class="mlab-active-title">{active}</div>
                <div class="mlab-active-subtitle">{description}</div>
            </div>
            <div class="mlab-active-loaded">◷&nbsp; Last saved: {now}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workspace_strip(active: str) -> str:
    """Render the internal Model Lab workspace navigation strip as text tabs."""

    st.markdown(
        """
        <style>
        .mlab-active-header {
            display: grid;
            grid-template-columns: 2.35rem minmax(0, 1fr) auto;
            align-items: start;
            gap: .9rem;
            margin: .05rem 0 .9rem;
            padding: .15rem 0 1rem;
            border-bottom: 1px solid rgba(43,60,82,.92);
        }
        .mlab-active-header-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 2.05rem;
            height: 2.05rem;
            border-radius: 9px;
            color: #9ed0ff;
            border: 1px solid rgba(96, 165, 250, .62);
            background: linear-gradient(180deg, rgba(11, 31, 57, .96), rgba(7, 20, 37, .98));
            font-size: 1.25rem;
            font-weight: 900;
            line-height: 1;
        }
        .mlab-active-title {
            color: #f8fbff;
            font-size: 1.85rem;
            font-weight: 950;
            letter-spacing: -.045em;
            line-height: 1.02;
        }
        .mlab-active-subtitle {
            color: #dbe7f5;
            font-size: .9rem;
            margin-top: .34rem;
            line-height: 1.35;
        }
        .mlab-active-loaded {
            color: #b9c8da;
            font-size: .78rem;
            font-weight: 720;
            white-space: nowrap;
            padding-top: .2rem;
        }
        .mlab-workspace-tabs {
            display: flex;
            align-items: center;
            gap: 2.05rem;
            margin: .25rem 0 0;
            padding: .15rem 0 0;
            border-bottom: 1px solid rgba(43,60,82,.82);
            overflow-x: auto;
            scrollbar-width: none;
        }
        .mlab-workspace-tabs::-webkit-scrollbar {
            display: none;
        }
        .mlab-workspace-tab {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            color: #b8c7da !important;
            text-decoration: none !important;
            font-size: .86rem;
            font-weight: 760;
            line-height: 1;
            padding: .95rem 0 1rem;
            border-bottom: 2px solid transparent;
            white-space: nowrap;
        }
        .mlab-workspace-tab:hover {
            color: #f8fbff !important;
            border-bottom-color: rgba(96,165,250,.5);
        }
        .mlab-workspace-tab.active {
            color: #5fb7ff !important;
            border-bottom-color: #2f9bff;
        }
        @media (max-width: 900px) {
            .mlab-active-header {
                grid-template-columns: 2.35rem minmax(0, 1fr);
            }
            .mlab-active-loaded {
                grid-column: 1 / -1;
                padding-top: 0;
            }
            .mlab-workspace-tabs {
                gap: 1.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    links = []
    for workspace in WORKSPACES:
        active_class = " active" if workspace == active else ""
        href = f"?mlab_workspace={quote(workspace)}"
        links.append(f'<a class="mlab-workspace-tab{active_class}" href="{href}">{workspace}</a>')
    st.markdown(f'<nav class="mlab-workspace-tabs">{"".join(links)}</nav>', unsafe_allow_html=True)
    return active


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