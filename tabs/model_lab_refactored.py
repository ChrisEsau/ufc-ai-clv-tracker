from __future__ import annotations

from typing import Any

import streamlit as st

from tabs import model_lab as legacy_model_lab
from tabs.model_lab_sections.actions import render_actions
from tabs.model_lab_sections.backtest_enhanced import render_backtest
from tabs.model_lab_sections.compare import render_compare
from tabs.model_lab_sections.features import render_features
from tabs.model_lab_sections.model_setup import render_model_setup
from tabs.model_lab_sections.overview import render_overview
from tabs.model_lab_sections.performance import render_performance
from tabs.model_lab_sections.styles import inject_model_lab_control_css
import utils.model_lab_workflows as mlw


WORKSPACES = ["Overview", "Model Setup", "Features", "Performance", "Compare", "Backtest", "Actions"]
DEFAULT_WORKSPACE = "Model Setup"
LEGACY_WORKSPACE_MAP = {
    "Overview": "Overview",
    "Configuration": "Model Setup",
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
    """Read the current legacy sidebar workspace without letting it own routing forever.

    The sidebar still exists during this phase. We only remap it when the sidebar
    value changes, so clicks on the new internal Model Lab strip are not
    overwritten on every rerun.
    """

    legacy_workspace = st.session_state.get("sidebar_model_lab_workspace")
    previous_legacy = st.session_state.get("mlab_last_sidebar_workspace_seen")
    if legacy_workspace == previous_legacy:
        return

    st.session_state["mlab_last_sidebar_workspace_seen"] = legacy_workspace

    if legacy_workspace in LEGACY_WORKSPACE_MAP:
        st.session_state["mlab_active_workspace"] = LEGACY_WORKSPACE_MAP[legacy_workspace]
    elif st.session_state.get("mlab_active_workspace") not in WORKSPACES:
        st.session_state["mlab_active_workspace"] = DEFAULT_WORKSPACE


def _render_workspace_strip() -> str:
    """Render the internal Model Lab workspace navigation strip."""

    _sync_workspace_from_legacy_sidebar()
    active = st.session_state.get("mlab_active_workspace", DEFAULT_WORKSPACE)
    if active not in WORKSPACES:
        active = DEFAULT_WORKSPACE
        st.session_state["mlab_active_workspace"] = active

    st.markdown(
        """
        <style>
        .mlab-workspace-strip-caption {
            color: #dbe7f5;
            font-size: .76rem;
            margin: -.15rem 0 .35rem;
        }
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            font-weight: 900 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='mlab-workspace-strip-caption'>Model Lab Workspace</div>",
        unsafe_allow_html=True,
    )

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
    """Render Model Lab through the internal workspace router.

    This phase leaves the legacy sidebar untouched, but Model Lab now owns its
    page-level workspace routing through the internal strip.
    """

    mlw._inject_css()
    inject_model_lab_control_css()
    mlw._render_header()

    try:
        registry, rows, row_by_id = legacy_model_lab._load_registry_rows()
        if not rows:
            st.info("No models are registered in configs/models/model_registry.yaml.")
            return

        workspace = _render_workspace_strip()

        if workspace == "Overview":
            render_overview(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_overview_model"),
            )
        elif workspace == "Model Setup":
            render_model_setup(registry, rows, row_by_id)
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
