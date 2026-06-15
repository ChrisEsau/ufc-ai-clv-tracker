from __future__ import annotations

from typing import Any

import streamlit as st

from tabs import model_lab as legacy_model_lab
from tabs.model_lab_sections.actions import render_actions
from tabs.model_lab_sections.backtest_enhanced import render_backtest
from tabs.model_lab_sections.comparison import render_comparison
from tabs.model_lab_sections.features import render_features
from tabs.model_lab_sections.overview import render_overview
from tabs.model_lab_sections.performance import render_performance
from tabs.model_lab_sections.styles import inject_model_lab_control_css
import utils.model_lab_workflows as mlw


WORKSPACES = ["Overview", "Configuration", "Features", "Performance", "Backtest", "Comparison", "Actions"]


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


def render_model_lab() -> None:
    """Render Model Lab through modular workspace sections.

    Phase 1 keeps Configuration and workflow logic in the legacy module while
    routing low-risk workspaces through section modules.
    """

    mlw._inject_css()
    inject_model_lab_control_css()
    mlw._render_header()

    try:
        registry, rows, row_by_id = legacy_model_lab._load_registry_rows()
        if not rows:
            st.info("No models are registered in configs/models/model_registry.yaml.")
            return

        workspace = st.session_state.get("sidebar_model_lab_workspace", "Configuration")
        if workspace not in WORKSPACES:
            workspace = "Configuration"

        if workspace == "Overview":
            render_overview(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_overview_model"),
            )
        elif workspace == "Configuration":
            legacy_model_lab._render_configuration(registry, rows, row_by_id)
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
        elif workspace == "Backtest":
            render_backtest(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_backtest_model"),
            )
        elif workspace == "Comparison":
            render_comparison(
                registry,
                rows,
                row_by_id,
                existing_model_selector=_select_existing_model_with_key("mlab_comparison_model"),
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
