from __future__ import annotations

from typing import Any, Callable

import streamlit as st

import utils.model_lab_workflows as mlw
from tabs.model_lab_sections.performance import render_multiclass_summary


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def render_overview(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    existing_model_selector: ExistingModelSelector,
) -> None:
    st.markdown("## Overview")
    context = existing_model_selector(registry, rows, row_by_id)
    mlw._render_kpis(context)
    mlw._render_model_bar(context, registry)
    render_multiclass_summary(context, compact=True)
    mlw._render_registry_table(rows)
