from __future__ import annotations

from typing import Any, Callable

import streamlit as st

import utils.model_lab_workflows as mlw


ExistingModelSelector = Callable[[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]], dict[str, Any]]


def render_comparison(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    existing_model_selector: ExistingModelSelector,
) -> None:
    st.markdown("## Comparison")
    context = existing_model_selector(registry, rows, row_by_id)
    mlw._render_comparison(context, registry, context["model_id"])
