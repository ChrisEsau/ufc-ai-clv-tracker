from __future__ import annotations

from typing import Any

import streamlit as st


def _model_label(row: dict[str, Any]) -> str:
    model_id = str(row.get("model_id") or "")
    status = str(row.get("status") or "draft").upper()
    market = str(row.get("market_key") or "moneyline")
    return f"{model_id}  ·  {status}  ·  {market}"


def render_model_selector(rows: list[dict[str, Any]]) -> str | None:
    """Render the existing-model selector.

    The selector only loads existing models. New model creation is prepared by
    the page-level New button and persisted by Save when the generated ID does
    not yet exist in the registry.
    """

    if not rows:
        st.info("No registered models found.")
        return None

    model_ids = [str(row["model_id"]) for row in rows]
    row_by_id = {str(row["model_id"]): row for row in rows}
    current = st.session_state.get("model_setup_selected_model_id", model_ids[0])
    index = model_ids.index(current) if current in model_ids else 0

    selected = st.selectbox(
        "Model",
        model_ids,
        index=index,
        format_func=lambda model_id: _model_label(row_by_id[model_id]),
        key="model_setup_selected_model_id",
        help="Select an existing model to view/edit or use as the template for New.",
    )
    return str(selected)
