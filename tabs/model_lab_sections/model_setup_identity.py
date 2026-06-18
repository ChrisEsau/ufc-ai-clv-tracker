from __future__ import annotations

from typing import Any

import streamlit as st


def render_identity(context: dict[str, Any], *, editable: bool) -> dict[str, Any]:
    """Render model identity fields and return their form values."""

    display_name = st.text_input(
        "Display Name",
        value=str(context.get("display_name") or context["model_id"]),
        disabled=not editable,
        key="mlab_display_name",
    )
    description = st.text_area(
        "Description",
        value=str(context.get("description") or ""),
        disabled=not editable,
        height=72,
        key="mlab_description",
    )
    return {
        "display_name": display_name,
        "description": description,
    }
