from __future__ import annotations

from typing import Any

import streamlit as st


def render_identity_section(context: dict[str, Any]) -> dict[str, Any]:
    """Render the Model Identity card and return identity payload."""

    editable = bool(context.get("is_editable"))
    config = context.get("config") or {}

    st.markdown("#### 1. Model Identity")
    c1, c2 = st.columns(2)
    with c1:
        model_id = st.text_input(
            "Generated Model ID",
            value=str(context.get("model_id") or ""),
            disabled=True,
            key="model_setup_identity_model_id",
        )
        model_family = st.text_input(
            "Family",
            value=str(context.get("model_family") or ""),
            disabled=True,
            key="model_setup_identity_family",
        )
        algorithm = st.text_input(
            "Algorithm",
            value=str(context.get("algorithm") or config.get("algorithm") or "xgboost"),
            disabled=True,
            key="model_setup_identity_algorithm",
        )
    with c2:
        status = st.text_input(
            "Status",
            value=str(context.get("status") or "draft"),
            disabled=True,
            key="model_setup_identity_status",
        )
        market_key = st.text_input(
            "Market",
            value=str(context.get("market_key") or "moneyline"),
            disabled=True,
            key="model_setup_identity_market",
        )
        dashboard_selectable = st.toggle(
            "Dashboard Selectable",
            value=bool(context.get("dashboard_selectable", False)),
            disabled=not editable,
            key="model_setup_identity_dashboard_selectable",
        )

    display_name = st.text_input(
        "Display Name",
        value=str(context.get("display_name") or context.get("model_id") or ""),
        disabled=not editable,
        key="model_setup_identity_display_name",
    )
    description = st.text_area(
        "Description",
        value=str(context.get("description") or ""),
        disabled=not editable,
        height=86,
        key="model_setup_identity_description",
    )

    return {
        "model_id": model_id,
        "status": status,
        "model_family": model_family,
        "market_key": market_key,
        "algorithm": algorithm,
        "display_name": display_name,
        "description": description,
        "dashboard_selectable": dashboard_selectable,
    }
