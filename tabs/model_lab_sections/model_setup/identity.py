from __future__ import annotations

import re
from typing import Any

import streamlit as st


FAMILY_OPTIONS = ["moneyline", "prop"]
MARKET_OPTIONS = [
    "moneyline",
    "goes_distance",
    "inside_distance",
    "ko_tko",
    "submission",
    "decision",
    "over_1_5",
    "over_2_5",
    "over_3_5",
]
ALGORITHM_OPTIONS = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]


def _option_index(options: list[str], current: str, default_index: int = 0) -> int:
    return options.index(current) if current in options else default_index


def _safe_widget_key(value: Any) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value or ""))
    return cleaned.strip("_") or "model"


def _identity_key(context: dict[str, Any], suffix: str) -> str:
    """Return a model-scoped key for identity widgets."""

    model_key = _safe_widget_key(context.get("model_id") or context.get("config_path") or "model")
    return f"model_setup_identity_{suffix}_{model_key}"


def render_identity_section(context: dict[str, Any]) -> dict[str, Any]:
    """Render the Model Identity card and return identity payload."""

    editable = bool(context.get("is_editable"))
    config = context.get("config") or {}
    current_family = str(context.get("model_family") or "moneyline").strip().lower()
    current_market = str(context.get("market_key") or "moneyline").strip().lower()
    current_algorithm = str(context.get("algorithm") or config.get("algorithm") or "xgboost").strip().lower()

    family_options = FAMILY_OPTIONS.copy()
    if current_family and current_family not in family_options:
        family_options.append(current_family)
    market_options = MARKET_OPTIONS.copy()
    if current_market and current_market not in market_options:
        market_options.append(current_market)
    algorithm_options = ALGORITHM_OPTIONS.copy()
    if current_algorithm and current_algorithm not in algorithm_options:
        algorithm_options.append(current_algorithm)

    st.markdown('<div class="model-setup-card-heading"><span>1.</span> Model Identity</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        model_id = st.text_input(
            "Generated Model ID",
            value=str(context.get("model_id") or ""),
            disabled=True,
            key=_identity_key(context, "model_id"),
        )
        model_family = st.selectbox(
            "Family",
            family_options,
            index=_option_index(family_options, current_family),
            disabled=not editable,
            key=_identity_key(context, "family"),
        )
        algorithm = st.selectbox(
            "Algorithm",
            algorithm_options,
            index=_option_index(algorithm_options, current_algorithm),
            disabled=not editable,
            key=_identity_key(context, "algorithm"),
        )
    with c2:
        status = st.text_input(
            "Status",
            value=str(context.get("status") or "draft"),
            disabled=True,
            key=_identity_key(context, "status"),
        )
        market_key = st.selectbox(
            "Market",
            market_options,
            index=_option_index(market_options, current_market),
            disabled=not editable,
            key=_identity_key(context, "market"),
        )

    description = st.text_area(
        "Description",
        value=str(context.get("description") or ""),
        disabled=not editable,
        height=86,
        key=_identity_key(context, "description"),
    )

    display_name = str(context.get("display_name") or context.get("model_id") or "")
    dashboard_selectable = bool(context.get("dashboard_selectable", False))

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