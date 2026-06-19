from __future__ import annotations

import re
from typing import Any

import streamlit as st


def _safe_widget_key(value: Any) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value or ""))
    return cleaned.strip("_") or "model"


def _hyperparameter_key(context: dict[str, Any], suffix: str) -> str:
    model_key = _safe_widget_key(context.get("model_id") or context.get("config_path") or "model")
    return f"model_setup_hyperparameters_{suffix}_{model_key}"


def render_hyperparameters_section(context: dict[str, Any]) -> dict[str, Any]:
    """Render the Hyperparameters card and return params payload."""

    editable = bool(context.get("is_editable"))
    params = (context.get("config") or {}).get("params") or {}

    st.markdown("#### :blue[4.] Hyperparameters")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_estimators = st.number_input("N Estimators", value=int(params.get("n_estimators", 500)), step=50, min_value=50, disabled=not editable, key=_hyperparameter_key(context, "n_estimators"))
    with c2:
        max_depth = st.number_input("Max Depth", value=int(params.get("max_depth", 4)), step=1, min_value=1, max_value=20, disabled=not editable, key=_hyperparameter_key(context, "max_depth"))
    with c3:
        learning_rate = st.number_input("Learning Rate", value=float(params.get("learning_rate", 0.03)), step=0.01, min_value=0.001, max_value=1.0, disabled=not editable, key=_hyperparameter_key(context, "learning_rate"))

    c4, c5, c6 = st.columns(3)
    with c4:
        subsample = st.number_input("Subsample", value=float(params.get("subsample", 0.8)), step=0.05, min_value=0.1, max_value=1.0, disabled=not editable, key=_hyperparameter_key(context, "subsample"))
    with c5:
        colsample_bytree = st.number_input("Colsample Bytree", value=float(params.get("colsample_bytree", 0.8)), step=0.05, min_value=0.1, max_value=1.0, disabled=not editable, key=_hyperparameter_key(context, "colsample_bytree"), label_visibility="visible")
    with c6:
        random_state = st.number_input("Random State", value=int(params.get("random_state", 42)), step=1, min_value=0, disabled=not editable, key=_hyperparameter_key(context, "random_state"))

    eval_metric = st.text_input("Eval Metric", value=str(params.get("eval_metric", "logloss")), disabled=not editable, key=_hyperparameter_key(context, "eval_metric"))

    return {
        "params": {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "random_state": int(random_state),
            "eval_metric": eval_metric,
        }
    }