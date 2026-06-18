from __future__ import annotations

from typing import Any

import streamlit as st


def render_hyperparameters(params: dict[str, Any], *, editable: bool) -> dict[str, Any]:
    """Render XGBoost hyperparameter controls."""

    st.markdown("##### XGBoost Parameters")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        n_estimators = st.number_input(
            "N Estimators",
            value=int(params.get("n_estimators", 500)),
            step=50,
            min_value=50,
            disabled=not editable,
            key="mlab_n_estimators",
        )
    with p2:
        max_depth = st.number_input(
            "Max Depth",
            value=int(params.get("max_depth", 4)),
            step=1,
            min_value=1,
            max_value=12,
            disabled=not editable,
            key="mlab_max_depth",
        )
    with p3:
        learning_rate = st.number_input(
            "Learning Rate",
            value=float(params.get("learning_rate", 0.03)),
            step=0.01,
            min_value=0.001,
            max_value=1.0,
            disabled=not editable,
            key="mlab_learning_rate",
        )
    with p4:
        subsample = st.number_input(
            "Subsample",
            value=float(params.get("subsample", 0.8)),
            step=0.05,
            min_value=0.1,
            max_value=1.0,
            disabled=not editable,
            key="mlab_subsample",
        )
    with p5:
        colsample = st.number_input(
            "Colsample",
            value=float(params.get("colsample_bytree", 0.8)),
            step=0.05,
            min_value=0.1,
            max_value=1.0,
            disabled=not editable,
            key="mlab_colsample",
        )

    return {
        "params": {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample),
            "random_state": int(params.get("random_state", 42)),
            "eval_metric": params.get("eval_metric", "logloss"),
        }
    }
