from __future__ import annotations

from typing import Any

import streamlit as st


def _option_index(options: list[str], current: str, default_index: int = 0) -> int:
    return options.index(current) if current in options else default_index


def render_behavior_section(context: dict[str, Any]) -> dict[str, Any]:
    """Render the Model Behavior card and return behavior payload."""

    editable = bool(context.get("is_editable"))
    config = context.get("config") or {}
    calibration = config.get("calibration") or {}
    prediction = config.get("prediction") or {}
    probability = prediction.get("probability") or {}
    threshold = prediction.get("threshold") or {}

    st.markdown("#### 3. Model Behavior")

    calibration_enabled = st.toggle(
        "Calibration Enabled",
        value=bool(calibration.get("enabled", True)),
        disabled=not editable,
        key="model_setup_behavior_calibration_enabled",
    )

    calibration_options = ["isotonic", "sigmoid", "none"]
    calibration_method = st.selectbox(
        "Calibration Method",
        calibration_options,
        index=_option_index(calibration_options, str(calibration.get("method") or "isotonic")),
        disabled=not editable,
        key="model_setup_behavior_calibration_method",
    )

    c1, c2 = st.columns(2)
    with c1:
        clip_low = st.number_input(
            "Probability Clip Low",
            value=float(probability.get("clip_low", 0.02)),
            step=0.01,
            min_value=0.0,
            max_value=0.49,
            disabled=not editable,
            key="model_setup_behavior_clip_low",
        )
    with c2:
        clip_high = st.number_input(
            "Probability Clip High",
            value=float(probability.get("clip_high", 0.98)),
            step=0.01,
            min_value=0.51,
            max_value=1.0,
            disabled=not editable,
            key="model_setup_behavior_clip_high",
        )

    threshold_options = ["fixed", "best_sweep", "model_card"]
    threshold_source = st.selectbox(
        "Threshold Source",
        threshold_options,
        index=_option_index(threshold_options, str(threshold.get("source") or "fixed")),
        disabled=not editable,
        key="model_setup_behavior_threshold_source",
    )
    threshold_value = st.number_input(
        "Threshold Value",
        value=float(threshold.get("value", 0.5)),
        step=0.01,
        min_value=0.0,
        max_value=1.0,
        disabled=not editable,
        key="model_setup_behavior_threshold_value",
    )

    return {
        "calibration_enabled": calibration_enabled,
        "calibration_method": calibration_method,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "threshold_source": threshold_source,
        "threshold_value": threshold_value,
    }
