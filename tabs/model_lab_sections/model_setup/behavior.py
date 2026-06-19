from __future__ import annotations

import re
from typing import Any

import streamlit as st


def _option_index(options: list[str], current: str, default_index: int = 0) -> int:
    return options.index(current) if current in options else default_index


def _safe_widget_key(value: Any) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value or ""))
    return cleaned.strip("_") or "model"


def _behavior_key(context: dict[str, Any], suffix: str) -> str:
    model_key = _safe_widget_key(context.get("model_id") or context.get("config_path") or "model")
    return f"model_setup_behavior_{suffix}_{model_key}"


def render_behavior_section(context: dict[str, Any]) -> dict[str, Any]:
    """Render the Model Behavior card and return behavior payload."""

    editable = bool(context.get("is_editable"))
    config = context.get("config") or {}
    calibration = config.get("calibration") or {}
    prediction = config.get("prediction") or {}
    probability = prediction.get("probability") or {}
    threshold = prediction.get("threshold") or {}
    symmetry = config.get("symmetry") or {}

    st.markdown("#### 3. Model Behavior")

    calibration_enabled = st.toggle("Calibration Enabled", value=bool(calibration.get("enabled", True)), disabled=not editable, key=_behavior_key(context, "calibration_enabled"))

    calibration_options = ["isotonic", "sigmoid", "none"]
    calibration_method = st.selectbox("Calibration Method", calibration_options, index=_option_index(calibration_options, str(calibration.get("method") or "isotonic")), disabled=not editable, key=_behavior_key(context, "calibration_method"))

    c1, c2 = st.columns(2)
    with c1:
        clip_low = st.number_input("Probability Clip Low", value=float(probability.get("clip_low", 0.02)), step=0.01, min_value=0.0, max_value=0.49, disabled=not editable, key=_behavior_key(context, "clip_low"))
    with c2:
        clip_high = st.number_input("Probability Clip High", value=float(probability.get("clip_high", 0.98)), step=0.01, min_value=0.51, max_value=1.0, disabled=not editable, key=_behavior_key(context, "clip_high"))

    threshold_options = ["fixed", "best_sweep", "model_card"]
    t1, t2 = st.columns(2)
    with t1:
        threshold_source = st.selectbox("Threshold Source", threshold_options, index=_option_index(threshold_options, str(threshold.get("source") or "fixed")), disabled=not editable, key=_behavior_key(context, "threshold_source"))
    with t2:
        threshold_value = st.number_input("Threshold Value", value=float(threshold.get("value", 0.5)), step=0.01, min_value=0.0, max_value=1.0, disabled=not editable, key=_behavior_key(context, "threshold_value"))

    symmetry_options = ["flip_all", "none"]
    current_symmetry_mode = str(symmetry.get("mode") or "none")
    if current_symmetry_mode not in symmetry_options:
        symmetry_options.append(current_symmetry_mode)

    s1, s2 = st.columns([0.9, 1.1])
    with s1:
        symmetry_enabled = st.toggle("Symmetry Enabled", value=bool(symmetry.get("enabled", False)), disabled=not editable, key=_behavior_key(context, "symmetry_enabled"))
    with s2:
        symmetry_mode = st.selectbox("Symmetry Mode", symmetry_options, index=_option_index(symmetry_options, current_symmetry_mode), disabled=not editable, key=_behavior_key(context, "symmetry_mode"), help="Controls red/blue mirrored training augmentation.")

    return {
        "calibration_enabled": calibration_enabled,
        "calibration_method": calibration_method,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "threshold_source": threshold_source,
        "threshold_value": threshold_value,
        "symmetry_enabled": symmetry_enabled,
        "symmetry_mode": symmetry_mode,
    }