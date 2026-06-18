from __future__ import annotations

from typing import Any

import streamlit as st


def render_behavior(
    calibration: dict[str, Any],
    probability: dict[str, Any],
    context: dict[str, Any],
    *,
    editable: bool,
) -> dict[str, Any]:
    """Render calibration, probability clipping, and dashboard selection controls."""

    calibration_enabled = st.toggle(
        "Calibration Enabled",
        value=bool(calibration.get("enabled", True)),
        disabled=not editable,
        key="mlab_cal_enabled",
    )
    calibration_options = ["isotonic", "sigmoid", "none"]
    calibration_current = str(calibration.get("method", "isotonic"))
    calibration_method = st.selectbox(
        "Calibration Method",
        calibration_options,
        index=calibration_options.index(calibration_current) if calibration_current in calibration_options else 0,
        disabled=not editable,
        key="mlab_cal_method",
    )
    clip_low = st.number_input(
        "Probability Clip Low",
        value=float(probability.get("clip_low", 0.02)),
        step=0.01,
        min_value=0.0,
        max_value=0.49,
        disabled=not editable,
        key="mlab_clip_low",
    )
    clip_high = st.number_input(
        "Probability Clip High",
        value=float(probability.get("clip_high", 0.98)),
        step=0.01,
        min_value=0.51,
        max_value=1.0,
        disabled=not editable,
        key="mlab_clip_high",
    )
    dashboard_selectable = st.toggle(
        "Dashboard Selectable",
        value=bool(context.get("dashboard_selectable", False)),
        disabled=not editable,
        key="mlab_dashboard_selectable",
    )
    return {
        "calibration_enabled": calibration_enabled,
        "calibration_method": calibration_method,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "dashboard_selectable": dashboard_selectable,
    }
