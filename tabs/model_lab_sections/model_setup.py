from __future__ import annotations

from copy import deepcopy
from typing import Any

import streamlit as st

from tabs import model_lab as legacy_model_lab
from tabs.model_lab_sections.model_setup_identity import render_identity
from tabs.model_lab_sections.model_setup_training import render_training
from tabs.model_lab_sections.model_setup_behavior import render_behavior
from tabs.model_lab_sections.model_setup_hyperparameters import render_hyperparameters
from tabs.model_lab_sections import model_setup_feature_selection  # noqa: F401
from tabs.model_lab_sections import model_setup_advanced  # noqa: F401
import utils.model_lab_workflows as mlw


def _render_config_editor(context: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    """Render editable base model configuration through modular section controls."""

    config = deepcopy(context["config"])
    editable = context["status"] in mlw.EDITABLE_STATUSES or bool(context.get("is_new_model"))
    split = config.setdefault("split", {})
    calibration = config.setdefault("calibration", {})
    params = config.setdefault("params", {})
    probability = config.setdefault("prediction", {}).setdefault("probability", {})

    st.html("<div class='mlab-card'><div class='mlab-section'><div class='mlab-section-title'>Configuration</div>")
    if not editable:
        st.caption("Read-only because this model is not draft.")

    identity_values = render_identity(context, editable=editable)

    c1, c2 = st.columns(2)
    with c1:
        training_values = render_training(split, editable=editable)
        behavior_left_values = {}
        # Keep calibration controls in the left column to preserve the legacy layout.
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
        behavior_left_values = {
            "calibration_enabled": calibration_enabled,
            "calibration_method": calibration_method,
        }
    with c2:
        # Keep probability and dashboard controls in the right column to preserve the legacy layout.
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
        behavior_right_values = {
            "clip_low": clip_low,
            "clip_high": clip_high,
            "dashboard_selectable": dashboard_selectable,
        }

    hyperparameter_values = render_hyperparameters(params, editable=editable)

    st.html("</div></div>")

    return {
        **identity_values,
        **training_values,
        **behavior_left_values,
        **behavior_right_values,
        **hyperparameter_values,
    }


def render_model_setup(
    registry: dict[str, Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
) -> None:
    """Render the Model Setup workspace.

    This phase delegates the base configuration editor sections to modular
    controls while preserving the existing legacy save/delete/advanced flow.
    """

    original_editor = legacy_model_lab._render_config_editor
    legacy_model_lab._render_config_editor = _render_config_editor
    try:
        legacy_model_lab._render_configuration(registry, rows, row_by_id)
    finally:
        legacy_model_lab._render_config_editor = original_editor
