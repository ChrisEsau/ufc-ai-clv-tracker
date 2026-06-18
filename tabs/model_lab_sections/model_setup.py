from __future__ import annotations

from copy import deepcopy
from typing import Any

import streamlit as st

from tabs import model_lab as legacy_model_lab
from tabs.model_lab_sections.model_setup_identity import render_identity
from tabs.model_lab_sections.model_setup_training import render_training
from tabs.model_lab_sections.model_setup_behavior import (
    render_calibration_controls,
    render_probability_controls,
)
from tabs.model_lab_sections.model_setup_hyperparameters import render_hyperparameters
from tabs.model_lab_sections import model_setup_feature_selection  # noqa: F401
from tabs.model_lab_sections import model_setup_advanced  # noqa: F401
from tabs.model_lab_sections import model_setup_save_actions as save_actions
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
        behavior_left_values = render_calibration_controls(calibration, editable=editable)
    with c2:
        behavior_right_values = render_probability_controls(probability, context, editable=editable)

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

    This phase delegates base editor and save/delete actions to modular controls
    while preserving the existing legacy advanced/configuration flow.
    """

    original_editor = legacy_model_lab._render_config_editor
    original_apply_advanced = legacy_model_lab._apply_advanced_config_updates
    original_save = legacy_model_lab._save_new_or_existing_model
    original_github_delete = legacy_model_lab._github_delete_file
    original_delete = legacy_model_lab._delete_model
    original_delete_dialog = legacy_model_lab._render_delete_dialog

    legacy_model_lab._render_config_editor = _render_config_editor
    legacy_model_lab._apply_advanced_config_updates = save_actions.apply_advanced_config_updates
    legacy_model_lab._save_new_or_existing_model = save_actions.save_new_or_existing_model
    legacy_model_lab._github_delete_file = save_actions.github_delete_file
    legacy_model_lab._delete_model = save_actions.delete_model
    legacy_model_lab._render_delete_dialog = save_actions.render_delete_dialog
    try:
        legacy_model_lab._render_configuration(registry, rows, row_by_id)
    finally:
        legacy_model_lab._render_config_editor = original_editor
        legacy_model_lab._apply_advanced_config_updates = original_apply_advanced
        legacy_model_lab._save_new_or_existing_model = original_save
        legacy_model_lab._github_delete_file = original_github_delete
        legacy_model_lab._delete_model = original_delete
        legacy_model_lab._render_delete_dialog = original_delete_dialog
