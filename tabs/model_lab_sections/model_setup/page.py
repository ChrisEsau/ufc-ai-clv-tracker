from __future__ import annotations

import streamlit as st

from tabs.model_lab_sections.model_setup.actions import render_action_bar
from tabs.model_lab_sections.model_setup.behavior import render_behavior_section
from tabs.model_lab_sections.model_setup.feature_selection import render_feature_selection_section
from tabs.model_lab_sections.model_setup.hyperparameters import render_hyperparameters_section
from tabs.model_lab_sections.model_setup.identity import render_identity_section
from tabs.model_lab_sections.model_setup.selectors import render_model_selector
from tabs.model_lab_sections.model_setup.styles import inject_styles
from tabs.model_lab_sections.model_setup.training import render_training_section
from tabs.model_lab_sections.model_setup.validation import render_validation_summary
from utils.model_lab_setup import model_context, registry_io


MODEL_SETUP_WIDGET_KEYS = [
    "model_setup_identity_model_id",
    "model_setup_identity_family",
    "model_setup_identity_algorithm",
    "model_setup_identity_status",
    "model_setup_identity_market",
    "model_setup_identity_dashboard_selectable",
    "model_setup_identity_display_name",
    "model_setup_identity_description",
    "model_setup_training_train_start_date",
    "model_setup_training_train_end_date",
    "model_setup_training_calibration_end_date",
    "model_setup_behavior_calibration_enabled",
    "model_setup_behavior_calibration_method",
    "model_setup_behavior_clip_low",
    "model_setup_behavior_clip_high",
    "model_setup_behavior_threshold_source",
    "model_setup_behavior_threshold_value",
    "model_setup_hyperparameters_n_estimators",
    "model_setup_hyperparameters_max_depth",
    "model_setup_hyperparameters_learning_rate",
    "model_setup_hyperparameters_subsample",
    "model_setup_hyperparameters_colsample_bytree",
    "model_setup_hyperparameters_random_state",
    "model_setup_hyperparameters_eval_metric",
    "model_setup_features_selected_bundles",
    "model_setup_features_include_features",
    "model_setup_features_exclude_features",
    "model_setup_features_resolved_preview",
    "model_setup_features_expected_count",
    "model_setup_features_resolved_count",
]


def _clear_form_widget_state_if_context_changed(context: dict) -> None:
    context_id = str(context.get("model_id") or "")
    previous_context_id = st.session_state.get("model_setup_last_context_id")
    if previous_context_id == context_id:
        return
    for key in MODEL_SETUP_WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state["model_setup_last_context_id"] = context_id


def _resolve_page_context(registry: dict, selected_model_id: str) -> dict:
    draft_context = st.session_state.get("model_setup_draft_context")
    draft_template_id = st.session_state.get("model_setup_draft_template_model_id")
    if isinstance(draft_context, dict) and draft_template_id == selected_model_id:
        return draft_context
    if draft_context and draft_template_id != selected_model_id:
        st.session_state.pop("model_setup_draft_context", None)
        st.session_state.pop("model_setup_draft_template_model_id", None)
    return model_context.resolve_existing_model_context(registry, selected_model_id)


def _render_context_banner(context: dict) -> None:
    summary = model_context.summarize_context_for_ui(context)
    status = summary["status"].upper()
    st.markdown(
        f"""
        <div class="model-setup-shell">
            <div class="model-setup-banner-left">
                <div class="model-setup-title">{summary['model_id']} <span class="model-setup-status">{status}</span></div>
                <div class="model-setup-subtitle">
                    Family: {summary['family']} &nbsp; · &nbsp;
                    Market: {summary['market']} &nbsp; · &nbsp;
                    {summary['editable_label']}
                </div>
                <div class="model-setup-note">
                    Config: {summary['config_path']}<br/>
                    Artifacts: {summary['artifact_dir']}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page() -> None:
    """Render the new clean Model Setup workspace."""

    inject_styles()

    try:
        registry = registry_io.load_model_registry()
        rows = registry_io.get_registered_model_rows(registry)
        selected_model_id = render_model_selector(rows)
        if not selected_model_id:
            return
        context = _resolve_page_context(registry, selected_model_id)
        _clear_form_widget_state_if_context_changed(context)
        _render_context_banner(context)

        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns([1.25, 0.92, 1.05, 1.25], gap="medium")
        with row1_col1:
            with st.container(border=True):
                identity_payload = render_identity_section(context)
        with row1_col2:
            with st.container(border=True):
                training_payload = render_training_section(context)
        with row1_col3:
            with st.container(border=True):
                behavior_payload = render_behavior_section(context)
        with row1_col4:
            with st.container(border=True):
                hyperparameters_payload = render_hyperparameters_section(context)

        with st.container(border=True):
            feature_payload = render_feature_selection_section(context)

        payload = {
            "identity": identity_payload,
            "training": training_payload,
            "behavior": behavior_payload,
            "hyperparameters": hyperparameters_payload,
            "features": feature_payload,
        }

        st.session_state["model_setup_identity_payload"] = identity_payload
        st.session_state["model_setup_training_payload"] = training_payload
        st.session_state["model_setup_behavior_payload"] = behavior_payload
        st.session_state["model_setup_hyperparameters_payload"] = hyperparameters_payload
        st.session_state["model_setup_feature_payload"] = feature_payload

        st.markdown('<div class="model-setup-footer-spacer"></div>', unsafe_allow_html=True)
        footer_col1, footer_col2 = st.columns([1.35, 1.0], gap="medium")
        with footer_col1:
            with st.container(border=True):
                validation_result = render_validation_summary(context, registry, payload)
        with footer_col2:
            with st.container(border=True):
                render_action_bar(context, registry, payload, validation_result)
        st.session_state["model_setup_validation_result"] = validation_result
    except Exception as exc:
        st.error(f"Unable to load Model Setup: {exc}")
