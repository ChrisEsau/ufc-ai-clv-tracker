from __future__ import annotations

import streamlit as st

from tabs.model_lab_sections.model_setup.behavior import render_behavior_section
from tabs.model_lab_sections.model_setup.feature_selection import render_feature_selection_section
from tabs.model_lab_sections.model_setup.hyperparameters import render_hyperparameters_section
from tabs.model_lab_sections.model_setup.identity import render_identity_section
from tabs.model_lab_sections.model_setup.selectors import render_model_selector
from tabs.model_lab_sections.model_setup.styles import inject_styles
from tabs.model_lab_sections.model_setup.training import render_training_section
from utils.model_lab_setup import model_context, registry_io


def _render_context_banner(context: dict) -> None:
    summary = model_context.summarize_context_for_ui(context)
    status = summary["status"].upper()
    st.markdown(
        f"""
        <div class="model-setup-shell">
            <div class="model-setup-title">{summary['model_id']} <span class="model-setup-status">{status}</span></div>
            <div class="model-setup-subtitle">
                Family: {summary['family']} &nbsp; | &nbsp;
                Market: {summary['market']} &nbsp; | &nbsp;
                {summary['editable_label']}
            </div>
            <div class="model-setup-note">
                Config: {summary['config_path']}<br/>
                Artifacts: {summary['artifact_dir']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page() -> None:
    """Render the new clean Model Setup workspace."""

    inject_styles()
    st.markdown("## Model Setup")
    st.caption("Configure model identity, training setup, behavior, hyperparameters, and feature selection.")

    try:
        registry = registry_io.load_model_registry()
        rows = registry_io.get_registered_model_rows(registry)
        selected_model_id = render_model_selector(rows)
        if not selected_model_id:
            return
        context = model_context.resolve_existing_model_context(registry, selected_model_id)
        _render_context_banner(context)

        row1_col1, row1_col2, row1_col3 = st.columns([1.05, 1.0, 1.0], gap="medium")
        with row1_col1:
            with st.container(border=True):
                identity_payload = render_identity_section(context)
        with row1_col2:
            with st.container(border=True):
                training_payload = render_training_section(context)
        with row1_col3:
            with st.container(border=True):
                behavior_payload = render_behavior_section(context)

        row2_col1, row2_col2 = st.columns([1.0, 1.5], gap="medium")
        with row2_col1:
            with st.container(border=True):
                hyperparameters_payload = render_hyperparameters_section(context)
        with row2_col2:
            with st.container(border=True):
                feature_payload = render_feature_selection_section(context)

        st.session_state["model_setup_identity_payload"] = identity_payload
        st.session_state["model_setup_training_payload"] = training_payload
        st.session_state["model_setup_behavior_payload"] = behavior_payload
        st.session_state["model_setup_hyperparameters_payload"] = hyperparameters_payload
        st.session_state["model_setup_feature_payload"] = feature_payload
        st.info("Next: add Validation and Actions.")
    except Exception as exc:
        st.error(f"Unable to load Model Setup: {exc}")
