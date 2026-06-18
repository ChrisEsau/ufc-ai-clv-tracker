from __future__ import annotations

import streamlit as st

from tabs.model_lab_sections.model_setup.selectors import render_model_selector
from tabs.model_lab_sections.model_setup.styles import inject_styles
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
        st.info("Next: add Model Setup cards for Identity, Training, Behavior, Hyperparameters, and Feature Selection.")
    except Exception as exc:
        st.error(f"Unable to load Model Setup: {exc}")
