from __future__ import annotations

import streamlit as st

from tabs.model_lab_sections.model_setup.styles import inject_styles


def render_page() -> None:
    """Render the new clean Model Setup workspace shell.

    This page is intentionally built beside the legacy Configuration workspace.
    It will receive clean backend + UI implementation in later phases.
    """

    inject_styles()
    st.markdown("## Model Setup")
    st.markdown(
        """
        <div class="model-setup-shell">
            <div class="model-setup-title">Clean Model Setup Workspace</div>
            <div class="model-setup-subtitle">New implementation in progress.</div>
            <div class="model-setup-note">
                Legacy Configuration remains available until this page reaches feature parity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
