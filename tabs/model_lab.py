import streamlit as st

from utils.model_lab_workflows import render_model_workflow_launcher
from utils.ui.sections import page_header


def render_model_lab():
    page_header(
        "Model Lab",
        "Simple V2 model workflow launcher driven by the model registry.",
        kicker="Research Workspace",
    )

    st.info(
        "Select a registered model below, then launch the feature-view, training, "
        "or prediction workflow. Workflow inputs are resolved from the registry "
        "and model config."
    )

    render_model_workflow_launcher()
