import streamlit as st

from utils.dm_pipeline_status import render_pipeline_status
from utils.dm_validation_gate import render_validation_gate
from utils.dm_workflow_controls import render_workflow_controls

from utils.dm_append_status import (
    render_append_status,
)

from utils.dm_dataset_health import (
    render_dataset_health,
)

def render_data_maintenance():

    st.title("Data Maintenance")

    st.caption(
        "UFC ingestion control tower"
    )

    render_append_status()

    st.markdown("---")

    render_pipeline_status()

    st.markdown("---")

    render_workflow_controls()

    st.markdown("---")

    render_validation_gate()

    render_dataset_health()