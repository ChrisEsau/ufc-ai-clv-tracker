import streamlit as st
from utils.dm_workflow_controls import render_workflow_controls

from utils.dm_validation_gate import (
    render_validation_gate,
)


def render_data_maintenance():

    st.title("Data Maintenance")

    st.markdown(
        """
        UFC ingestion control tower.

        Pipeline flow:

        SCRAPE → MAP → ENRICH → VALIDATE → APPEND
        """
    )

    st.markdown("---")
    
    render_workflow_controls()
    
    st.markdown("---")


    render_validation_gate()