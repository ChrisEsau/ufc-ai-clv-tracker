import streamlit as st

from utils.dm_pipeline_status import render_pipeline_status
from utils.dm_validation_gate import render_validation_gate
from utils.dm_workflow_controls import render_workflow_controls

from utils.dm_append_status import render_append_status
from utils.dm_dataset_health import render_dataset_health
from utils.dm_event_discovery import render_event_discovery
from utils.dm_fight_scrape import render_fight_scrape
from utils.dm_enrichment import render_enrichment
from utils.dm_final_review import render_final_review
from utils.dm_audit_history import (render_audit_history)

def render_data_maintenance():

    st.title("Data Maintenance")

    st.caption(
        "UFC ingestion control tower"
    )

    #render_pipeline_status()

    #st.markdown("---")

    render_dataset_health()
    render_event_discovery()
    render_fight_scrape()
    render_enrichment()
    render_validation_gate()
    render_final_review()
    render_audit_history()
    render_append_status()
