import streamlit as st

from utils.dm_audit_history import render_audit_history
from utils.dm_dataset_health import render_dataset_health
from utils.dm_event_discovery import render_event_discovery
from utils.dm_final_review import render_final_review


def render_data_maintenance():
    st.title("Data Maintenance")

    st.caption("UFC ingestion control tower")

    render_dataset_health()
    render_event_discovery()
    render_final_review()
    render_audit_history()
