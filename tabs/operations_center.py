from __future__ import annotations

import streamlit as st

from tabs.operations_center_sections.header import render_header
from tabs.operations_center_sections.model_status import render_model_status
from tabs.operations_center_sections.operation_cards import render_operation_cards
from tabs.operations_center_sections.pipeline_status import render_pipeline_status
from tabs.operations_center_sections.recent_jobs import render_recent_jobs
from tabs.operations_center_sections.schedules import render_schedules
from tabs.operations_center_sections.status_summary import render_status_summary
from tabs.operations_center_sections.styles import inject_operations_css
from tabs.operations_center_sections.system_status import render_system_status


def render_operations_center() -> None:
    inject_operations_css()
    render_header()
    render_status_summary()
    render_pipeline_status()
    render_model_status()
    render_operation_cards()

    left, middle, right = st.columns([1, 1.05, 1.1], gap="medium")
    with left:
        render_system_status()
    with middle:
        render_recent_jobs()
    with right:
        render_schedules()
