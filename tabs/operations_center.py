from __future__ import annotations

import streamlit as st

from tabs.operations_center_sections.autopilot import (
    render_autopilot_footer,
    render_autopilot_summary,
    render_recent_activity_compact,
    render_review_alerts,
    render_runbook_progress,
    render_system_health_compact,
    render_upcoming_runs,
)
from tabs.operations_center_sections.autopilot_styles import inject_autopilot_css
from tabs.operations_center_sections.header import render_header
from tabs.operations_center_sections.styles import inject_operations_css


def render_operations_center() -> None:
    inject_operations_css()
    inject_autopilot_css()
    render_header()
    render_autopilot_summary()

    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        render_runbook_progress()
    with right:
        render_upcoming_runs()

    alerts, health, activity = st.columns([1, 0.8, 1.15], gap="medium")
    with alerts:
        render_review_alerts()
    with health:
        render_system_health_compact()
    with activity:
        render_recent_activity_compact()

    render_autopilot_footer()
