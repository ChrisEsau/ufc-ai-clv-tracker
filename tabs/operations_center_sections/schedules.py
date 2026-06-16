from __future__ import annotations

import html

import streamlit as st

from utils.operations_status import schedule_rows


def _escape(value) -> str:
    return html.escape(str(value or ""))


def render_schedules() -> None:
    rows = []
    for name, schedule, next_run in schedule_rows():
        rows.append(
            "<tr>"
            f"<td>{_escape(name)}</td>"
            f"<td>{_escape(schedule)}</td>"
            f"<td>{_escape(next_run)}</td>"
            "</tr>"
        )
    st.html(
        '<div class="ops-card"><div class="ops-card-title">Scheduled Operations</div>'
        '<table class="ops-table"><thead><tr><th>Job Name</th><th>Schedule</th><th>Next Run</th></tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table><div class="ops-link">Manage Schedules →</div></div>'
    )
