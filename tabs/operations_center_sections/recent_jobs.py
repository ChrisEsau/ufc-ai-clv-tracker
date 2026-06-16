from __future__ import annotations

import html

import streamlit as st

from utils.operations_status import recent_job_rows


def _escape(value) -> str:
    return html.escape(str(value or ""))


def render_recent_jobs() -> None:
    dispatches = st.session_state.get("ops_recent_dispatches", [])
    rows = []
    for dispatch in dispatches[:5]:
        rows.append(
            "<tr>"
            f"<td>{_escape(dispatch.get('label'))}</td>"
            f"<td>Workflow</td>"
            f"<td class='ops-blue'>Dispatched</td>"
            f"<td>{_escape(dispatch.get('workflow'))}</td>"
            "</tr>"
        )
    if not rows:
        for name, job_type, status, detail in recent_job_rows():
            rows.append(
                "<tr>"
                f"<td>{_escape(name)}</td>"
                f"<td>{_escape(job_type)}</td>"
                f"<td class='ops-yellow'>{_escape(status)}</td>"
                f"<td>{_escape(detail)}</td>"
                "</tr>"
            )
    st.html(
        '<div class="ops-card"><div class="ops-card-title">Recent Jobs</div>'
        '<table class="ops-table"><thead><tr><th>Job Name</th><th>Type</th><th>Status</th><th>Detail</th></tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table><div class="ops-link">View All Job History →</div></div>'
    )
