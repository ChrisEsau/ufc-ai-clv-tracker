from __future__ import annotations

import html

import streamlit as st

from utils.operations_runbook_registry import get_runbook
from utils.operations_runbook_state import load_state
from utils.operations_workflow_launcher import launch_next_workflow


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _status_card(label: str, value: str, caption: str, icon: str, tone: str = "success") -> str:
    return (
        '<div class="ops-card ops-auto-card">'
        f'<div class="ops-auto-icon {tone}">{_escape(icon)}</div>'
        '<div>'
        f'<div class="ops-auto-label">{_escape(label)}</div>'
        f'<div class="ops-auto-value {tone}">{_escape(value)}</div>'
        f'<div class="ops-auto-caption">{_escape(caption)}</div>'
        '</div></div>'
    )


def _tone_for_status(status: str) -> str:
    normalized = str(status or "idle").lower()
    if normalized == "running":
        return "info"
    if normalized == "failed":
        return "warning"
    if normalized == "completed":
        return "success"
    return "purple"


def _step_state(step_index: int, state: dict) -> tuple[str, str, str]:
    status = str(state.get("status") or "idle").lower()
    current_index = state.get("current_step_index")

    if status == "failed" and current_index == step_index:
        return "Failed", "Failed", "failed"
    if status == "running" and current_index == step_index:
        return "Running", "In Progress", "progress"
    if status == "completed":
        return "Complete", "Complete", "complete"
    if isinstance(current_index, int) and step_index < current_index:
        return "Complete", "Complete", "complete"
    return "—", "Waiting", "waiting"


def render_autopilot_summary() -> None:
    state = load_state()
    runbook = get_runbook(str(state.get("runbook_id") or "market_refresh_v2"))
    status = str(state.get("status") or "idle").title()
    tone = _tone_for_status(status)
    current_step = str(state.get("current_step_id") or "No active step")
    if current_step != "No active step":
        step_lookup = {step["step_id"]: step["display_name"] for step in runbook.get("steps", [])}
        current_step = step_lookup.get(current_step, current_step)

    cards = [
        ("Autopilot Status", status, "Operations state", "BOT", tone),
        ("Current Runbook", runbook.get("display_name", "Market Refresh"), current_step, "CAL", "info"),
        ("Next Scheduled Run", "Not scheduled", "Manual runbook mode", "CLK", "purple"),
        ("Last Completed Run", state.get("completed_at") or "—", "Runbook completion", "OK", "success"),
        ("Alerts Requiring Review", "—", "Alert engine pending", "ALR", "warning"),
    ]
    st.html('<div class="ops-auto-grid">' + ''.join(_status_card(*card) for card in cards) + '</div>')


def render_runbook_progress() -> None:
    state = load_state()
    runbook = get_runbook(str(state.get("runbook_id") or "market_refresh_v2"))

    action_left, action_right = st.columns([1, 2])
    with action_left:
        if st.button("Run Next Workflow", key="ops_run_next_workflow", type="primary", use_container_width=True):
            ok, message, _state = launch_next_workflow(str(runbook.get("runbook_id") or "market_refresh_v2"))
            if ok:
                st.success(message)
            else:
                st.error(message)
            st.rerun()
    with action_right:
        st.caption("Manual validation mode: launches one mapped GitHub workflow at a time.")

    rows = []
    for idx, step in enumerate(runbook.get("steps", [])):
        stamp, status_label, tone = _step_state(idx, state)
        workflow_count = len(step.get("workflows", []))
        workflow_label = f"{workflow_count} workflow" if workflow_count == 1 else f"{workflow_count} workflows"
        desc = f"{step.get('description', '')} ({workflow_label})"
        rows.append(
            f'<div class="ops-runbook-row {tone}">'
            f'<div class="ops-runbook-num {tone}">{idx + 1}</div>'
            '<div class="ops-runbook-main">'
            f'<div class="ops-runbook-title">{_escape(step.get("display_name"))}</div>'
            f'<div class="ops-runbook-desc">{_escape(desc)}</div>'
            '</div>'
            f'<div class="ops-runbook-time">{_escape(stamp)}</div>'
            f'<div class="ops-runbook-state {tone}">{_escape(status_label)}</div>'
            '</div>'
        )

    note = "Runbook registry loaded. Use Run Next Workflow to launch one mapped workflow."
    if state.get("current_workflow_file"):
        note = f"Current workflow: {state.get('current_workflow_file')}"
    if state.get("error"):
        note = f"Error: {state.get('error')}"

    st.html(
        '<div class="ops-card ops-panel">'
        f'<div class="ops-panel-header"><div><div class="ops-panel-title">Runbook Progress</div><div class="ops-panel-subtitle">{_escape(runbook.get("display_name", "Market Refresh"))}</div></div>'
        '<div class="ops-legend"><span><i class="complete"></i>Complete</span><span><i class="progress"></i>In Progress</span><span><i class="waiting"></i>Waiting</span><span><i class="failed"></i>Failed</span></div></div>'
        '<div class="ops-runbook-list">' + ''.join(rows) + '</div>'
        f'<div class="ops-panel-note">{_escape(note)}</div>'
        '</div>'
    )


def render_upcoming_runs() -> None:
    runs = [
        ("Market Refresh", "Manual mode", "Runbook registry ready; scheduler not wired", "Ready"),
        ("Monday Reset", "Future", "Weekly settlement and ingestion runbook", "Planned"),
        ("Fight Day Monitor", "Future", "Increased refresh cadence and final snapshots", "Planned"),
    ]
    rows = []
    for name, when, desc, badge in runs:
        badge_class = "purple" if badge == "Planned" else "info"
        rows.append(
            '<div class="ops-upcoming-row">'
            '<div class="ops-upcoming-icon">▣</div>'
            '<div class="ops-upcoming-main">'
            f'<div class="ops-upcoming-title">{_escape(name)}</div>'
            f'<div class="ops-upcoming-desc">{_escape(desc)}</div>'
            '</div>'
            f'<div class="ops-upcoming-time">{_escape(when)}</div>'
            f'<div class="ops-mini-badge {badge_class}">{_escape(badge)}</div>'
            '</div>'
        )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">Upcoming Runs</div><div class="ops-link-inline">Schedule pending</div></div>'
        + ''.join(rows) + '</div>'
    )


def render_review_alerts() -> None:
    rows = [
        ("INFO", "Alert engine pending", "Betting outcomes will feed this panel later", "Pending", "Not wired", "—"),
    ]
    html_rows = []
    for level, title, detail, priority, edge, stamp in rows:
        html_rows.append(
            '<div class="ops-alert-row">'
            f'<div class="ops-alert-badge">{_escape(level)}</div>'
            '<div class="ops-alert-main">'
            f'<div class="ops-alert-title">{_escape(title)}</div>'
            f'<div class="ops-alert-detail">{_escape(detail)}</div>'
            '</div>'
            f'<div class="ops-alert-edge"><div>{_escape(priority)}</div><span>{_escape(edge)}</span></div>'
            f'<div class="ops-alert-time">{_escape(stamp)}</div>'
            '</div>'
        )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">Review Alerts</div><div class="ops-link-inline">View All Alerts</div></div>'
        + ''.join(html_rows) + '<div class="ops-alert-footer">No live alerts wired yet</div></div>'
    )


def render_system_health_compact() -> None:
    rows = ["Runbook Registry", "Runbook State", "GitHub Workflows", "Artifact Checks"]
    html_rows = ''.join(
        f'<div class="ops-health-row"><span>OK {_escape(row)}</span><span class="ops-green">Ready</span><span>—</span></div>'
        for row in rows
    )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">System Health</div><div class="ops-link-inline">View Details</div></div>'
        + html_rows + '<div class="ops-health-footer">Workflow execution pending</div></div>'
    )


def render_recent_activity_compact() -> None:
    state = load_state()
    rows = [
        ("STATE", f"Runbook state: {state.get('status', 'idle')}", state.get("updated_at") or "—"),
        ("REG", "Market Refresh registry loaded", "—"),
        ("NEXT", "Next: workflow launcher", "—"),
    ]
    html_rows = ''.join(
        '<div class="ops-activity-row">'
        f'<span>{_escape(icon)}</span><span>{_escape(text)}</span><span>{_escape(stamp)}</span>'
        '</div>'
        for icon, text, stamp in rows
    )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">Recent Activity</div><div class="ops-link-inline">View All Logs</div></div>'
        + html_rows + '</div>'
    )


def render_autopilot_footer() -> None:
    st.html(
        '<div class="ops-card ops-footer">'
        '<div>Operations Center is reading the Market Refresh registry. Workflow launch controls are in manual validation mode.</div>'
        '<button class="ops-settings-button">Autopilot Settings</button>'
        '</div>'
    )
