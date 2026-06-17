from __future__ import annotations

import html
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st

from utils.github_actions import trigger_workflow
from utils.operations_runbook_registry import get_runbook
from utils.operations_status_writer import read_status, start_runbook


ORCHESTRATOR_WORKFLOW_FILE = "run-market-refresh-orchestrator.yml"
CENTRAL_TZ = ZoneInfo("America/Chicago")


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _format_cst(value) -> str:
    if not value:
        return "—"
    if isinstance(value, str) and value.strip() in {"—", "-"}:
        return value
    try:
        if isinstance(value, str):
            cleaned = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
        elif isinstance(value, datetime):
            dt = value
        else:
            return str(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        central = dt.astimezone(CENTRAL_TZ)
        return central.strftime("%b %-d, %Y %-I:%M %p CST")
    except Exception:
        return str(value)


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


def _display_status(status: dict) -> tuple[str, str]:
    local_status = str(status.get("status") or "idle").lower()
    return local_status, status.get("message") or "Orchestrator idle"


def _tone_for_status(status: str) -> str:
    normalized = str(status or "idle").lower()
    if normalized == "running":
        return "info"
    if normalized == "failed":
        return "warning"
    if normalized == "completed":
        return "success"
    return "purple"


def _step_completion_times(status: dict) -> dict[str, str]:
    completion_times: dict[str, str] = {}
    for event in status.get("history") or []:
        if event.get("event") != "step_completed":
            continue
        step_id = event.get("step_id")
        timestamp = event.get("timestamp")
        if step_id and timestamp:
            completion_times[str(step_id)] = _format_cst(timestamp)
    return completion_times


def _step_state(step: dict, status: dict, completion_times: dict[str, str]) -> tuple[str, str, str]:
    display_status, _message = _display_status(status)
    step_id = str(step.get("step_id") or "")
    current_step_id = str(status.get("current_step_id") or "")

    if display_status == "completed":
        return completion_times.get(step_id, _format_cst(status.get("completed_at"))), "Complete", "complete"
    if display_status == "running":
        return "In Progress", "In Progress", "progress"
    if display_status == "failed" and step_id == current_step_id:
        return "Failed", "Failed", "failed"
    if step_id in completion_times:
        return completion_times[step_id], "Complete", "complete"
    return "—", "Waiting", "waiting"


def _launch_market_refresh() -> None:
    inputs = {
        "mode": "test",
        "max_upcoming_events": "",
        "max_draftkings_events": "5",
        "model_id": "moneyline_xgboost_v5",
        "model_mode": "production",
        "snapshot_model_mode": "all",
    }
    ok, message = trigger_workflow(ORCHESTRATOR_WORKFLOW_FILE, inputs=inputs)
    if ok:
        runbook = get_runbook("market_refresh_v2")
        start_runbook(
            runbook_id="market_refresh_v2",
            mode="test",
            step_total=len(runbook.get("steps", [])),
            message="Market Refresh launched from Operations Center",
        )
        st.success(message)
    else:
        st.error(message)
    st.rerun()


def render_autopilot_summary() -> None:
    status = read_status()
    runbook = get_runbook(str(status.get("runbook_id") or "market_refresh_v2"))
    display_status, display_message = _display_status(status)
    status_label = display_status.title()
    tone = _tone_for_status(display_status)
    current_step = status.get("current_step_name") or "No active step"
    current_substep = status.get("current_substep_name") or display_message

    cards = [
        ("Autopilot Status", status_label, current_substep, "BOT", tone),
        ("Current Runbook", runbook.get("display_name", "Market Refresh"), current_step, "CAL", "info"),
        ("Next Scheduled Run", "Not scheduled", "Manual test launch mode", "CLK", "purple"),
        ("Last Completed Run", _format_cst(status.get("completed_at")), "Market Refresh completion", "OK", "success"),
        ("Alerts Requiring Review", "—", "Alert engine pending", "ALR", "warning"),
    ]
    st.html('<div class="ops-auto-grid">' + ''.join(_status_card(*card) for card in cards) + '</div>')


def render_runbook_progress() -> None:
    status = read_status()
    runbook = get_runbook(str(status.get("runbook_id") or "market_refresh_v2"))
    completion_times = _step_completion_times(status)

    action_left, action_right = st.columns([1, 2])
    with action_left:
        if st.button("Run Market Refresh", key="ops_run_market_refresh", type="primary", use_container_width=True):
            _launch_market_refresh()
    with action_right:
        st.caption("Launches the GitHub Market Refresh orchestrator in test mode. Runbook cards are driven only by the local status artifact.")

    rows = []
    for idx, step in enumerate(runbook.get("steps", []), start=1):
        stamp, status_label, tone = _step_state(step, status, completion_times)
        workflow_count = len(step.get("workflows", []))
        workflow_label = f"{workflow_count} workflow" if workflow_count == 1 else f"{workflow_count} workflows"
        desc = f"{step.get('description', '')} ({workflow_label})"
        rows.append(
            f'<div class="ops-runbook-row {tone}">'
            f'<div class="ops-runbook-num {tone}">{idx}</div>'
            '<div class="ops-runbook-main">'
            f'<div class="ops-runbook-title">{_escape(step.get("display_name"))}</div>'
            f'<div class="ops-runbook-desc">{_escape(desc)}</div>'
            '</div>'
            f'<div class="ops-runbook-time">{_escape(stamp)}</div>'
            f'<div class="ops-runbook-state {tone}">{_escape(status_label)}</div>'
            '</div>'
        )

    _status_value, message = _display_status(status)
    note = status.get("message") or message
    if status.get("current_substep_name"):
        note = f"Current substep: {status.get('current_substep_name')}"
    if status.get("error"):
        note = f"Error: {status.get('error')}"

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
        ("Market Refresh", "Manual test mode", "Runs the GitHub orchestrator workflow", "Ready"),
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
    rows = [("INFO", "Alert engine pending", "Betting outcomes will feed this panel later", "Pending", "Not wired", "—")]
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
    rows = ["Runbook Registry", "Orchestrator Workflow", "Status File", "Artifact Checks"]
    html_rows = ''.join(
        f'<div class="ops-health-row"><span>OK {_escape(row)}</span><span class="ops-green">Ready</span><span>—</span></div>'
        for row in rows
    )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">System Health</div><div class="ops-link-inline">View Details</div></div>'
        + html_rows + '<div class="ops-health-footer">Orchestrator launch mode active</div></div>'
    )


def render_recent_activity_compact() -> None:
    status = read_status()
    rows = [
        ("STATE", f"Market Refresh: {_display_status(status)[0]}", _format_cst(status.get("updated_at"))),
        ("STEP", f"Step: {status.get('current_step_name') or 'none'}", status.get("current_substep_name") or "—"),
        ("STATUS", status.get("message") or "Orchestrator idle", _format_cst(status.get("completed_at"))),
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
        '<div>Operations Center launches the Market Refresh orchestrator and shows runbook status from the local status artifact in CST.</div>'
        '<button class="ops-settings-button">Autopilot Settings</button>'
        '</div>'
    )
