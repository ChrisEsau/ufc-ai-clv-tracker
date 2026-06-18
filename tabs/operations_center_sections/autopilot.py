from __future__ import annotations

import html
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st

from utils.github_actions import trigger_workflow
from utils.operations_runbook_registry import get_runbook, list_runbooks
from utils.operations_status_writer import read_status, start_runbook


CENTRAL_TZ = ZoneInfo("America/Chicago")
DEFAULT_RUNBOOK_ID = "market_refresh_v2"
RUNBOOK_LAUNCH_CONFIG = {
    "market_refresh_v2": {
        "workflow_file": "run-market-refresh-orchestrator.yml",
        "button_label": "Run Market Refresh",
        "caption": "Launches Market Refresh using all production models from the registry.",
        "inputs": {
            "mode": "test",
            "max_draftkings_events": "all",
            "model_mode": "production",
            "snapshot_model_mode": "production",
        },
    },
    "monday_reset_v1": {
        "workflow_file": "run-monday-reset-orchestrator.yml",
        "button_label": "Run Monday Reset",
        "caption": "Launches Monday Reset in production mode with max_events=all and auto_append=true.",
        "inputs": {
            "mode": "production",
            "max_events": "all",
            "auto_append": True,
            "skip_bankroll": False,
            "skip_clv": False,
        },
    },
    "fight_day_monitor_v1": {
        "workflow_file": "run-fight-day-monitor.yml",
        "button_label": "Run Fight Day Monitor",
        "caption": "Refreshes live DraftKings markets, recalculates betting outcomes, captures snapshots, and stores closing lines.",
        "inputs": {
            "mode": "production",
            "max_draftkings_events": "all",
            "model_mode": "production",
            "snapshot_model_mode": "production",
            "official_closing_snapshot": True,
        },
    },
}


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


def _available_runbooks() -> list[dict]:
    return list_runbooks()


def _read_runbook_status(runbook_id: str) -> dict:
    return read_status(runbook_id=runbook_id)


def _default_selected_runbook_id() -> str:
    available_ids = {str(runbook.get("runbook_id")) for runbook in _available_runbooks()}
    market_status = _read_runbook_status(DEFAULT_RUNBOOK_ID)
    status_runbook_id = str(market_status.get("runbook_id") or DEFAULT_RUNBOOK_ID)
    if status_runbook_id in available_ids:
        return status_runbook_id
    return DEFAULT_RUNBOOK_ID


def _selected_runbook_id() -> str:
    available_ids = [str(runbook.get("runbook_id")) for runbook in _available_runbooks()]
    if "ops_selected_runbook_id" not in st.session_state:
        st.session_state["ops_selected_runbook_id"] = _default_selected_runbook_id()
    if st.session_state["ops_selected_runbook_id"] not in available_ids:
        st.session_state["ops_selected_runbook_id"] = DEFAULT_RUNBOOK_ID
    return str(st.session_state["ops_selected_runbook_id"])


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
        if str(step.get("status") or "").lower() == "planned":
            return completion_times.get(step_id, _format_cst(status.get("completed_at"))), "Complete", "complete"
        return completion_times.get(step_id, _format_cst(status.get("completed_at"))), "Complete", "complete"
    if display_status == "running":
        return "In Progress", "In Progress", "progress"
    if display_status == "failed" and step_id == current_step_id:
        return "Failed", "Failed", "failed"
    if str(step.get("status") or "").lower() == "planned":
        return "Future", "Planned", "waiting"
    if step_id in completion_times:
        return completion_times[step_id], "Complete", "complete"
    return "—", "Waiting", "waiting"


def _launch_selected_runbook(runbook_id: str) -> None:
    config = RUNBOOK_LAUNCH_CONFIG.get(runbook_id)
    runbook = get_runbook(runbook_id)
    if not config:
        st.warning(f"{runbook.get('display_name', runbook_id)} is planned but not wired to a workflow yet.")
        return

    ok, message = trigger_workflow(config["workflow_file"], inputs=config["inputs"])
    if ok:
        start_runbook(
            runbook_id=runbook_id,
            mode=str(config["inputs"].get("mode") or "test"),
            step_total=len(runbook.get("steps", [])),
            message=f"{runbook.get('display_name', runbook_id)} launched from Operations Center",
        )
        st.success(message)
    else:
        st.error(message)
    st.rerun()


def render_autopilot_summary() -> None:
    selected_id = _selected_runbook_id()
    status = _read_runbook_status(selected_id)
    runbook = get_runbook(selected_id)
    display_status, display_message = _display_status(status)
    status_label = display_status.title()
    tone = _tone_for_status(display_status)
    current_step = status.get("current_step_name") or runbook.get("display_name", "No active step")
    current_substep = status.get("current_substep_name") or display_message

    cards = [
        ("Autopilot Status", status_label, current_substep, "BOT", tone),
        ("Current Runbook", runbook.get("display_name", "Market Refresh"), current_step, "CAL", "info"),
        ("Next Scheduled Run", "Not scheduled", "Manual launch mode", "CLK", "purple"),
        ("Last Completed Run", _format_cst(status.get("completed_at")), f"{runbook.get('display_name', 'Runbook')} completion", "OK", "success"),
        ("Alerts Requiring Review", "—", "Alert engine pending", "ALR", "warning"),
    ]
    st.html('<div class="ops-auto-grid">' + ''.join(_status_card(*card) for card in cards) + '</div>')


def render_runbook_progress() -> None:
    selected_id = _selected_runbook_id()
    runbooks = _available_runbooks()
    runbook_by_id = {str(runbook.get("runbook_id")): runbook for runbook in runbooks}
    runbook_ids = list(runbook_by_id.keys())

    selector_col, action_col, caption_col = st.columns([1.15, 1, 2])
    with selector_col:
        selected_id = st.selectbox(
            "Current Runbook",
            options=runbook_ids,
            index=runbook_ids.index(selected_id) if selected_id in runbook_ids else 0,
            format_func=lambda rid: runbook_by_id[rid].get("display_name", rid),
            key="ops_selected_runbook_id",
        )
        runbook = runbook_by_id[selected_id]
        status = _read_runbook_status(selected_id)
    with action_col:
        button_label = RUNBOOK_LAUNCH_CONFIG.get(selected_id, {}).get("button_label", f"Run {runbook.get('display_name', 'Runbook')}")
        if st.button(button_label, key=f"ops_run_{selected_id}", type="primary", use_container_width=True):
            _launch_selected_runbook(selected_id)
    with caption_col:
        caption = RUNBOOK_LAUNCH_CONFIG.get(selected_id, {}).get("caption", "This runbook is planned and not wired to an orchestrator workflow yet.")
        st.caption(f"{caption} Runbook rows are driven only by the local status artifact.")

    completion_times = _step_completion_times(status)
    rows = []
    for idx, step in enumerate(runbook.get("steps", []), start=1):
        stamp, status_label, tone = _step_state(step, status, completion_times)
        workflow_count = len(step.get("workflows", []))
        if str(step.get("status") or "").lower() == "planned":
            workflow_label = "planned"
        else:
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


def _manual_mode_label(runbook_id: str, ready: bool) -> str:
    if not ready:
        return "Future"
    if runbook_id in {"monday_reset_v1", "fight_day_monitor_v1"}:
        return "Manual production mode"
    return "Manual test mode"


def render_upcoming_runs() -> None:
    selected_id = _selected_runbook_id()
    runbooks = _available_runbooks()
    rows = []
    for runbook in runbooks:
        runbook_id = str(runbook.get("runbook_id"))
        ready = runbook_id in RUNBOOK_LAUNCH_CONFIG
        badge = "Selected" if runbook_id == selected_id else ("Ready" if ready else "Planned")
        badge_class = "info" if ready else "purple"
        when = _manual_mode_label(runbook_id, ready)
        rows.append(
            '<div class="ops-upcoming-row">'
            '<div class="ops-upcoming-icon">▣</div>'
            '<div class="ops-upcoming-main">'
            f'<div class="ops-upcoming-title">{_escape(runbook.get("display_name"))}</div>'
            f'<div class="ops-upcoming-desc">{_escape(runbook.get("description"))}</div>'
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
    rows = ["Runbook Registry", "Orchestrator Workflows", "Status Files", "Artifact Checks"]
    html_rows = ''.join(
        f'<div class="ops-health-row"><span>OK {_escape(row)}</span><span class="ops-green">Ready</span><span>—</span></div>'
        for row in rows
    )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">System Health</div><div class="ops-link-inline">View Details</div></div>'
        + html_rows + '<div class="ops-health-footer">Generic runbook launch mode active</div></div>'
    )


def render_recent_activity_compact() -> None:
    selected_id = _selected_runbook_id()
    status = _read_runbook_status(selected_id)
    runbook = get_runbook(selected_id)
    rows = [
        ("STATE", f"{runbook.get('display_name')}: {_display_status(status)[0]}", _format_cst(status.get("updated_at"))),
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
        '<div>Operations Center launches registered orchestrator runbooks and shows selected runbook status from each runbook status artifact in CST.</div>'
        '<button class="ops-settings-button">Autopilot Settings</button>'
        '</div>'
    )
