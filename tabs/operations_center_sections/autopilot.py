from __future__ import annotations

import html
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st

from utils.github_actions import get_latest_workflow_run, trigger_workflow
from utils.operations_runbook_registry import get_runbook
from utils.operations_runbook_state import load_state
from utils.operations_status_writer import read_status


ORCHESTRATOR_WORKFLOW_FILE = "run-market-refresh-orchestrator.yml"
AUTO_REFRESH_SECONDS = 15
RUNNING_GITHUB_STATUSES = {"queued", "in_progress", "requested", "waiting", "pending"}
CENTRAL_TZ = ZoneInfo("America/Chicago")


def _fragment(run_every_seconds: int | None = None):
    fragment = getattr(st, "fragment", None)
    if fragment is None:
        def passthrough(func):
            return func
        return passthrough
    if run_every_seconds is None:
        return fragment
    return fragment(run_every=f"{run_every_seconds}s")


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


def _latest_orchestrator_run() -> dict | None:
    ok, _message, latest_run = get_latest_workflow_run(ORCHESTRATOR_WORKFLOW_FILE)
    if not ok:
        return None
    return latest_run


def _github_run_status(latest_run: dict | None) -> tuple[str | None, str | None, str | None]:
    if not latest_run:
        return None, None, None
    return (
        str(latest_run.get("status") or "").lower() or None,
        str(latest_run.get("conclusion") or "").lower() or None,
        latest_run.get("html_url"),
    )


def _is_github_running(latest_run: dict | None) -> bool:
    github_status, conclusion, _url = _github_run_status(latest_run)
    return bool(github_status in RUNNING_GITHUB_STATUSES or (github_status == "completed" and not conclusion))


def _display_status(status: dict, latest_run: dict | None) -> tuple[str, str]:
    github_status, conclusion, _url = _github_run_status(latest_run)
    local_status = str(status.get("status") or "idle").lower()

    if _is_github_running(latest_run):
        return "running", f"GitHub workflow {github_status}"
    if github_status == "completed" and conclusion == "success":
        return "completed", "GitHub workflow completed successfully"
    if github_status == "completed" and conclusion:
        return "failed", f"GitHub workflow {conclusion}"
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


def _auto_refresh_note(status: dict, latest_run: dict | None) -> None:
    display_status, _message = _display_status(status, latest_run)
    if display_status == "running":
        st.caption(f"Refreshing Operations Center data every {AUTO_REFRESH_SECONDS} seconds while Market Refresh is running.")


def _step_state(step_index: int, status: dict, latest_run: dict | None) -> tuple[str, str, str]:
    display_status, _message = _display_status(status, latest_run)
    current_index = status.get("step_index")

    if display_status == "running" and not isinstance(current_index, int):
        if step_index == 1:
            return "Running", "In Progress", "progress"
        return "—", "Waiting", "waiting"
    if not isinstance(current_index, int):
        return "—", "Waiting", "waiting"

    if display_status == "failed" and current_index == step_index:
        return "Failed", "Failed", "failed"
    if display_status == "running" and current_index == step_index:
        return "Running", "In Progress", "progress"
    if display_status == "completed" or step_index < current_index:
        return "Complete", "Complete", "complete"
    return "—", "Waiting", "waiting"


def _launch_market_refresh() -> None:
    inputs = {
        "mode": "test",
        "max_upcoming_events": "",
        "max_draftkings_events": "5",
        "model_id": "moneyline_xgboost_v5",
        "model_mode": "production",
    }
    ok, message = trigger_workflow(ORCHESTRATOR_WORKFLOW_FILE, inputs=inputs)
    if ok:
        st.success(message)
    else:
        st.error(message)
    st.rerun()


@_fragment(run_every_seconds=AUTO_REFRESH_SECONDS)
def render_autopilot_summary() -> None:
    status = read_status()
    latest_run = _latest_orchestrator_run()
    _auto_refresh_note(status, latest_run)
    runbook = get_runbook(str(status.get("runbook_id") or "market_refresh_v2"))
    display_status, display_message = _display_status(status, latest_run)
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


@_fragment(run_every_seconds=AUTO_REFRESH_SECONDS)
def render_runbook_progress() -> None:
    legacy_state = load_state()
    status = read_status()
    latest_run = _latest_orchestrator_run()
    runbook = get_runbook(str(status.get("runbook_id") or legacy_state.get("runbook_id") or "market_refresh_v2"))

    action_left, action_right = st.columns([1, 2])
    with action_left:
        if st.button("Run Market Refresh", key="ops_run_market_refresh", type="primary", use_container_width=True):
            _launch_market_refresh()
    with action_right:
        github_status, conclusion, url = _github_run_status(latest_run)
        status_text = f"GitHub: {github_status or 'not found'}"
        if conclusion:
            status_text += f" / {conclusion}"
        if url:
            st.caption(f"{status_text} — [Open in GitHub]({url})")
        else:
            st.caption("Launches the GitHub Market Refresh orchestrator in test mode. DraftKings discovery is limited to 5 matched events.")

    rows = []
    for idx, step in enumerate(runbook.get("steps", []), start=1):
        stamp, status_label, tone = _step_state(idx, status, latest_run)
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

    _status_value, message = _display_status(status, latest_run)
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


@_fragment(run_every_seconds=AUTO_REFRESH_SECONDS)
def render_recent_activity_compact() -> None:
    status = read_status()
    latest_run = _latest_orchestrator_run()
    github_status, conclusion, _url = _github_run_status(latest_run)
    rows = [
        ("STATE", f"Market Refresh: {_display_status(status, latest_run)[0]}", _format_cst(status.get("updated_at"))),
        ("STEP", f"Step: {status.get('current_step_name') or 'none'}", status.get("current_substep_name") or "—"),
        ("GHA", f"GitHub: {github_status or 'not found'}", conclusion or "—"),
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
        '<div>Operations Center launches the Market Refresh orchestrator, refreshes Operations data without a full browser reload, and shows times in CST.</div>'
        '<button class="ops-settings-button">Autopilot Settings</button>'
        '</div>'
    )
