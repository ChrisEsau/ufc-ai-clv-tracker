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
        "caption": "Launches the GitHub Market Refresh orchestrator in test mode.",
        "inputs": {
            "mode": "test",
            "max_upcoming_events": "",
            "max_draftkings_events": "all",
            "model_id": "moneyline_xgboost_v5",
            "model_mode": "production",
            "snapshot_model_mode": "all",
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
