from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

from utils.github_actions import get_latest_workflow_run, trigger_workflow


def _parse_github_time(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _state_key(key: str, suffix: str) -> str:
    return f"workflow_status_{key}_{suffix}"


def workflow_status_label(workflow_file: str, key: str, *, idle_label: str = "Ready") -> str:
    """Return a compact queued/running/completed label for a dispatched workflow."""

    running_key = _state_key(key, "running")
    launched_key = _state_key(key, "launched_at")

    if not st.session_state.get(running_key):
        return idle_label

    ok, _, run = get_latest_workflow_run(workflow_file)
    if not ok or run is None:
        return "Queued..."

    launched_at = _parse_github_time(st.session_state.get(launched_key))
    created_at = _parse_github_time(run.get("created_at"))
    stale_run = bool(launched_at and created_at and created_at < launched_at - timedelta(seconds=30))
    if stale_run:
        return "Queued..."

    status = str(run.get("status") or "unknown").lower()
    if status == "completed":
        st.session_state[running_key] = False
        conclusion = str(run.get("conclusion") or "unknown").lower()
        if conclusion == "success":
            return "Completed"
        return f"{conclusion.title()}"

    if status in {"queued", "requested", "waiting"}:
        return "Queued..."
    if status in {"in_progress", "pending"}:
        return "Running..."
    return status.replace("_", " ").title()


def launch_workflow_with_status(workflow_file: str, key: str, inputs: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Dispatch a workflow and mark its status as running in Streamlit session state."""

    ok, message = trigger_workflow(workflow_file, inputs=inputs)
    if ok:
        st.session_state[_state_key(key, "running")] = True
        st.session_state[_state_key(key, "launched_at")] = datetime.now(timezone.utc).isoformat()
    return ok, message


def render_workflow_status(workflow_file: str, key: str, *, idle_label: str = "Ready") -> None:
    label = workflow_status_label(workflow_file, key, idle_label=idle_label)
    st.caption(f"Status: {label}")
