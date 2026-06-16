from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.common.paths import STATUS_DIR, ensure_data_dirs
from utils.operations_runbook_registry import DEFAULT_RUNBOOK_ID


RUNBOOK_STATE_PATH = STATUS_DIR / "operations_runbook_state.json"

IDLE = "idle"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
PAUSED = "paused"

ACTIVE_STATUSES = {RUNNING, PAUSED}
TERMINAL_STATUSES = {IDLE, COMPLETED, FAILED}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state(runbook_id: str = DEFAULT_RUNBOOK_ID) -> dict[str, Any]:
    return {
        "runbook_id": runbook_id,
        "status": IDLE,
        "current_step_id": None,
        "current_step_index": None,
        "current_workflow_file": None,
        "current_workflow_index": None,
        "current_workflow_run_id": None,
        "started_at": None,
        "completed_at": None,
        "updated_at": utc_now_iso(),
        "error": None,
        "history": [],
    }


def load_state(path: Path = RUNBOOK_STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return default_state()
    try:
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file) or {}
    except Exception:
        return default_state()
    merged = default_state(str(state.get("runbook_id") or DEFAULT_RUNBOOK_ID))
    merged.update(state)
    if not isinstance(merged.get("history"), list):
        merged["history"] = []
    return merged


def save_state(state: dict[str, Any], path: Path = RUNBOOK_STATE_PATH) -> dict[str, Any]:
    ensure_data_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = utc_now_iso()
    with path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True)
        file.write("\n")
    return state


def reset_state(runbook_id: str = DEFAULT_RUNBOOK_ID, path: Path = RUNBOOK_STATE_PATH) -> dict[str, Any]:
    return save_state(default_state(runbook_id), path=path)


def set_running(
    *,
    runbook_id: str,
    step_id: str,
    step_index: int,
    workflow_file: str,
    workflow_index: int,
    workflow_run_id: str | int | None = None,
    path: Path = RUNBOOK_STATE_PATH,
) -> dict[str, Any]:
    state = load_state(path)
    if state.get("status") not in ACTIVE_STATUSES:
        state["started_at"] = utc_now_iso()
        state["history"] = []
    state.update(
        {
            "runbook_id": runbook_id,
            "status": RUNNING,
            "current_step_id": step_id,
            "current_step_index": int(step_index),
            "current_workflow_file": workflow_file,
            "current_workflow_index": int(workflow_index),
            "current_workflow_run_id": None if workflow_run_id is None else str(workflow_run_id),
            "completed_at": None,
            "error": None,
        }
    )
    return save_state(state, path=path)


def set_completed(path: Path = RUNBOOK_STATE_PATH) -> dict[str, Any]:
    state = load_state(path)
    state.update(
        {
            "status": COMPLETED,
            "current_step_id": None,
            "current_step_index": None,
            "current_workflow_file": None,
            "current_workflow_index": None,
            "current_workflow_run_id": None,
            "completed_at": utc_now_iso(),
            "error": None,
        }
    )
    return save_state(state, path=path)


def set_failed(*, error: str, path: Path = RUNBOOK_STATE_PATH) -> dict[str, Any]:
    state = load_state(path)
    state["status"] = FAILED
    state["completed_at"] = utc_now_iso()
    state["error"] = str(error)
    return save_state(state, path=path)


def is_active(state: dict[str, Any] | None = None) -> bool:
    state = state or load_state()
    return str(state.get("status") or "").lower() in ACTIVE_STATUSES
