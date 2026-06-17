from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.common.paths import STATUS_DIR, ensure_data_dirs


MARKET_REFRESH_STATUS_PATH = STATUS_DIR / "market_refresh_status.json"
MONDAY_RESET_STATUS_PATH = STATUS_DIR / "monday_reset_status.json"
RUNBOOK_STATUS_PATHS: dict[str, Path] = {
    "market_refresh_v2": MARKET_REFRESH_STATUS_PATH,
    "monday_reset_v1": MONDAY_RESET_STATUS_PATH,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def status_path_for_runbook(runbook_id: str) -> Path:
    return RUNBOOK_STATUS_PATHS.get(str(runbook_id), MARKET_REFRESH_STATUS_PATH)


def _base_status(runbook_id: str) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "runbook_id": runbook_id,
        "status": "idle",
        "mode": None,
        "current_step_id": None,
        "current_step_name": None,
        "current_substep_id": None,
        "current_substep_name": None,
        "step_index": None,
        "step_total": None,
        "substep_index": None,
        "substep_total": None,
        "started_at": None,
        "completed_at": None,
        "updated_at": now,
        "message": None,
        "error": None,
        "history": [],
    }


def read_status(path: Path | None = None, runbook_id: str = "market_refresh_v2") -> dict[str, Any]:
    resolved_path = path or status_path_for_runbook(runbook_id)
    if not resolved_path.exists():
        return _base_status(runbook_id)
    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            status = json.load(file) or {}
    except Exception:
        return _base_status(runbook_id)
    merged = _base_status(str(status.get("runbook_id") or runbook_id))
    merged.update(status)
    if not isinstance(merged.get("history"), list):
        merged["history"] = []
    return merged


def write_status(status: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    ensure_data_dirs()
    status = dict(status)
    resolved_path = path or status_path_for_runbook(str(status.get("runbook_id") or "market_refresh_v2"))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = utc_now_iso()
    with resolved_path.open("w", encoding="utf-8") as file:
        json.dump(status, file, indent=2, sort_keys=True)
        file.write("\n")
    return status


def append_history(status: dict[str, Any], event: str, message: str | None = None) -> dict[str, Any]:
    history = list(status.get("history") or [])
    history.append(
        {
            "timestamp": utc_now_iso(),
            "event": event,
            "message": message,
            "step_id": status.get("current_step_id"),
            "step_name": status.get("current_step_name"),
            "substep_id": status.get("current_substep_id"),
            "substep_name": status.get("current_substep_name"),
        }
    )
    status["history"] = history
    return status


def start_runbook(*, runbook_id: str, mode: str, step_total: int, message: str | None = None) -> dict[str, Any]:
    status = _base_status(runbook_id)
    now = utc_now_iso()
    status.update(
        {
            "status": "running",
            "mode": mode,
            "step_total": step_total,
            "started_at": now,
            "message": message or "Runbook started",
            "error": None,
            "history": [],
        }
    )
    append_history(status, "runbook_started", status["message"])
    return write_status(status)


def start_step(
    *,
    step_id: str,
    step_name: str,
    step_index: int,
    step_total: int,
    substep_total: int,
    message: str | None = None,
    runbook_id: str = "market_refresh_v2",
) -> dict[str, Any]:
    status = read_status(runbook_id=runbook_id)
    status.update(
        {
            "status": "running",
            "current_step_id": step_id,
            "current_step_name": step_name,
            "current_substep_id": None,
            "current_substep_name": None,
            "step_index": step_index,
            "step_total": step_total,
            "substep_index": None,
            "substep_total": substep_total,
            "message": message or f"Started step: {step_name}",
            "error": None,
        }
    )
    append_history(status, "step_started", status["message"])
    return write_status(status)


def start_substep(
    *,
    substep_id: str,
    substep_name: str,
    substep_index: int,
    substep_total: int,
    message: str | None = None,
    runbook_id: str = "market_refresh_v2",
) -> dict[str, Any]:
    status = read_status(runbook_id=runbook_id)
    status.update(
        {
            "status": "running",
            "current_substep_id": substep_id,
            "current_substep_name": substep_name,
            "substep_index": substep_index,
            "substep_total": substep_total,
            "message": message or f"Started substep: {substep_name}",
            "error": None,
        }
    )
    append_history(status, "substep_started", status["message"])
    return write_status(status)


def complete_substep(message: str | None = None, *, runbook_id: str = "market_refresh_v2") -> dict[str, Any]:
    status = read_status(runbook_id=runbook_id)
    append_history(status, "substep_completed", message or "Substep completed")
    status["message"] = message or "Substep completed"
    return write_status(status)


def complete_step(message: str | None = None, *, runbook_id: str = "market_refresh_v2") -> dict[str, Any]:
    status = read_status(runbook_id=runbook_id)
    append_history(status, "step_completed", message or "Step completed")
    status["message"] = message or "Step completed"
    return write_status(status)


def complete_runbook(message: str | None = None, *, runbook_id: str = "market_refresh_v2") -> dict[str, Any]:
    status = read_status(runbook_id=runbook_id)
    status.update(
        {
            "status": "completed",
            "completed_at": utc_now_iso(),
            "message": message or "Runbook completed",
            "error": None,
        }
    )
    append_history(status, "runbook_completed", status["message"])
    return write_status(status)


def fail_runbook(error: str, *, runbook_id: str = "market_refresh_v2") -> dict[str, Any]:
    status = read_status(runbook_id=runbook_id)
    status.update(
        {
            "status": "failed",
            "completed_at": utc_now_iso(),
            "message": "Runbook failed",
            "error": str(error),
        }
    )
    append_history(status, "runbook_failed", str(error))
    return write_status(status)
