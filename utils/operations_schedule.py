from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEDULE_PATH = Path("data/status/operations_schedule.json")
SCHEDULER_STATUS_PATH = Path("data/status/operations_scheduler_status.json")
DEFAULT_TIMEZONE = "America/Chicago"
DAY_ORDER = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
RUNBOOK_ORDER = ["monday_reset_v1", "market_refresh_v2", "fight_day_monitor_v1"]
DEFAULT_WINDOW_MINUTES = 30
DEFAULT_CATCHUP_GRACE_MINUTES = 1440

DEFAULT_SCHEDULE: dict[str, Any] = {
    "version": 1,
    "timezone": DEFAULT_TIMEZONE,
    "window_minutes": DEFAULT_WINDOW_MINUTES,
    "catchup_grace_minutes": DEFAULT_CATCHUP_GRACE_MINUTES,
    "schedules": {
        "monday_reset_v1": {
            "enabled": False,
            "display_name": "Monday Reset",
            "days": ["MON"],
            "times": ["08:00"],
            "catchup_grace_minutes": 2880,
            "workflow_file": "run-monday-reset-orchestrator.yml",
            "inputs": {
                "mode": "production",
                "max_events": "all",
                "auto_append": True,
                "skip_bankroll": False,
                "skip_clv": False,
            },
        },
        "market_refresh_v2": {
            "enabled": False,
            "display_name": "Market Refresh",
            "days": ["FRI", "SAT"],
            "times": ["09:00", "15:00", "21:00"],
            "catchup_grace_minutes": 1440,
            "workflow_file": "run-market-refresh-orchestrator.yml",
            "inputs": {
                "mode": "production",
                "max_draftkings_events": "all",
                "model_mode": "production",
                "snapshot_model_mode": "production",
            },
        },
        "fight_day_monitor_v1": {
            "enabled": False,
            "display_name": "Fight Day Monitor",
            "days": ["SAT"],
            "times": ["17:30", "17:55"],
            "catchup_grace_minutes": 180,
            "workflow_file": "run-fight-day-monitor.yml",
            "inputs": {
                "mode": "production",
                "max_draftkings_events": "all",
                "model_mode": "production",
                "snapshot_model_mode": "production",
                "official_closing_snapshot": True,
            },
        },
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_schedule(path: Path = SCHEDULE_PATH) -> dict[str, Any]:
    if not path.exists():
        return deepcopy(DEFAULT_SCHEDULE)
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file) or {}
    except Exception:
        return deepcopy(DEFAULT_SCHEDULE)
    return normalize_schedule(raw)


def write_schedule(schedule: dict[str, Any], path: Path = SCHEDULE_PATH) -> dict[str, Any]:
    normalized = normalize_schedule(schedule)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(normalized, file, indent=2, sort_keys=True)
        file.write("\n")
    return normalized


def load_scheduler_status(path: Path = SCHEDULER_STATUS_PATH) -> dict[str, Any]:
    if not path.exists():
        return _empty_scheduler_status()
    try:
        with path.open("r", encoding="utf-8") as file:
            status = json.load(file) or {}
    except Exception:
        return _empty_scheduler_status()
    base = _empty_scheduler_status()
    base.update(status)
    if not isinstance(base.get("dispatch_history"), list):
        base["dispatch_history"] = []
    return base


def write_scheduler_status(status: dict[str, Any], path: Path = SCHEDULER_STATUS_PATH) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = dict(status)
    status["updated_at"] = utc_now_iso()
    with path.open("w", encoding="utf-8") as file:
        json.dump(status, file, indent=2, sort_keys=True)
        file.write("\n")
    return status


def normalize_schedule(raw: dict[str, Any]) -> dict[str, Any]:
    schedule = deepcopy(DEFAULT_SCHEDULE)
    if isinstance(raw, dict):
        schedule["version"] = _safe_int(raw.get("version"), schedule["version"])
        schedule["timezone"] = str(raw.get("timezone") or schedule["timezone"])
        schedule["window_minutes"] = _safe_int(raw.get("window_minutes"), schedule["window_minutes"])
        schedule["catchup_grace_minutes"] = _safe_int(
            raw.get("catchup_grace_minutes"),
            schedule["catchup_grace_minutes"],
        )
        raw_schedules = raw.get("schedules") or {}
        if isinstance(raw_schedules, dict):
            for runbook_id in RUNBOOK_ORDER:
                existing = schedule["schedules"][runbook_id]
                incoming = raw_schedules.get(runbook_id) or {}
                if not isinstance(incoming, dict):
                    continue
                existing["enabled"] = bool(incoming.get("enabled", existing["enabled"]))
                existing["display_name"] = str(incoming.get("display_name") or existing["display_name"])
                existing["days"] = normalize_days(incoming.get("days", existing["days"]))
                existing["times"] = normalize_times(incoming.get("times", existing["times"]))
                existing["workflow_file"] = str(incoming.get("workflow_file") or existing["workflow_file"])
                existing["catchup_grace_minutes"] = _safe_int(
                    incoming.get("catchup_grace_minutes"),
                    existing.get("catchup_grace_minutes", schedule["catchup_grace_minutes"]),
                )
                inputs = incoming.get("inputs")
                if isinstance(inputs, dict):
                    existing_inputs = dict(existing.get("inputs") or {})
                    existing_inputs.update(inputs)
                    existing["inputs"] = existing_inputs
    return schedule


def normalize_days(days: Any) -> list[str]:
    if isinstance(days, str):
        parts = [part.strip().upper() for part in days.replace(";", ",").split(",")]
    elif isinstance(days, list):
        parts = [str(part).strip().upper() for part in days]
    else:
        parts = []
    valid = [day for day in parts if day in DAY_ORDER]
    return sorted(set(valid), key=DAY_ORDER.index) or ["MON"]


def normalize_times(times: Any) -> list[str]:
    if isinstance(times, str):
        parts = [part.strip() for part in times.replace(";", ",").split(",")]
    elif isinstance(times, list):
        parts = [str(part).strip() for part in times]
    else:
        parts = []
    valid: list[str] = []
    for part in parts:
        try:
            parsed = time.fromisoformat(part)
            valid.append(parsed.strftime("%H:%M"))
        except Exception:
            continue
    return sorted(set(valid)) or ["08:00"]


def get_due_runbooks(
    *,
    schedule: dict[str, Any],
    status: dict[str, Any],
    now_utc: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    now_utc = now_utc or datetime.now(timezone.utc)
    timezone_name = str(schedule.get("timezone") or DEFAULT_TIMEZONE)
    tz = ZoneInfo(timezone_name)
    now_local = now_utc.astimezone(tz)
    default_catchup_minutes = _safe_int(
        schedule.get("catchup_grace_minutes"),
        _safe_int(schedule.get("window_minutes"), DEFAULT_WINDOW_MINUTES),
    )
    dispatch_history = status.get("dispatch_history") or []
    dispatched_keys = {str(row.get("dispatch_key")) for row in dispatch_history if row.get("dispatch_key")}

    due: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    schedules = schedule.get("schedules") or {}

    for runbook_id in RUNBOOK_ORDER:
        cfg = schedules.get(runbook_id) or {}
        if not bool(cfg.get("enabled")):
            skipped.append({"runbook_id": runbook_id, "reason": "disabled"})
            continue
        days = normalize_days(cfg.get("days"))
        times = normalize_times(cfg.get("times"))
        catchup_minutes = max(1, _safe_int(cfg.get("catchup_grace_minutes"), default_catchup_minutes))
        for scheduled_local in _scheduled_datetimes_in_window(
            now_local=now_local,
            scheduled_times=times,
            lookback_minutes=catchup_minutes,
        ):
            day = DAY_ORDER[scheduled_local.weekday()]
            if day not in days:
                continue
            minutes_late = (now_local - scheduled_local).total_seconds() / 60.0
            if minutes_late < 0 or minutes_late > catchup_minutes:
                continue
            dispatch_key = f"{runbook_id}:{scheduled_local.strftime('%Y-%m-%dT%H:%M')}:{timezone_name}"
            row = {
                "runbook_id": runbook_id,
                "display_name": cfg.get("display_name", runbook_id),
                "workflow_file": cfg.get("workflow_file"),
                "inputs": cfg.get("inputs") or {},
                "scheduled_time": scheduled_local.strftime("%H:%M"),
                "scheduled_at_local": scheduled_local.isoformat(),
                "dispatch_key": dispatch_key,
                "minutes_late": round(minutes_late, 2),
                "catchup_grace_minutes": catchup_minutes,
            }
            if dispatch_key in dispatched_keys:
                skipped.append({**row, "reason": "already_dispatched"})
            else:
                due.append(row)
    return sorted(due, key=lambda row: row["scheduled_at_local"]), skipped


def next_due_runbook(schedule: dict[str, Any], now_utc: datetime | None = None) -> dict[str, Any] | None:
    now_utc = now_utc or datetime.now(timezone.utc)
    timezone_name = str(schedule.get("timezone") or DEFAULT_TIMEZONE)
    tz = ZoneInfo(timezone_name)
    now_local = now_utc.astimezone(tz)
    schedules = schedule.get("schedules") or {}
    candidates: list[dict[str, Any]] = []
    for days_ahead in range(0, 14):
        candidate_date = now_local.date() + timedelta(days=days_ahead)
        for runbook_id in RUNBOOK_ORDER:
            cfg = schedules.get(runbook_id) or {}
            if not bool(cfg.get("enabled")):
                continue
            days = normalize_days(cfg.get("days"))
            for scheduled_time in normalize_times(cfg.get("times")):
                hour, minute = [int(part) for part in scheduled_time.split(":")]
                dt = datetime.combine(candidate_date, time(hour, minute), tzinfo=tz)
                if dt <= now_local:
                    continue
                if DAY_ORDER[dt.weekday()] not in days:
                    continue
                candidates.append(
                    {
                        "runbook_id": runbook_id,
                        "display_name": cfg.get("display_name", runbook_id),
                        "workflow_file": cfg.get("workflow_file"),
                        "scheduled_at_local": dt.isoformat(),
                        "scheduled_time": scheduled_time,
                        "catchup_grace_minutes": _safe_int(
                            cfg.get("catchup_grace_minutes"),
                            _safe_int(schedule.get("catchup_grace_minutes"), DEFAULT_CATCHUP_GRACE_MINUTES),
                        ),
                    }
                )
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row["scheduled_at_local"])[0]


def _scheduled_datetimes_in_window(
    *,
    now_local: datetime,
    scheduled_times: list[str],
    lookback_minutes: int,
) -> list[datetime]:
    lookback_start = now_local - timedelta(minutes=lookback_minutes)
    lookback_days = max(1, int(lookback_minutes / 1440) + 2)
    candidates: list[datetime] = []
    for days_back in range(0, lookback_days + 1):
        candidate_date = now_local.date() - timedelta(days=days_back)
        for scheduled_time in scheduled_times:
            hour, minute = [int(part) for part in scheduled_time.split(":")]
            candidate = datetime.combine(candidate_date, time(hour, minute), tzinfo=now_local.tzinfo)
            if lookback_start <= candidate <= now_local:
                candidates.append(candidate)
    return sorted(candidates)


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _empty_scheduler_status() -> dict[str, Any]:
    return {
        "status": "idle",
        "last_checked_at": None,
        "updated_at": None,
        "timezone": DEFAULT_TIMEZONE,
        "due_runbooks": [],
        "dispatched_runbooks": [],
        "skipped_runbooks": [],
        "errors": [],
        "next_due_runbook": None,
        "dispatch_history": [],
    }
