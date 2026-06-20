from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Any, Sequence

import requests

from utils.operations_schedule import (
    DEFAULT_TIMEZONE,
    get_due_runbooks,
    load_schedule,
    load_scheduler_status,
    next_due_runbook,
    write_scheduler_status,
)

GITHUB_API_BASE = "https://api.github.com"


def _github_context() -> tuple[str, str, str, str]:
    repository = os.getenv("GITHUB_REPOSITORY", "")
    if "/" in repository:
        owner, repo = repository.split("/", 1)
    else:
        owner = os.getenv("GITHUB_OWNER", "")
        repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "") or os.getenv("GH_TOKEN", "")
    ref = os.getenv("GITHUB_REF_NAME", "") or os.getenv("GITHUB_BRANCH", "dev")
    return owner, repo, token, ref


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _dispatch_workflow(*, workflow_file: str, inputs: dict[str, Any], dry_run: bool) -> tuple[bool, str]:
    owner, repo, token, ref = _github_context()
    if dry_run:
        return True, f"DRY RUN: would dispatch {workflow_file} on {ref} with {inputs}"
    if not owner or not repo or not token:
        return False, "Missing GitHub repository or token context."

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/actions/workflows/{workflow_file}/dispatches"
    payload = {"ref": ref, "inputs": inputs or {}}
    response = requests.post(url, headers=_headers(token), json=payload, timeout=30)
    if response.status_code in {200, 201, 202, 204}:
        return True, f"Dispatched {workflow_file}"
    return False, f"GitHub API error {response.status_code}: {response.text}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch due Operations Center runbooks from operations_schedule.json.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate due runbooks without dispatching workflows.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dry_run = bool(args.dry_run)
    now_utc = datetime.now(timezone.utc)
    schedule = load_schedule()
    status = load_scheduler_status()
    due, skipped = get_due_runbooks(schedule=schedule, status=status, now_utc=now_utc)

    dispatched: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print("=" * 80)
    print("OPERATIONS CENTER SCHEDULER")
    print("=" * 80)
    print("Checked at:", now_utc.isoformat())
    print("Timezone:", schedule.get("timezone", DEFAULT_TIMEZONE))
    print("Dry run:", dry_run)
    print("Due runbooks:", len(due))

    for row in due:
        ok, message = _dispatch_workflow(
            workflow_file=str(row.get("workflow_file") or ""),
            inputs=row.get("inputs") or {},
            dry_run=dry_run,
        )
        record = {
            **row,
            "dispatched_at": now_utc.isoformat(),
            "dry_run": dry_run,
            "message": message,
        }
        if ok:
            dispatched.append(record)
            label = "DRY RUN" if dry_run else "DISPATCHED"
            print(f"{label}:", row.get("runbook_id"), message)
        else:
            errors.append(record)
            print("ERROR:", row.get("runbook_id"), message)

    history = list(status.get("dispatch_history") or [])
    if not dry_run:
        history.extend(dispatched)
    history = history[-250:]

    new_status = {
        "status": "dry_run_completed" if dry_run and not errors else "completed" if not errors else "completed_with_errors",
        "last_checked_at": now_utc.isoformat(),
        "timezone": schedule.get("timezone", DEFAULT_TIMEZONE),
        "due_runbooks": due,
        "dispatched_runbooks": dispatched,
        "skipped_runbooks": skipped,
        "errors": errors,
        "next_due_runbook": next_due_runbook(schedule, now_utc=now_utc),
        "dispatch_history": history,
    }
    write_scheduler_status(new_status)

    print()
    print("========== SCHEDULER SUMMARY ==========")
    print("Dispatched:", len(dispatched))
    print("Skipped:", len(skipped))
    print("Errors:", len(errors))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
