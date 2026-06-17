from __future__ import annotations

from typing import Any

from utils.github_actions import get_latest_workflow_run
from utils.operations_runbook_state import load_state, save_state, set_completed, set_failed


RUNNING_GITHUB_STATUSES = {"queued", "in_progress", "requested", "waiting", "pending"}
SUCCESS_CONCLUSION = "success"
FAILURE_CONCLUSIONS = {"failure", "cancelled", "timed_out", "action_required", "startup_failure"}


def _matches_active_run(latest_run: dict[str, Any], active_run_id: str | None) -> bool:
    if not active_run_id:
        return True
    return str(latest_run.get("id") or "") == str(active_run_id)


def refresh_active_workflow_status() -> tuple[bool, str, dict[str, Any]]:
    """Refresh persisted Operations state from the active GitHub workflow run.

    This only monitors the currently active workflow. It does not auto-advance to
    the next workflow; the next orchestration step will do that after this status
    check is validated in the dashboard.
    """

    state = load_state()
    workflow_file = state.get("current_workflow_file")
    if not workflow_file:
        return False, "No active workflow to refresh.", state

    ok, message, latest_run = get_latest_workflow_run(str(workflow_file))
    if not ok:
        return False, message, state
    if not latest_run:
        return False, f"No workflow runs found for {workflow_file}.", state

    active_run_id = state.get("current_workflow_run_id")
    if not _matches_active_run(latest_run, None if active_run_id is None else str(active_run_id)):
        return False, "Latest workflow run does not match active run id yet.", state

    github_status = str(latest_run.get("status") or "").lower()
    conclusion = str(latest_run.get("conclusion") or "").lower()
    html_url = latest_run.get("html_url")

    state["current_workflow_run_id"] = str(latest_run.get("id") or active_run_id or "") or None
    state["current_workflow_status"] = github_status or None
    state["current_workflow_conclusion"] = conclusion or None
    state["current_workflow_url"] = html_url

    if github_status in RUNNING_GITHUB_STATUSES or not conclusion:
        saved = save_state(state)
        return True, f"Workflow still running: {workflow_file} ({github_status or 'unknown'}).", saved

    if conclusion == SUCCESS_CONCLUSION:
        saved = save_state(state)
        return True, f"Workflow completed successfully: {workflow_file}.", saved

    if conclusion in FAILURE_CONCLUSIONS:
        failed = set_failed(error=f"Workflow failed: {workflow_file} ({conclusion})")
        failed["current_workflow_status"] = github_status or None
        failed["current_workflow_conclusion"] = conclusion or None
        failed["current_workflow_url"] = html_url
        saved = save_state(failed)
        return False, f"Workflow failed: {workflow_file} ({conclusion}).", saved

    saved = save_state(state)
    return True, f"Workflow status refreshed: {workflow_file} ({github_status}/{conclusion}).", saved
