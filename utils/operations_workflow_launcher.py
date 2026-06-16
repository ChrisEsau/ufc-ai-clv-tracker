from __future__ import annotations

from typing import Any

from utils.github_actions import get_latest_workflow_run, trigger_workflow
from utils.operations_runbook_registry import DEFAULT_RUNBOOK_ID, get_runbook
from utils.operations_runbook_state import is_active, load_state, set_failed, set_running


class OperationsWorkflowLauncherError(RuntimeError):
    """Raised when Operations Center cannot launch a mapped workflow."""


def _current_position(state: dict[str, Any]) -> tuple[int, int]:
    if is_active(state):
        step_index = state.get("current_step_index")
        workflow_index = state.get("current_workflow_index")
        if isinstance(step_index, int) and isinstance(workflow_index, int):
            return step_index, workflow_index + 1
    return 0, 0


def next_workflow(runbook_id: str = DEFAULT_RUNBOOK_ID) -> tuple[dict[str, Any], dict[str, Any], int, int] | None:
    """Return the next workflow spec from the persisted runbook state."""

    state = load_state()
    runbook = get_runbook(runbook_id or str(state.get("runbook_id") or DEFAULT_RUNBOOK_ID))
    start_step_index, start_workflow_index = _current_position(state)

    steps = runbook.get("steps", [])
    for step_index in range(start_step_index, len(steps)):
        step = steps[step_index]
        workflows = step.get("workflows", [])
        workflow_start = start_workflow_index if step_index == start_step_index else 0
        for workflow_index in range(workflow_start, len(workflows)):
            return step, workflows[workflow_index], step_index, workflow_index
    return None


def launch_next_workflow(runbook_id: str = DEFAULT_RUNBOOK_ID) -> tuple[bool, str, dict[str, Any]]:
    """Launch the next mapped workflow and persist active workflow state.

    This does not poll or auto-advance. It is intentionally one workflow at a time
    so the Operations Center can validate dispatch and state before orchestration
    is expanded.
    """

    selection = next_workflow(runbook_id)
    if selection is None:
        return False, "No remaining workflows to launch.", load_state()

    step, workflow, step_index, workflow_index = selection
    workflow_file = str(workflow.get("workflow_file") or "")
    if not workflow_file:
        state = set_failed(error="Selected workflow is missing workflow_file.")
        return False, "Selected workflow is missing workflow_file.", state

    inputs = workflow.get("inputs") or {}
    ok, message = trigger_workflow(workflow_file, inputs=inputs)
    if not ok:
        state = set_failed(error=message)
        return False, message, state

    workflow_run_id = None
    run_ok, _run_message, latest_run = get_latest_workflow_run(workflow_file)
    if run_ok and latest_run:
        workflow_run_id = latest_run.get("id")

    state = set_running(
        runbook_id=runbook_id,
        step_id=str(step.get("step_id")),
        step_index=step_index,
        workflow_file=workflow_file,
        workflow_index=workflow_index,
        workflow_run_id=workflow_run_id,
    )
    return True, message, state
