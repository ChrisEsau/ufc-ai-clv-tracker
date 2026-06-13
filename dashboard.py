# ============================================================
# UFC BETTING INTELLIGENCE DASHBOARD
# ============================================================


from datetime import datetime, timedelta, timezone

import streamlit as st

from utils.theme import apply_theme
from tabs.betting_board_v2 import render_betting_board
from tabs.line_movement import render_line_movement
from tabs.bankroll import render_bankroll
from tabs.model_lab import render_model_lab
from tabs.data_maintenance import render_data_maintenance
from utils.github_actions import get_latest_workflow_run
from utils.sidebar import render_sidebar

# ============================================================
# PAGE CONFIG
# ============================================================

apply_theme()

# The Model Lab feature contract is derived from selected/resolved features.
# Keep the legacy value available to save logic, but do not render a manual
# "Expected Feature Count" number input in Configuration.
if not hasattr(st, "_ufc_original_number_input"):
    st._ufc_original_number_input = st.number_input


def _dashboard_number_input(label, *args, **kwargs):
    if str(label) == "Expected Feature Count":
        return kwargs.get("value", 0)
    return st._ufc_original_number_input(label, *args, **kwargs)


st.number_input = _dashboard_number_input

# Add CLV-style workflow feedback to Model Lab action buttons without changing
# the underlying workflow launchers. The wrapped button only records which
# workflow was requested; the existing Model Lab code still performs dispatch.
if not hasattr(st, "_ufc_original_button"):
    st._ufc_original_button = st.button

_MODEL_LAB_WORKFLOW_BY_BUTTON_KEY_PREFIX = {
    "mlab_build_": ("Build Feature View", "run-build-feature-view-v2.yml"),
    "mlab_train_": ("Train Model", "run-train-model-v2.yml"),
    "mlab_predict_": ("Run Predictions", "run-prediction-v2.yml"),
    "mlab_outcomes_": ("Run Outcomes", "run-betting-outcomes-v2.yml"),
}


def _dashboard_button(label, *args, **kwargs):
    clicked = st._ufc_original_button(label, *args, **kwargs)
    key = str(kwargs.get("key") or "")
    if clicked:
        for prefix, (workflow_label, workflow_file) in _MODEL_LAB_WORKFLOW_BY_BUTTON_KEY_PREFIX.items():
            if key.startswith(prefix):
                st.session_state["mlab_workflow_status"] = {
                    "label": workflow_label,
                    "workflow_file": workflow_file,
                    "launched_at": datetime.now(timezone.utc).isoformat(),
                    "running": True,
                }
                break
    return clicked


st.button = _dashboard_button


def _parse_github_time(value: str | None):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _render_model_lab_workflow_status() -> None:
    status = st.session_state.get("mlab_workflow_status") or {}
    if not status:
        return

    label = status.get("label", "Workflow")
    workflow_file = status.get("workflow_file")
    if not workflow_file:
        return

    ok, message, run = get_latest_workflow_run(workflow_file)
    if not ok:
        st.warning(f"{label} status unavailable: {message}")
        return

    launched_at = _parse_github_time(status.get("launched_at"))
    created_at = _parse_github_time((run or {}).get("created_at")) if run else None
    stale_run = bool(launched_at and created_at and created_at < launched_at - timedelta(seconds=30))

    if run is None or stale_run:
        st.info(f"{label} workflow queued...")
        return

    run_status = run.get("status")
    conclusion = run.get("conclusion")
    html_url = run.get("html_url")
    link = f" [Open workflow run]({html_url})" if html_url else ""

    if run_status == "completed":
        st.session_state["mlab_workflow_status"]["running"] = False
        if conclusion == "success":
            st.success(f"{label} workflow completed successfully." + link)
        else:
            st.warning(f"{label} workflow completed with status: {conclusion or 'unknown'}." + link)
        return

    st.info(f"{label} workflow running..." + link)


page = render_sidebar()

if page == "Betting Board":
    render_betting_board()
elif page == "Line Movement / CLV":
    render_line_movement()
elif page in {"Bankroll", "Bet Ledger / Bankroll"}:
    render_bankroll()
elif page == "Model Lab":
    render_model_lab()
    _render_model_lab_workflow_status()
elif page == "Data Maintenance":
    render_data_maintenance()
