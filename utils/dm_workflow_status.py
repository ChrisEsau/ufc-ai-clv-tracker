from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from utils.github_actions import get_latest_workflow_run


SESSION_KEY = "dm_workflow_status"

STATUS_DISPLAY = {
    "queued": ("⚪", "Queued", 15),
    "in_progress": ("🟡", "Running", 60),
    "completed": ("✅", "Completed", 100),
}

CONCLUSION_DISPLAY = {
    "success": ("✅", "Succeeded", 100),
    "failure": ("❌", "Failed", 100),
    "cancelled": ("⚠️", "Cancelled", 100),
    "skipped": ("⚠️", "Skipped", 100),
    "timed_out": ("⚠️", "Timed out", 100),
    "action_required": ("⚠️", "Action required", 100),
}


def workflow_state():
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = {}

    return st.session_state[SESSION_KEY]


def remember_launched_workflow(status_key, label, workflow_file, inputs=None):
    workflow_state()[status_key] = {
        "label": label,
        "workflow_file": workflow_file,
        "inputs": inputs or {},
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_github_timestamp(value):
    if not value:
        return None

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def format_timestamp(value):
    parsed = parse_github_timestamp(value) if isinstance(value, str) else value

    if parsed is None:
        return None

    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def workflow_display(run):
    status = run.get("status")
    conclusion = run.get("conclusion")

    if status == "completed":
        return CONCLUSION_DISPLAY.get(
            conclusion,
            ("ℹ️", f"Completed: {conclusion or 'unknown'}", 100),
        )

    return STATUS_DISPLAY.get(status, ("ℹ️", status or "Unknown", 0))


def latest_run_is_stale(run, launched_at):
    created_at = parse_github_timestamp(run.get("created_at"))

    if created_at is None or launched_at is None:
        return False

    return created_at < launched_at - timedelta(seconds=30)


def render_workflow_status(status_key):
    tracked = workflow_state().get(status_key)

    if not tracked:
        return

    label = tracked["label"]
    workflow_file = tracked["workflow_file"]
    launched_at = parse_github_timestamp(tracked.get("launched_at"))

    st.markdown("#### Workflow Status")

    ok, msg, run = get_latest_workflow_run(workflow_file)

    if not ok:
        st.error(msg)
        return

    if run is None:
        st.warning(
            "Workflow dispatch was accepted, but no GitHub run is visible yet. "
            "Wait a few seconds and refresh status."
        )
        return

    icon, status_label, progress_value = workflow_display(run)
    run_url = run.get("html_url")
    stale = latest_run_is_stale(run, launched_at)

    if stale:
        st.warning(
            "Latest visible workflow run appears older than this dashboard launch. "
            "GitHub may still be creating the new run. Refresh in a few seconds."
        )

    status_df = pd.DataFrame(
        [
            {
                "Workflow": label,
                "Status": f"{icon} {status_label}",
                "Branch": run.get("head_branch"),
                "Run Number": run.get("run_number"),
                "Started": format_timestamp(run.get("run_started_at")),
                "Updated": format_timestamp(run.get("updated_at")),
            }
        ]
    )

    st.progress(progress_value)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    controls = st.columns([1, 1])

    with controls[0]:
        if run_url:
            st.link_button("Open GitHub Run", run_url, use_container_width=True)

    with controls[1]:
        if st.button(
            "Refresh Workflow Status",
            use_container_width=True,
            key=f"refresh_workflow_status_{status_key}",
        ):
            st.rerun()
