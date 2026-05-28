import streamlit as st

from utils.github_actions import trigger_workflow


def render_workflow_controls():
    st.subheader("Workflow Controls")

    st.caption(
        "Launch GitHub Actions workflows from the Data Maintenance dashboard."
    )

    if st.button(
        "Run Event Check",
        use_container_width=True,
    ):
        ok, msg = trigger_workflow(
            "run-ufcstats-event-check.yml"
        )

        if ok:
            st.success(msg)
        else:
            st.error(msg)