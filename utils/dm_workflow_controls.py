import streamlit as st

from utils.github_actions import trigger_workflow


WORKFLOWS = [
    {
        "label": "Run Dataset Status",
        "workflow": "run-dataset-status.yml",
    },
    {
        "label": "Run Event Check",
        "workflow": "run-ufcstats-event-check.yml",
    },
    {
        "label": "Run Fight Scrape",
        "workflow": "run-ufcstats-fight-scrape.yml",
    },
    {
        "label": "Run Detail Scrape",
        "workflow": "run-ufcstats-fight-detail-scrape.yml",
    },
    {
        "label": "Run Mapper",
        "workflow": "run-staged-master-mapper.yml",
    },
    {
        "label": "Run Derived Stats",
        "workflow": "run-staged-derived-stats-transformer.yml",
    },
    {
        "label": "Run Fighter Enrichment",
        "workflow": "run-fighter-profile-enrichment.yml",
    },
    {
        "label": "Run Column Validation",
        "workflow": "run-master-column-validation.yml",
    },
    {
        "label": "Run Append Precheck",
        "workflow": "run-append-precheck-validation.yml",
    },
]


def render_workflow_button(label, workflow):
    if st.button(
        label,
        use_container_width=True,
        key=f"workflow_{workflow}",
    ):
        ok, msg = trigger_workflow(workflow)

        if ok:
            st.success(msg)
        else:
            st.error(msg)


def render_workflow_controls():
    st.subheader("Workflow Controls")

    st.caption(
        "Launch GitHub Actions workflows from the Data Maintenance dashboard."
    )

    with st.expander("Pipeline Workflow Launchers", expanded=False):

        for item in WORKFLOWS:
            render_workflow_button(
                item["label"],
                item["workflow"],
            )