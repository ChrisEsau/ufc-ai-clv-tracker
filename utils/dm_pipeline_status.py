from pathlib import Path

import streamlit as st

from pipeline.common.paths import (
    APPEND_PRECHECK_PATH,
    STAGED_DERIVED_STATS_AUDIT_PATH,
    STAGED_FIGHT_ROWS_PATH,
    STAGED_MASTER_ROWS_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
)


def artifact_exists(path):
    return Path(path).exists()


def render_pipeline_status():

    st.subheader("Pipeline Status")

    pipeline = [
        (
            "SCRAPE",
            artifact_exists(STAGED_FIGHT_ROWS_PATH),
        ),
        (
            "MAP",
            artifact_exists(STAGED_MASTER_ROWS_PATH),
        ),
        (
            "DERIVED",
            artifact_exists(STAGED_DERIVED_STATS_AUDIT_PATH),
        ),
        (
            "ENRICH",
            artifact_exists(STAGED_MASTER_ROWS_PROFILED_PATH),
        ),
        (
            "VALIDATE",
            artifact_exists(APPEND_PRECHECK_PATH),
        ),
    ]

    cols = st.columns(len(pipeline))

    for col, (name, complete) in zip(cols, pipeline):

        with col:

            if complete:
                st.success("✓")

            else:
                st.warning("○")

            st.caption(name)
