import streamlit as st
import pandas as pd
from pathlib import Path


def artifact_exists(path):
    return Path(path).exists()


def render_pipeline_status():

    st.subheader("Pipeline Status")

    pipeline = [
        (
            "SCRAPE",
            artifact_exists("ufc_staged_fight_rows.parquet"),
        ),
        (
            "MAP",
            artifact_exists("ufc_staged_master_rows.parquet"),
        ),
        (
            "DERIVED",
            artifact_exists("ufc_staged_derived_stats_audit.parquet"),
        ),
        (
            "ENRICH",
            artifact_exists("ufc_staged_master_rows_profiled.parquet"),
        ),
        (
            "VALIDATE",
            artifact_exists("ufc_append_precheck.parquet"),
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