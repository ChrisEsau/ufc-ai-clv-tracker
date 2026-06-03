from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import (
    FIGHTER_PROFILE_SCRAPE_AUDIT_PATH,
    STAGED_DERIVED_STATS_AUDIT_PATH,
    STAGED_FIGHTER_PROFILES_PATH,
    STAGED_MASTER_ROWS_ENRICHED_PATH,
    STAGED_MASTER_ROWS_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
)
from utils.github_actions import trigger_workflow


def safe_read_parquet(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def render_artifact_summary(label, path):
    df = safe_read_parquet(path)

    if df is None:
        st.warning(f"{label}: artifact not found at `{path}`")
        return

    st.success(f"{label}: found")
    st.caption(f"Path: {path} | Rows: {len(df)} | Columns: {len(df.columns)}")

    with st.expander(f"View {label}", expanded=False):
        st.dataframe(
            df.head(50),
            use_container_width=True,
            hide_index=True,
        )


def render_enrichment():

    with st.expander(
        "🧬 Enrichment",
        expanded=False,
    ):

        st.caption(
            "Maps staged fight data into the master schema, derives stats, "
            "and enriches rows with fighter profile data."
        )

        if st.button(
            "Run Mapper",
            use_container_width=True,
            key="run_mapper",
        ):
            ok, msg = trigger_workflow(
                "run-staged-master-mapper.yml"
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)

        if st.button(
            "Run Derived Stats",
            use_container_width=True,
            key="run_derived_stats",
        ):
            ok, msg = trigger_workflow(
                "run-staged-derived-stats-transformer.yml"
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)

        if st.button(
            "Run Fighter Enrichment",
            use_container_width=True,
            key="run_fighter_enrichment",
        ):
            ok, msg = trigger_workflow(
                "run-fighter-profile-enrichment.yml"
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.markdown("#### Enrichment Artifacts")

        render_artifact_summary(
            "Mapped Master Rows",
            STAGED_MASTER_ROWS_PATH,
        )

        render_artifact_summary(
            "Derived Stats Audit",
            STAGED_DERIVED_STATS_AUDIT_PATH,
        )

        render_artifact_summary(
            "Derived/Enriched Master Rows",
            STAGED_MASTER_ROWS_ENRICHED_PATH,
        )

        render_artifact_summary(
            "Fighter Profiles",
            STAGED_FIGHTER_PROFILES_PATH,
        )

        render_artifact_summary(
            "Profiled Master Rows",
            STAGED_MASTER_ROWS_PROFILED_PATH,
        )

        render_artifact_summary(
            "Fighter Profile Scrape Audit",
            FIGHTER_PROFILE_SCRAPE_AUDIT_PATH,
        )
