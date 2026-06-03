from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import (
    FIGHT_DETAIL_SCRAPE_AUDIT_PATH,
    FIGHT_SCRAPE_AUDIT_PATH,
    STAGED_FIGHT_DETAILS_PATH,
    STAGED_FIGHT_ROWS_PATH,
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


def render_fight_scrape():

    with st.expander(
        "🥊 Fight Scrape",
        expanded=False,
    ):

        st.caption(
            "Stages fights from missing UFCStats events, then scrapes detailed fight stats."
        )

        if st.button(
            "Run Fight Scrape",
            use_container_width=True,
            key="run_fight_scrape",
        ):
            ok, msg = trigger_workflow(
                "run-ufcstats-fight-scrape.yml"
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)

        if st.button(
            "Run Detail Scrape",
            use_container_width=True,
            key="run_detail_scrape",
        ):
            ok, msg = trigger_workflow(
                "run-ufcstats-fight-detail-scrape.yml"
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.markdown("#### Fight Scrape Artifacts")

        render_artifact_summary(
            "Staged Fight Rows",
            STAGED_FIGHT_ROWS_PATH,
        )

        render_artifact_summary(
            "Fight Scrape Audit",
            FIGHT_SCRAPE_AUDIT_PATH,
        )

        render_artifact_summary(
            "Staged Fight Details",
            STAGED_FIGHT_DETAILS_PATH,
        )

        render_artifact_summary(
            "Fight Detail Scrape Audit",
            FIGHT_DETAIL_SCRAPE_AUDIT_PATH,
        )
