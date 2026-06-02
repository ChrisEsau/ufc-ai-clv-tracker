from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import (
    APPEND_AUDIT_PATH,
    APPEND_PRECHECK_PATH,
    DATASET_STATUS_PATH,
    FIGHT_DETAIL_SCRAPE_AUDIT_PATH,
    FIGHT_SCRAPE_AUDIT_PATH,
    FIGHTER_PROFILE_SCRAPE_AUDIT_PATH,
    MASTER_COLUMN_VALIDATION_PATH,
    STAGED_DERIVED_STATS_AUDIT_PATH,
    UFCSTATS_EVENT_CHECK_PATH,
)


AUDIT_ARTIFACTS = [
    (
        "Dataset Status",
        DATASET_STATUS_PATH,
    ),
    (
        "Event Check",
        UFCSTATS_EVENT_CHECK_PATH,
    ),
    (
        "Fight Scrape Audit",
        FIGHT_SCRAPE_AUDIT_PATH,
    ),
    (
        "Fight Detail Audit",
        FIGHT_DETAIL_SCRAPE_AUDIT_PATH,
    ),
    (
        "Derived Stats Audit",
        STAGED_DERIVED_STATS_AUDIT_PATH,
    ),
    (
        "Fighter Profile Audit",
        FIGHTER_PROFILE_SCRAPE_AUDIT_PATH,
    ),
    (
        "Column Validation",
        MASTER_COLUMN_VALIDATION_PATH,
    ),
    (
        "Append Precheck",
        APPEND_PRECHECK_PATH,
    ),
    (
        "Append Audit",
        APPEND_AUDIT_PATH,
    ),
]


def safe_read_parquet(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)

    except Exception:
        return None


def render_audit_history():

    with st.expander(
        "📜 Audit History",
        expanded=False,
    ):

        summary_rows = []

        for label, artifact in AUDIT_ARTIFACTS:

            path = Path(artifact)

            exists = path.exists()

            modified = (
                pd.Timestamp(path.stat().st_mtime, unit="s")
                if exists
                else None
            )

            summary_rows.append(
                {
                    "Artifact": label,
                    "Path": str(path),
                    "Exists": exists,
                    "Last Updated": modified,
                }
            )

        summary_df = pd.DataFrame(summary_rows)

        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("### Audit Details")

        selected = st.selectbox(
            "Select Audit Artifact",
            options=[x[0] for x in AUDIT_ARTIFACTS],
        )

        artifact_path = dict(AUDIT_ARTIFACTS)[selected]

        audit_df = safe_read_parquet(
            artifact_path
        )

        if audit_df is None:

            st.warning(
                f"{artifact_path} not found."
            )

        else:

            st.caption(
                f"Path: {artifact_path} | "
                f"Rows: {len(audit_df)} | "
                f"Columns: {len(audit_df.columns)}"
            )

            st.dataframe(
                audit_df.head(100),
                use_container_width=True,
                hide_index=True,
            )
