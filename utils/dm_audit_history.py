from pathlib import Path

import pandas as pd
import streamlit as st


AUDIT_ARTIFACTS = [
    (
        "Dataset Status",
        "ufc_dataset_status.parquet",
    ),
    (
        "Event Check",
        "ufc_ufcstats_event_check.parquet",
    ),
    (
        "Fight Scrape Audit",
        "ufc_fight_scrape_audit.parquet",
    ),
    (
        "Fight Detail Audit",
        "ufc_fight_detail_scrape_audit.parquet",
    ),
    (
        "Derived Stats Audit",
        "ufc_staged_derived_stats_audit.parquet",
    ),
    (
        "Fighter Profile Audit",
        "ufc_fighter_profile_scrape_audit.parquet",
    ),
    (
        "Column Validation",
        "ufc_master_column_validation.parquet",
    ),
    (
        "Append Precheck",
        "ufc_append_precheck.parquet",
    ),
    (
        "Append Audit",
        "ufc_append_audit.parquet",
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
                f"Rows: {len(audit_df)} | "
                f"Columns: {len(audit_df.columns)}"
            )

            st.dataframe(
                audit_df.head(100),
                use_container_width=True,
                hide_index=True,
            )