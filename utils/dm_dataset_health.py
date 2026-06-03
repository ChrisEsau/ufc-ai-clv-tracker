from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import DATASET_STATUS_PATH
from utils.github_actions import trigger_workflow
from utils.dm_workflow_status import (
    remember_launched_workflow,
    render_workflow_status,
)


def safe_read_parquet(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def render_dataset_health():

    status = safe_read_parquet(DATASET_STATUS_PATH)

    with st.expander(
        "📊 Dataset Health",
        expanded=False,
    ):

        if st.button(
            "Run Dataset Status",
            use_container_width=True,
            key="run_dataset_status",
        ):
            ok, msg = trigger_workflow(
                "run-dataset-status.yml"
            )

            if ok:
                remember_launched_workflow(
                    "dataset_status",
                    "Run Dataset Status",
                    "run-dataset-status.yml",
                )
                st.success(msg)
            else:
                st.error(msg)

        render_workflow_status("dataset_status")

        if status is None:
            st.warning(
                f"Dataset status artifact not found at `{DATASET_STATUS_PATH}`."
            )
            return

        row = status.iloc[0]

        health_df = pd.DataFrame(
            [
                {
                    "Rows": row.get("row_count"),
                    "Columns": row.get("column_count"),
                    "Events": row.get("unique_events"),
                    "Fighters": row.get("unique_fighters"),
                }
            ]
        )

        st.dataframe(
            health_df,
            hide_index=True,
            use_container_width=True,
        )

        detail_df = pd.DataFrame(
            [
                {
                    "Latest Fight Date":
                        row.get("latest_fight_date"),
                    "Duplicate Fights":
                        row.get("duplicate_fight_count"),
                    "Invalid Dates":
                        row.get("invalid_date_count"),
                    "Missing Results":
                        row.get("missing_result_count"),
                }
            ]
        )

        st.dataframe(
            detail_df,
            hide_index=True,
            use_container_width=True,
        )
