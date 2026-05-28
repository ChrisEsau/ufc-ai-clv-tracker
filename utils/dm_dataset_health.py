from pathlib import Path

import pandas as pd
import streamlit as st


def safe_read_parquet(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def render_dataset_health():

    status = safe_read_parquet(
        "ufc_dataset_status.parquet"
    )

    with st.expander(
        "📊 Dataset Health",
        expanded=False,
    ):

        if status is None:
            st.warning(
                "Dataset status artifact not found."
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