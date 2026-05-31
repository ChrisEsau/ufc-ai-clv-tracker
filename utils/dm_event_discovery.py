from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import MISSING_EVENTS_PATH
from utils.github_actions import trigger_workflow


def safe_read_parquet(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def render_event_discovery():

    with st.expander(
        "📅 Event Discovery",
        expanded=False,
    ):

        st.caption(
            "Checks UFCStats against the local master dataset "
            "and identifies missing completed events."
        )

        if st.button(
            "Run Event Check",
            use_container_width=True,
            key="event_discovery_run_event_check",
        ):
            ok, msg = trigger_workflow(
                "run-ufcstats-event-check.yml"
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)

        missing_events = safe_read_parquet(
            MISSING_EVENTS_PATH
        )

        if missing_events is None:
            st.warning(
                "Missing events artifact not found. "
                "Run Event Check first."
            )
            return

        missing_count = len(missing_events)

        summary_df = pd.DataFrame(
            [
                {
                    "Missing Events": missing_count,
                    "Artifact": str(MISSING_EVENTS_PATH),
                }
            ]
        )

        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
        )

        preview_cols = [
            c for c in [
                "ufcstats_event_name",
                "ufcstats_event_date",
                "ufcstats_event_id",
                "ufcstats_event_url",
            ]
            if c in missing_events.columns
        ]

        if preview_cols:
            st.markdown("#### Missing Events Preview")

            st.dataframe(
                missing_events[preview_cols].head(20),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Ingest Selected Event")

        if "ufcstats_event_id" not in missing_events.columns:
            st.error(
                "Missing events artifact does not contain "
                "`ufcstats_event_id`."
            )
            return

        event_options = (
            missing_events[
                [
                    "ufcstats_event_name",
                    "ufcstats_event_date",
                    "ufcstats_event_id",
                ]
            ]
            .dropna(subset=["ufcstats_event_id"])
            .copy()
        )

        event_options["label"] = (
            event_options["ufcstats_event_date"].astype(str)
            + " | "
            + event_options["ufcstats_event_name"].astype(str)
            + " | "
            + event_options["ufcstats_event_id"].astype(str)
        )

        selected_label = st.selectbox(
            "Select missing event",
            options=event_options["label"].tolist(),
            key="event_discovery_selected_event",
        )

        selected_row = event_options[
            event_options["label"] == selected_label
        ].iloc[0]

        selected_event_id = selected_row["ufcstats_event_id"]

        st.code(str(selected_event_id), language="text")

        if st.button(
            "Ingest Selected Event",
            use_container_width=True,
            key="event_discovery_ingest_selected_event",
        ):
            ok, msg = trigger_workflow(
                "dm-ingest-single-event.yml",
                inputs={
                    "event_id": str(selected_event_id),
                },
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)