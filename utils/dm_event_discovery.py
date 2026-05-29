from pathlib import Path

import pandas as pd
import streamlit as st

from utils.github_actions import trigger_workflow


MISSING_EVENTS_PATH = "ufc_missing_events.parquet"


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
                    "Artifact": MISSING_EVENTS_PATH,
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
        else:
            st.dataframe(
                missing_events.head(20),
                use_container_width=True,
                hide_index=True,
            )