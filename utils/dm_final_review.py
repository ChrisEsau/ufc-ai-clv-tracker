from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import (
    STAGED_FINAL_REVIEW_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
)
from utils.github_actions import trigger_workflow


REVIEW_SUMMARY_COLUMNS = [
    "event_name",
    "date",
    "fight_id",
    "r_name",
    "r_id",
    "b_name",
    "b_id",
    "winner",
    "winner_id",
    "method",
    "finish_round",
    "match_time_sec",
]


def safe_read_parquet(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)

    except Exception as e:
        st.warning(f"Could not read `{path}`: {e}")
        return None


def get_final_review_status():
    final_review = safe_read_parquet(STAGED_FINAL_REVIEW_PATH)

    if final_review is None or final_review.empty:
        return False, None

    if "final_review_pass" not in final_review.columns:
        return False, final_review

    final_review_pass = bool(final_review["final_review_pass"].iloc[0])

    return final_review_pass, final_review


def final_review_failed_checks(final_review):
    if final_review is None or "status" not in final_review.columns:
        return pd.DataFrame()

    return final_review[final_review["status"] == "fail"]


def render_staged_row_preview():
    staged = safe_read_parquet(STAGED_MASTER_ROWS_PROFILED_PATH)

    st.markdown("#### Staged Row Preview")

    if staged is None or staged.empty:
        st.warning(
            f"No profiled staged rows found at `{STAGED_MASTER_ROWS_PROFILED_PATH}`."
        )
        return

    display_cols = [
        col for col in REVIEW_SUMMARY_COLUMNS
        if col in staged.columns
    ]

    st.caption(f"Source: `{STAGED_MASTER_ROWS_PROFILED_PATH}`")
    st.dataframe(
        staged[display_cols].head(25),
        use_container_width=True,
        hide_index=True,
    )


def render_final_review():
    st.markdown("---")
    st.subheader("🧾 Final Staged Review")
    st.caption(
        "Human-readable semantic review before append. The append button remains "
        "disabled until both append precheck and final review pass."
    )

    if st.button(
        "Run Final Staged Review",
        use_container_width=True,
        key="run_staged_final_review",
    ):
        ok, msg = trigger_workflow("run-staged-final-review.yml")

        if ok:
            st.success(msg)
        else:
            st.error(msg)

    final_review_pass, final_review = get_final_review_status()

    if final_review is None:
        st.warning(
            f"No final review artifact found at `{STAGED_FINAL_REVIEW_PATH}`."
        )
        render_staged_row_preview()
        return

    failed = final_review_failed_checks(final_review)
    blocking_failed = (
        failed[failed["severity"] == "block"]
        if "severity" in failed.columns
        else failed
    )
    warning_failed = (
        failed[failed["severity"] == "warning"]
        if "severity" in failed.columns
        else pd.DataFrame()
    )

    if final_review_pass:
        st.success("✅ Final review passed. Staged row is semantically ready.")
    else:
        st.error("❌ Final review blocked append. Review failed checks below.")

    staged_rows = (
        int(final_review["staged_rows"].iloc[0])
        if "staged_rows" in final_review.columns
        else 0
    )
    master_rows = (
        int(final_review["master_rows"].iloc[0])
        if "master_rows" in final_review.columns
        else 0
    )
    run_timestamp = (
        final_review["run_timestamp"].iloc[0]
        if "run_timestamp" in final_review.columns
        else None
    )

    summary = pd.DataFrame(
        [
            {
                "Final Review Pass": final_review_pass,
                "Staged Rows": staged_rows,
                "Master Rows": master_rows,
                "Blocking Failures": len(blocking_failed),
                "Warning Failures": len(warning_failed),
                "Run Timestamp": run_timestamp,
                "Artifact": str(STAGED_FINAL_REVIEW_PATH),
            }
        ]
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    render_staged_row_preview()

    display_cols = [
        col for col in [
            "check_name",
            "severity",
            "status",
            "failure_count",
            "details",
        ]
        if col in final_review.columns
    ]

    with st.expander("Final Review Checks", expanded=True):
        st.dataframe(
            final_review[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    if not failed.empty:
        with st.expander("Failed Final Review Checks", expanded=True):
            st.dataframe(
                failed[display_cols],
                use_container_width=True,
                hide_index=True,
            )
