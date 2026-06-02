from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import APPEND_PRECHECK_PATH
from utils.github_actions import trigger_workflow


def safe_read_parquet(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)

    except Exception as e:
        st.warning(f"Could not read `{path}`: {e}")
        return None


def get_append_precheck():
    precheck = safe_read_parquet(APPEND_PRECHECK_PATH)

    if precheck is None or precheck.empty:
        return False, None

    if "append_ready" not in precheck.columns:
        return False, precheck

    append_ready = bool(precheck["append_ready"].iloc[0])

    return append_ready, precheck


def render_validation_gate():

    with st.expander(
        "✅ Validation Gate",
        expanded=False,
    ):

        st.caption(
            "Validates staged data before append. "
            "Append execution lives in the final Append Status section."
        )

        if st.button(
            "Run Column Validation",
            use_container_width=True,
            key="run_column_validation",
        ):
            ok, msg = trigger_workflow(
                "run-master-column-validation.yml"
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)

        if st.button(
            "Run Append Precheck",
            use_container_width=True,
            key="run_append_precheck",
        ):
            ok, msg = trigger_workflow(
                "run-append-precheck-validation.yml"
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)

        append_ready, precheck = get_append_precheck()

        if precheck is None:

            st.warning(
                f"No append precheck artifact found at `{APPEND_PRECHECK_PATH}`."
            )
            return

        if "status" in precheck.columns:
            failed = precheck[
                precheck["status"] == "fail"
            ]
        else:
            failed = pd.DataFrame()

        if append_ready:
            st.success(
                "✅ Append validation passed. "
                "Staged data is ready for append."
            )
        else:
            st.error(
                "❌ Append blocked. "
                "Validation failures detected."
            )

        staged_rows = (
            int(precheck["staged_rows"].iloc[0])
            if "staged_rows" in precheck.columns
            else 0
        )

        master_rows = (
            int(precheck["master_rows"].iloc[0])
            if "master_rows" in precheck.columns
            else 0
        )

        failed_checks = len(failed)

        summary_df = pd.DataFrame(
            [
                {
                    "Append Ready": append_ready,
                    "Staged Rows": staged_rows,
                    "Master Rows": master_rows,
                    "Failed Checks": failed_checks,
                    "Artifact": str(APPEND_PRECHECK_PATH),
                }
            ]
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        display_cols = [
            c for c in [
                "check_name",
                "status",
                "failure_count",
                "details",
            ]
            if c in precheck.columns
        ]

        st.markdown("#### Validation Details")

        st.dataframe(
            precheck[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        if not failed.empty:

            st.markdown("#### Failed Checks")

            failed_cols = [
                c for c in [
                    "check_name",
                    "failure_count",
                    "details",
                ]
                if c in failed.columns
            ]

            st.dataframe(
                failed[failed_cols],
                use_container_width=True,
                hide_index=True,
            )
