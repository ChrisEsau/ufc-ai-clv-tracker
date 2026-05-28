from pathlib import Path

import pandas as pd
import streamlit as st


APPEND_PRECHECK_PATH = "ufc_append_precheck.parquet"


def safe_read_parquet(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)

    except Exception as e:
        st.warning(f"Could not read `{path.name}`: {e}")
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

    st.subheader("Append Validation Gate")

    append_ready, precheck = get_append_precheck()

    if precheck is None:

        st.warning("No append precheck artifact found yet.")

        st.button(
            "⚠️ Append / Ingest Staged Data",
            disabled=True,
            type="secondary",
            use_container_width=True,
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
            "Staged data is ready to ingest."
        )
    else:
        st.error(
            "❌ Append blocked. "
            "Validation failures detected."
        )

    metric_cols = st.columns(3)

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

    metric_cols[0].metric(
        "Staged Rows",
        staged_rows,
    )

    metric_cols[1].metric(
        "Master Rows",
        master_rows,
    )

    metric_cols[2].metric(
        "Failed Checks",
        failed_checks,
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

    st.dataframe(
        precheck[display_cols],
        use_container_width=True,
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
        )

    button_type = (
        "primary"
        if append_ready
        else "secondary"
    )

    st.button(
        "⚠️ Append / Ingest Staged Data",
        disabled=not append_ready,
        type=button_type,
        use_container_width=True,
    )