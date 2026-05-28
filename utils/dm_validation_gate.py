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
            use_container_width=True,
        )
        return

    failed = precheck[
        precheck["status"] == "fail"
    ] if "status" in precheck.columns else pd.DataFrame()

    if append_ready:
        st.success("✅ Append validation passed. Staged data is ready to ingest.")
    else:
        st.error("❌ Append blocked. Validation failures detected.")

    display_cols = [
        c for c in [
            "check_name",
            "status",
            "failure_count",
            "details",
            "staged_rows",
            "master_rows",
            "append_ready",
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

    st.button(
        "⚠️ Append / Ingest Staged Data",
        disabled=not append_ready,
        type="primary",
        use_container_width=True,
    )