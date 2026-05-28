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


def render_append_status():

    precheck = safe_read_parquet(
        "ufc_append_precheck.parquet"
    )

    append_ready = False
    staged_rows = 0
    failed_checks = 0
    master_rows = 0

    if precheck is not None:

        if "append_ready" in precheck.columns:
            append_ready = bool(
                precheck["append_ready"].iloc[0]
            )

        if "staged_rows" in precheck.columns:
            staged_rows = int(
                precheck["staged_rows"].iloc[0]
            )

        if "master_rows" in precheck.columns:
            master_rows = int(
                precheck["master_rows"].iloc[0]
            )

        if "status" in precheck.columns:
            failed_checks = len(
                precheck[
                    precheck["status"] == "fail"
                ]
            )

    st.subheader("🚀 Append Status")

    status_icon = (
        "✅ READY"
        if append_ready
        else "❌ BLOCKED"
    )

    status_df = pd.DataFrame(
        [
            {
                "Status": status_icon,
                "Staged Rows": staged_rows,
                "Failed Checks": failed_checks,
                "Master Rows": master_rows,
            }
        ]
    )

    st.dataframe(
        status_df,
        hide_index=True,
        use_container_width=True,
    )