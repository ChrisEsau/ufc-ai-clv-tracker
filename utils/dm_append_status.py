from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import (
    APPEND_AUDIT_PATH,
    APPEND_PRECHECK_PATH,
    STAGED_FINAL_REVIEW_PATH,
)
from utils.github_actions import trigger_workflow
from utils.dm_final_review import get_final_review_status


def safe_read_parquet(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def render_append_status():

    st.markdown("---")

    st.subheader("🚀 Append Status")

    precheck = safe_read_parquet(APPEND_PRECHECK_PATH)

    append_ready = False
    final_review_pass, final_review = get_final_review_status()
    append_allowed = False
    staged_rows = 0
    failed_checks = 0
    master_rows = 0

    if precheck is not None and not precheck.empty:

        if "append_ready" in precheck.columns:
            append_ready = bool(precheck["append_ready"].iloc[0])

        if "staged_rows" in precheck.columns:
            staged_rows = int(precheck["staged_rows"].iloc[0])

        if "master_rows" in precheck.columns:
            master_rows = int(precheck["master_rows"].iloc[0])

        if "status" in precheck.columns:
            failed_checks = len(
                precheck[precheck["status"] == "fail"]
            )

    append_allowed = append_ready and final_review_pass

    status_label = "✅ READY" if append_allowed else "❌ BLOCKED"
    precheck_label = "✅ PASS" if append_ready else "❌ BLOCKED"
    final_review_label = "✅ PASS" if final_review_pass else "❌ BLOCKED"

    summary_df = pd.DataFrame(
        [
            {
                "Append Allowed": status_label,
                "Precheck": precheck_label,
                "Final Review": final_review_label,
                "Staged Rows": staged_rows,
                "Failed Precheck Checks": failed_checks,
                "Master Rows": master_rows,
                "Precheck Artifact": str(APPEND_PRECHECK_PATH),
                "Final Review Artifact": str(STAGED_FINAL_REVIEW_PATH),
            }
        ]
    )

    st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True,
    )

    button_type = "primary" if append_allowed else "secondary"

    if final_review is None:
        st.warning(
            f"No final review artifact found at `{STAGED_FINAL_REVIEW_PATH}`. "
            "Run Final Staged Review before append."
        )

    if st.button(
        "⚠️ Append To Master",
        disabled=not append_allowed,
        type=button_type,
        use_container_width=True,
        key="append_to_master_final",
    ):
        ok, msg = trigger_workflow(
            "run-append-staged-to-master.yml"
        )

        if ok:
            st.success(msg)
        else:
            st.error(msg)

    append_audit = safe_read_parquet(APPEND_AUDIT_PATH)

    with st.expander("Latest Append Audit", expanded=False):

        if append_audit is None:
            st.info(f"No append audit artifact found at `{APPEND_AUDIT_PATH}`.")
        else:
            st.dataframe(
                append_audit,
                hide_index=True,
                use_container_width=True,
            )
