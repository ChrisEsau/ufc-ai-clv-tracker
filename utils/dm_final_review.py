from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import (
    APPEND_AUDIT_PATH,
    APPEND_PRECHECK_PATH,
    FIGHT_DETAIL_SCRAPE_AUDIT_PATH,
    FIGHT_SCRAPE_AUDIT_PATH,
    FIGHTER_PROFILE_SCRAPE_AUDIT_PATH,
    MASTER_COLUMN_VALIDATION_PATH,
    STAGED_DERIVED_STATS_AUDIT_PATH,
    STAGED_FINAL_REVIEW_PATH,
    STAGED_FIGHT_DETAILS_PATH,
    STAGED_FIGHT_ROWS_PATH,
    STAGED_FIGHTER_PROFILES_PATH,
    STAGED_MASTER_MAPPING_AUDIT_PATH,
    STAGED_MASTER_ROWS_ENRICHED_PATH,
    STAGED_MASTER_ROWS_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
)
from utils.github_actions import trigger_workflow
from utils.dm_workflow_status import (
    remember_launched_workflow,
    render_workflow_status,
)


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

INGESTION_ARTIFACTS = [
    ("Fight Rows", STAGED_FIGHT_ROWS_PATH),
    ("Fight Details", STAGED_FIGHT_DETAILS_PATH),
    ("Mapped Master Rows", STAGED_MASTER_ROWS_PATH),
    ("Derived Master Rows", STAGED_MASTER_ROWS_ENRICHED_PATH),
    ("Fighter Profiles", STAGED_FIGHTER_PROFILES_PATH),
    ("Profiled Master Rows", STAGED_MASTER_ROWS_PROFILED_PATH),
    ("Fight Scrape Audit", FIGHT_SCRAPE_AUDIT_PATH),
    ("Fight Detail Audit", FIGHT_DETAIL_SCRAPE_AUDIT_PATH),
    ("Mapping Audit", STAGED_MASTER_MAPPING_AUDIT_PATH),
    ("Derived Stats Audit", STAGED_DERIVED_STATS_AUDIT_PATH),
    ("Fighter Profile Audit", FIGHTER_PROFILE_SCRAPE_AUDIT_PATH),
    ("Column Validation", MASTER_COLUMN_VALIDATION_PATH),
    ("Append Precheck", APPEND_PRECHECK_PATH),
    ("Final Review", STAGED_FINAL_REVIEW_PATH),
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


def artifact_modified(path):
    path = Path(path)

    if not path.exists():
        return None

    return pd.Timestamp(path.stat().st_mtime, unit="s")


def get_final_review_status():
    final_review = safe_read_parquet(STAGED_FINAL_REVIEW_PATH)

    if final_review is None or final_review.empty:
        return False, None

    if "final_review_pass" not in final_review.columns:
        return False, final_review

    final_review_pass = bool(final_review["final_review_pass"].iloc[0])

    return final_review_pass, final_review


def get_append_precheck_status():
    precheck = safe_read_parquet(APPEND_PRECHECK_PATH)

    if precheck is None or precheck.empty:
        return False, None

    if "append_ready" not in precheck.columns:
        return False, precheck

    append_ready = bool(precheck["append_ready"].iloc[0])

    return append_ready, precheck


def final_review_failed_checks(final_review):
    if final_review is None or "status" not in final_review.columns:
        return pd.DataFrame()

    return final_review[final_review["status"] == "fail"]


def render_artifact_summary():
    rows = []

    for label, path in INGESTION_ARTIFACTS:
        path = Path(path)
        df = safe_read_parquet(path)

        rows.append(
            {
                "Stage": label,
                "Path": str(path),
                "Exists": df is not None,
                "Rows": len(df) if df is not None else None,
                "Columns": len(df.columns) if df is not None else None,
                "Last Modified": artifact_modified(path),
            }
        )

    st.markdown("#### Ingestion Output Summary")
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def render_staged_event_summary(staged):
    st.markdown("#### Current Staged Event")

    if staged is None or staged.empty:
        st.warning("No profiled staged rows are available for review.")
        return

    event_names = sorted(staged.get("event_name", pd.Series(dtype=object)).dropna().astype(str).unique())
    event_ids = sorted(staged.get("event_id", pd.Series(dtype=object)).dropna().astype(str).unique())
    dates = sorted(staged.get("date", pd.Series(dtype=object)).dropna().astype(str).unique())
    fighter_count = len(
        pd.concat(
            [
                staged.get("r_id", pd.Series(dtype=object)),
                staged.get("b_id", pd.Series(dtype=object)),
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
    )

    summary = pd.DataFrame(
        [
            {
                "Event Name": ", ".join(event_names[:3]),
                "Event ID": ", ".join(event_ids[:3]),
                "Date": ", ".join(dates[:3]),
                "Staged Fights": len(staged),
                "Unique Fighters": fighter_count,
                "Profiled Rows Artifact": str(STAGED_MASTER_ROWS_PROFILED_PATH),
            }
        ]
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_staged_row_preview(staged=None):
    if staged is None:
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
        staged[display_cols].head(50),
        use_container_width=True,
        hide_index=True,
    )


def render_precheck_summary(append_ready, precheck):
    st.markdown("#### Append Precheck")

    if precheck is None:
        st.warning(
            f"No append precheck artifact found at `{APPEND_PRECHECK_PATH}`. "
            "Run single-event ingestion or Append Precheck + Final Review."
        )
        return pd.DataFrame()

    failed = (
        precheck[precheck["status"] == "fail"]
        if "status" in precheck.columns
        else pd.DataFrame()
    )

    if append_ready:
        st.success("✅ Append precheck passed.")
    else:
        st.error("❌ Append precheck blocked append.")

    summary = pd.DataFrame(
        [
            {
                "Append Ready": append_ready,
                "Staged Rows": int(precheck["staged_rows"].iloc[0])
                if "staged_rows" in precheck.columns
                else None,
                "Master Rows": int(precheck["master_rows"].iloc[0])
                if "master_rows" in precheck.columns
                else None,
                "Failed Checks": len(failed),
                "Artifact": str(APPEND_PRECHECK_PATH),
            }
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    display_cols = [
        col for col in ["check_name", "severity", "status", "failure_count", "details"]
        if col in precheck.columns
    ]

    with st.expander("Append Precheck Checks", expanded=False):
        st.dataframe(precheck[display_cols], use_container_width=True, hide_index=True)

    if not failed.empty:
        with st.expander("Failed Append Precheck Checks", expanded=True):
            st.dataframe(failed[display_cols], use_container_width=True, hide_index=True)

    return failed


def render_final_review_summary(final_review_pass, final_review):
    st.markdown("#### Final Review")

    if final_review is None:
        st.warning(
            f"No final review artifact found at `{STAGED_FINAL_REVIEW_PATH}`. "
            "Run single-event ingestion or Append Precheck + Final Review."
        )
        return pd.DataFrame()

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
        st.success("✅ Final review passed. Staged data is semantically ready.")
    else:
        st.error("❌ Final review blocked append. Review failed checks below.")

    summary = pd.DataFrame(
        [
            {
                "Final Review Pass": final_review_pass,
                "Staged Rows": int(final_review["staged_rows"].iloc[0])
                if "staged_rows" in final_review.columns
                else None,
                "Master Rows": int(final_review["master_rows"].iloc[0])
                if "master_rows" in final_review.columns
                else None,
                "Blocking Failures": len(blocking_failed),
                "Warning Failures": len(warning_failed),
                "Run Timestamp": final_review["run_timestamp"].iloc[0]
                if "run_timestamp" in final_review.columns
                else None,
                "Artifact": str(STAGED_FINAL_REVIEW_PATH),
            }
        ]
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)

    display_cols = [
        col for col in ["check_name", "severity", "status", "failure_count", "details"]
        if col in final_review.columns
    ]

    with st.expander("Final Review Checks", expanded=False):
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

    return failed


def render_append_decision(append_ready, final_review_pass):
    st.markdown("#### Append Decision")

    append_allowed = bool(append_ready and final_review_pass)
    status_label = "✅ READY" if append_allowed else "❌ BLOCKED"

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Append Allowed": status_label,
                    "Precheck": "✅ PASS" if append_ready else "❌ BLOCKED",
                    "Final Review": "✅ PASS" if final_review_pass else "❌ BLOCKED",
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not append_allowed:
        st.info(
            "Append remains disabled until both append precheck and final review pass."
        )

    confirmed = st.checkbox(
        "I reviewed the staged rows and understand this will append them to master.",
        disabled=not append_allowed,
        key="append_to_master_human_confirmation",
    )

    if st.button(
        "⚠️ Append To Master",
        disabled=not (append_allowed and confirmed),
        type="primary" if append_allowed and confirmed else "secondary",
        use_container_width=True,
        key="append_to_master_final",
    ):
        ok, msg = trigger_workflow("run-append-staged-to-master.yml")

        if ok:
            remember_launched_workflow(
                "append_to_master",
                "Append Staged Rows To Master",
                "run-append-staged-to-master.yml",
            )
            st.success(msg)
        else:
            st.error(msg)

    render_workflow_status("append_to_master")

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


def render_next_action(staged, append_ready, final_review_pass):
    if staged is None or staged.empty:
        st.info("Next step: go to Event Discovery and ingest a selected event.")
    elif not append_ready:
        st.info("Next step: resolve append precheck failures, then rerun review.")
    elif not final_review_pass:
        st.info("Next step: resolve final review failures, then rerun review.")
    else:
        st.success(
            "Next step: review staged rows below. If everything looks correct, "
            "confirm and append to master."
        )


def render_final_review():
    with st.expander("🧾 Final Staged Review", expanded=False):
        st.caption(
            "Review staged ingestion outputs, append precheck, final review, and "
            "human append approval in one place."
        )

        if st.button(
            "Run Append Precheck + Final Review",
            use_container_width=True,
            key="run_staged_final_review",
        ):
            ok, msg = trigger_workflow("run-append-precheck-validation.yml")

            if ok:
                remember_launched_workflow(
                    "append_precheck_final_review",
                    "Append Precheck + Final Review",
                    "run-append-precheck-validation.yml",
                )
                st.success(msg)
            else:
                st.error(msg)

        render_workflow_status("append_precheck_final_review")

        staged = safe_read_parquet(STAGED_MASTER_ROWS_PROFILED_PATH)
        append_ready, precheck = get_append_precheck_status()
        final_review_pass, final_review = get_final_review_status()

        render_next_action(staged, append_ready, final_review_pass)
        render_artifact_summary()
        render_staged_event_summary(staged)
        render_staged_row_preview(staged)
        render_precheck_summary(append_ready, precheck)
        render_final_review_summary(final_review_pass, final_review)
        render_append_decision(append_ready, final_review_pass)
