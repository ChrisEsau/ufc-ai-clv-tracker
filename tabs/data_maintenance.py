from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.booleans import coerce_bool
from pipeline.common.paths import (
    APPEND_AUDIT_PATH,
    APPEND_PRECHECK_PATH,
    DATASET_STATUS_PATH,
    FIGHT_DETAIL_SCRAPE_AUDIT_PATH,
    FIGHT_SCRAPE_AUDIT_PATH,
    FIGHTER_PROFILE_SCRAPE_AUDIT_PATH,
    MASTER_COLUMN_VALIDATION_PATH,
    MISSING_EVENTS_PATH,
    STAGED_DERIVED_STATS_AUDIT_PATH,
    STAGED_FIGHT_DETAILS_PATH,
    STAGED_FIGHT_ROWS_PATH,
    STAGED_FIGHTER_PROFILES_PATH,
    STAGED_FINAL_REVIEW_PATH,
    STAGED_MASTER_MAPPING_AUDIT_PATH,
    STAGED_MASTER_ROWS_ENRICHED_PATH,
    STAGED_MASTER_ROWS_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
    UFCSTATS_EVENT_CHECK_PATH,
)
from utils.dm_workflow_status import remember_launched_workflow, render_workflow_status
from utils.github_actions import trigger_workflow
from utils.ui.badges import status_badge
from utils.ui.cards import metric_card
from utils.ui.sections import page_header, section_heading


PIPELINE_ARTIFACTS = [
    ("Event Check", UFCSTATS_EVENT_CHECK_PATH),
    ("Fight Rows", STAGED_FIGHT_ROWS_PATH),
    ("Fight Details", STAGED_FIGHT_DETAILS_PATH),
    ("Mapped Rows", STAGED_MASTER_ROWS_PATH),
    ("Derived Rows", STAGED_MASTER_ROWS_ENRICHED_PATH),
    ("Fighter Profiles", STAGED_FIGHTER_PROFILES_PATH),
    ("Profiled Rows", STAGED_MASTER_ROWS_PROFILED_PATH),
    ("Column Validation", MASTER_COLUMN_VALIDATION_PATH),
    ("Append Precheck", APPEND_PRECHECK_PATH),
    ("Final Review", STAGED_FINAL_REVIEW_PATH),
]

AUDIT_ARTIFACTS = [
    ("Dataset Status", DATASET_STATUS_PATH),
    ("Event Check", UFCSTATS_EVENT_CHECK_PATH),
    ("Fight Scrape Audit", FIGHT_SCRAPE_AUDIT_PATH),
    ("Fight Detail Audit", FIGHT_DETAIL_SCRAPE_AUDIT_PATH),
    ("Mapping Audit", STAGED_MASTER_MAPPING_AUDIT_PATH),
    ("Derived Stats Audit", STAGED_DERIVED_STATS_AUDIT_PATH),
    ("Fighter Profile Audit", FIGHTER_PROFILE_SCRAPE_AUDIT_PATH),
    ("Column Validation", MASTER_COLUMN_VALIDATION_PATH),
    ("Append Precheck", APPEND_PRECHECK_PATH),
    ("Final Review", STAGED_FINAL_REVIEW_PATH),
    ("Append Audit", APPEND_AUDIT_PATH),
]

PREVIEW_COLUMNS = [
    "event_name",
    "date",
    "fight_id",
    "r_name",
    "b_name",
    "winner",
    "method",
    "finish_round",
    "match_time_sec",
]


def _safe_read(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        st.warning(f"Could not read `{path}`: {exc}")
        return pd.DataFrame()


def _modified(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.Timestamp(path.stat().st_mtime, unit="s")


def _append_ready(precheck):
    if precheck.empty or "append_ready" not in precheck.columns:
        return False
    return coerce_bool(precheck["append_ready"].iloc[0])


def _final_review_pass(final_review):
    if final_review.empty:
        return False
    if "final_review_pass" in final_review.columns:
        return coerce_bool(final_review["final_review_pass"].iloc[0])
    if "check_passed" in final_review.columns:
        return all(coerce_bool(value) for value in final_review["check_passed"])
    return False


def _failed_checks(df):
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    return df[df["status"].astype(str).str.lower() == "fail"].copy()


def _format_timestamp(value):
    if value is None or pd.isna(value):
        return "—"
    try:
        return pd.to_datetime(value).strftime("%b %-d, %Y %I:%M %p")
    except Exception:
        return str(value)


def _artifact_rows(artifacts):
    rows = []
    for label, path in artifacts:
        df = _safe_read(path)
        exists = Path(path).exists()
        rows.append(
            {
                "Stage": label,
                "Status": "Ready" if exists else "Missing",
                "Rows": len(df) if exists else 0,
                "Columns": len(df.columns) if exists and not df.empty else 0,
                "Last Updated": _format_timestamp(_modified(path)),
                "Path": str(path),
            }
        )
    return pd.DataFrame(rows)


def _render_top_kpis(status, missing_events, staged, precheck, final_review):
    row = status.iloc[0] if not status.empty else {}
    append_ready = _append_ready(precheck)
    final_pass = _final_review_pass(final_review)
    append_allowed = append_ready and final_pass

    cols = st.columns(6)
    with cols[0]:
        metric_card("Total Events", row.get("unique_events", "—"), caption="Master dataset", status="info")
    with cols[1]:
        metric_card("Total Fights", row.get("row_count", "—"), caption="Master rows", status="info")
    with cols[2]:
        metric_card(
            "Last Status Run",
            _format_timestamp(row.get("run_timestamp")),
            caption="Dataset health artifact",
            status="neutral",
        )
    with cols[3]:
        column_count = row.get("column_count")
        schema_ok = column_count == 128
        metric_card(
            "Schema Health",
            "128/128" if schema_ok else column_count or "—",
            caption="Locked master schema",
            status="success" if schema_ok else "warning",
        )
    with cols[4]:
        metric_card(
            "Pending Events",
            len(missing_events) if not missing_events.empty else 0,
            caption="Missing completed events",
            status="warning" if not missing_events.empty else "success",
        )
    with cols[5]:
        metric_card(
            "Append Gate",
            "Ready" if append_allowed else "Blocked",
            caption=f"{len(staged) if not staged.empty else 0} staged rows",
            status="success" if append_allowed else "danger",
        )


def _render_dataset_status_panel(status):
    section_heading("Dataset Status Overview", "Master dataset health and schema status.")
    with st.container(border=True):
        button_col, status_col = st.columns([1, 2])
        with button_col:
            if st.button("Run Dataset Status", use_container_width=True, key="dm_run_dataset_status_panel"):
                ok, msg = trigger_workflow("run-dataset-status.yml")
                if ok:
                    remember_launched_workflow("dataset_status", "Run Dataset Status", "run-dataset-status.yml")
                    st.success(msg)
                else:
                    st.error(msg)
        with status_col:
            if status.empty:
                st.warning(f"Dataset status artifact not found at `{DATASET_STATUS_PATH}`.")
            else:
                row = status.iloc[0]
                st.markdown(
                    " ".join(
                        [
                            status_badge("Healthy" if row.get("column_count") == 128 else "Review", "success" if row.get("column_count") == 128 else "warning"),
                            f"Latest fight: `{row.get('latest_fight_date', '—')}`",
                            f"Rows: `{row.get('row_count', '—')}`",
                            f"Columns: `{row.get('column_count', '—')}`",
                        ]
                    ),
                    unsafe_allow_html=True,
                )

        render_workflow_status("dataset_status")

        if not status.empty:
            row = status.iloc[0]
            cols = st.columns(4)
            metrics = [
                ("Unique Fighters", row.get("unique_fighters", "—"), "info"),
                ("Duplicate Fights", row.get("duplicate_fight_count", "—"), "warning" if row.get("duplicate_fight_count", 0) else "success"),
                ("Invalid Dates", row.get("invalid_date_count", "—"), "warning" if row.get("invalid_date_count", 0) else "success"),
                ("Missing Results", row.get("missing_result_count", "—"), "warning" if row.get("missing_result_count", 0) else "success"),
            ]
            for col, (label, value, status_name) in zip(cols, metrics):
                with col:
                    metric_card(label, value, status=status_name)

            with st.expander("Dataset Status Artifact", expanded=False):
                st.dataframe(status, use_container_width=True, hide_index=True)


def _render_event_discovery_panel(missing_events):
    section_heading("Event Discovery", "Discover completed UFCStats events and stage one selected event for full ingestion.")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Events Discovered", len(missing_events) if not missing_events.empty else 0, caption="Missing from master", status="warning" if not missing_events.empty else "success")
        with c2:
            metric_card("Ingestion Mode", "Full", caption="Full selected-event ingestion", status="info")
        with c3:
            metric_card("Event Check", "Ready", caption=str(UFCSTATS_EVENT_CHECK_PATH), status="neutral")

        if st.button("Discover New Events", use_container_width=True, key="dm_run_event_check_panel"):
            ok, msg = trigger_workflow("run-ufcstats-event-check.yml")
            if ok:
                remember_launched_workflow("event_check", "Run Event Check", "run-ufcstats-event-check.yml")
                st.success(msg)
            else:
                st.error(msg)

        render_workflow_status("event_check")

        if missing_events.empty:
            st.info("No missing event artifact rows are available. Run Event Check to refresh discovery.")
            return

        preview_cols = [
            col
            for col in ["ufcstats_event_name", "ufcstats_event_date", "ufcstats_event_id"]
            if col in missing_events.columns
        ]
        if preview_cols:
            st.dataframe(missing_events[preview_cols].head(20), use_container_width=True, hide_index=True)

        if "ufcstats_event_id" not in missing_events.columns:
            st.error("Missing events artifact does not contain `ufcstats_event_id`.")
            return

        option_cols = ["ufcstats_event_name", "ufcstats_event_date", "ufcstats_event_id"]
        if not set(option_cols).issubset(missing_events.columns):
            st.error("Missing events artifact does not contain the expected event selection columns.")
            return

        event_options = missing_events[option_cols].dropna(subset=["ufcstats_event_id"]).copy()
        if event_options.empty:
            st.info("No selectable missing events are available.")
            return

        event_options["label"] = (
            event_options["ufcstats_event_date"].astype(str)
            + " | "
            + event_options["ufcstats_event_name"].astype(str)
            + " | "
            + event_options["ufcstats_event_id"].astype(str)
        )
        selected_label = st.selectbox(
            "Select missing event for full ingestion",
            options=event_options["label"].tolist(),
            key="event_discovery_selected_event",
        )
        selected_row = event_options[event_options["label"] == selected_label].iloc[0]
        selected_event_id = str(selected_row["ufcstats_event_id"])
        st.caption(f"Selected event ID: `{selected_event_id}`")

        if st.button("Ingest Selected Event", use_container_width=True, key="event_discovery_ingest_selected_event"):
            ok, msg = trigger_workflow("dm-ingest-single-event.yml", inputs={"event_id": selected_event_id})
            if ok:
                remember_launched_workflow(
                    "single_event_ingest",
                    "DM - Ingest Single Event",
                    "dm-ingest-single-event.yml",
                    inputs={"event_id": selected_event_id},
                )
                st.success(msg)
            else:
                st.error(msg)

        render_workflow_status("single_event_ingest")


def _render_pipeline_status_panel(staged):
    section_heading("Ingestion Pipeline Status", "Artifact-backed progress through discovery, scraping, mapping, enrichment, and review.")
    with st.container(border=True):
        artifact_df = _artifact_rows(PIPELINE_ARTIFACTS)
        ready_count = int((artifact_df["Status"] == "Ready").sum())
        st.progress(ready_count / len(artifact_df))
        st.caption(f"{ready_count} of {len(artifact_df)} pipeline artifacts are available.")

        badge_cols = st.columns(5)
        for idx, row in artifact_df.iterrows():
            with badge_cols[idx % len(badge_cols)]:
                st.markdown(
                    status_badge(row["Stage"], "success" if row["Status"] == "Ready" else "neutral"),
                    unsafe_allow_html=True,
                )

        if not staged.empty:
            event_names = sorted(staged.get("event_name", pd.Series(dtype=object)).dropna().astype(str).unique())
            event_ids = sorted(staged.get("event_id", pd.Series(dtype=object)).dropna().astype(str).unique())
            st.markdown("#### Current Staged Event")
            summary_cols = st.columns(3)
            with summary_cols[0]:
                metric_card("Staged Fights", len(staged), status="info")
            with summary_cols[1]:
                metric_card("Event", event_names[0] if event_names else "—", status="neutral")
            with summary_cols[2]:
                metric_card("Event ID", event_ids[0] if event_ids else "—", status="neutral")

            display_cols = [col for col in PREVIEW_COLUMNS if col in staged.columns]
            if display_cols:
                st.dataframe(staged[display_cols].head(25), use_container_width=True, hide_index=True)
        else:
            st.info("No profiled staged rows are currently available.")

        with st.expander("Pipeline Artifact Details", expanded=False):
            st.dataframe(artifact_df, use_container_width=True, hide_index=True)


def _render_data_quality_panel(precheck, final_review):
    section_heading("Data Quality Summary", "Append precheck and final review failures that block or warn before append.")
    with st.container(border=True):
        append_ready = _append_ready(precheck)
        final_pass = _final_review_pass(final_review)
        precheck_failed = _failed_checks(precheck)
        review_failed = _failed_checks(final_review)
        blocking_failed = review_failed[review_failed.get("severity", pd.Series(dtype=object)).astype(str) == "block"] if not review_failed.empty and "severity" in review_failed.columns else review_failed
        warning_failed = review_failed[review_failed.get("severity", pd.Series(dtype=object)).astype(str) == "warning"] if not review_failed.empty and "severity" in review_failed.columns else pd.DataFrame()

        cols = st.columns(4)
        with cols[0]:
            metric_card("Append Precheck", "Pass" if append_ready else "Blocked", status="success" if append_ready else "danger")
        with cols[1]:
            metric_card("Final Review", "Pass" if final_pass else "Blocked", status="success" if final_pass else "danger")
        with cols[2]:
            metric_card("Blocking Issues", len(blocking_failed), status="success" if len(blocking_failed) == 0 else "danger")
        with cols[3]:
            metric_card("Warnings", len(warning_failed), status="success" if len(warning_failed) == 0 else "warning")

        issues = []
        for source, failed in [("Append Precheck", precheck_failed), ("Final Review", review_failed)]:
            if failed.empty:
                continue
            display = failed.copy()
            display["source"] = source
            issues.append(display)

        if issues:
            issue_df = pd.concat(issues, ignore_index=True)
            display_cols = [col for col in ["source", "severity", "check_name", "status", "failure_count", "details"] if col in issue_df.columns]
            st.markdown("#### Recent Data Quality Issues")
            st.dataframe(issue_df[display_cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.success("No failed append precheck or final review checks are present in current artifacts.")

        with st.expander("All Append Precheck Checks", expanded=False):
            if precheck.empty:
                st.info(f"No append precheck artifact found at `{APPEND_PRECHECK_PATH}`.")
            else:
                st.dataframe(precheck, use_container_width=True, hide_index=True)

        with st.expander("All Final Review Checks", expanded=False):
            if final_review.empty:
                st.info(f"No final review artifact found at `{STAGED_FINAL_REVIEW_PATH}`.")
            else:
                st.dataframe(final_review, use_container_width=True, hide_index=True)


def _render_history_and_audits_panel(staged):
    section_heading("Recent Ingestion History & Audit Details", "Latest append audit plus inspectable data maintenance artifacts.")
    with st.container(border=True):
        append_audit = _safe_read(APPEND_AUDIT_PATH)
        if not append_audit.empty:
            latest = append_audit.tail(1)
            st.markdown("#### Latest Append Audit")
            st.dataframe(latest, use_container_width=True, hide_index=True)
        elif not staged.empty:
            st.info("Staged rows are available, but no append audit has been created yet.")
        else:
            st.info("No recent append audit is available.")

        audit_df = _artifact_rows(AUDIT_ARTIFACTS).rename(columns={"Stage": "Artifact"})
        st.markdown("#### Audit Artifact Status")
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

        selected = st.selectbox("Inspect audit artifact", options=[label for label, _ in AUDIT_ARTIFACTS])
        artifact_path = dict(AUDIT_ARTIFACTS)[selected]
        artifact = _safe_read(artifact_path)
        if artifact.empty:
            st.warning(f"No rows available at `{artifact_path}`.")
        else:
            st.caption(f"Path: `{artifact_path}` | Rows: {len(artifact):,} | Columns: {len(artifact.columns):,}")
            st.dataframe(artifact.head(100), use_container_width=True, hide_index=True)


def _render_append_approval_panel(precheck, final_review):
    section_heading("Append Approval", "Separate final action for appending reviewed staged rows into master.")
    with st.container(border=True):
        append_ready = _append_ready(precheck)
        final_pass = _final_review_pass(final_review)
        append_allowed = append_ready and final_pass

        cols = st.columns(4)
        with cols[0]:
            metric_card("Append Gate", "Ready" if append_allowed else "Blocked", status="success" if append_allowed else "danger")
        with cols[1]:
            metric_card("Precheck", "Pass" if append_ready else "Blocked", status="success" if append_ready else "danger")
        with cols[2]:
            metric_card("Final Review", "Pass" if final_pass else "Blocked", status="success" if final_pass else "danger")
        with cols[3]:
            staged_rows = precheck["staged_rows"].iloc[0] if not precheck.empty and "staged_rows" in precheck.columns else "—"
            metric_card("Rows To Append", staged_rows, status="info" if append_allowed else "neutral")

        if not append_allowed:
            st.info("Append remains disabled until append precheck and final review both pass.")

        if st.button(
            "⚠️ Append To Master",
            disabled=not append_allowed,
            type="primary" if append_allowed else "secondary",
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

        append_audit = _safe_read(APPEND_AUDIT_PATH)
        with st.expander("Latest Append Audit", expanded=False):
            if append_audit.empty:
                st.info(f"No append audit artifact found at `{APPEND_AUDIT_PATH}`.")
            else:
                st.dataframe(append_audit.tail(10), hide_index=True, use_container_width=True)


def render_data_maintenance():
    page_header(
        "Data Maintenance",
        "Monitor data health, discover new events, and manage the UFC master dataset.",
        kicker="Ingestion Control Tower",
    )

    status = _safe_read(DATASET_STATUS_PATH)
    missing_events = _safe_read(MISSING_EVENTS_PATH)
    staged = _safe_read(STAGED_MASTER_ROWS_PROFILED_PATH)
    precheck = _safe_read(APPEND_PRECHECK_PATH)
    final_review = _safe_read(STAGED_FINAL_REVIEW_PATH)

    _render_top_kpis(status, missing_events, staged, precheck, final_review)
    _render_dataset_status_panel(status)
    _render_event_discovery_panel(missing_events)
    _render_pipeline_status_panel(staged)
    _render_data_quality_panel(precheck, final_review)
    _render_history_and_audits_panel(staged)
    _render_append_approval_panel(precheck, final_review)
