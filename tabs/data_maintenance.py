from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import (
    DATASET_STATUS_PATH,
    STAGED_FINAL_REVIEW_PATH,
    MISSING_EVENTS_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
    APPEND_PRECHECK_PATH,
)
from utils.dm_audit_history import render_audit_history
from utils.dm_dataset_health import render_dataset_health
from utils.dm_event_discovery import render_event_discovery
from utils.dm_final_review import render_final_review
from utils.ui.cards import metric_card
from utils.ui.sections import page_header, section_heading


def _safe_read(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _append_ready(precheck):
    if precheck.empty or "append_ready" not in precheck.columns:
        return False
    return bool(precheck["append_ready"].iloc[0])


def _final_review_pass(final_review):
    if final_review.empty:
        return False
    if "final_review_pass" in final_review.columns:
        return bool(final_review["final_review_pass"].iloc[0])
    if "check_passed" in final_review.columns:
        return bool(final_review["check_passed"].fillna(False).astype(bool).all())
    return False


def render_data_maintenance():
    page_header(
        "Data Maintenance",
        "Monitor data health, discover missing events, review staged rows, and gate master appends.",
        kicker="Ingestion Control Tower",
    )

    status = _safe_read(DATASET_STATUS_PATH)
    missing_events = _safe_read(MISSING_EVENTS_PATH)
    staged = _safe_read(STAGED_MASTER_ROWS_PROFILED_PATH)
    precheck = _safe_read(APPEND_PRECHECK_PATH)
    final_review = _safe_read(STAGED_FINAL_REVIEW_PATH)

    row = status.iloc[0] if not status.empty else {}
    append_ready = _append_ready(precheck)
    final_pass = _final_review_pass(final_review)

    cols = st.columns(6)
    with cols[0]:
        metric_card("Master Rows", row.get("row_count", "—"), caption="ufc_master.parquet", status="info")
    with cols[1]:
        metric_card("Master Columns", row.get("column_count", "—"), caption="Locked schema target: 128", status="neutral")
    with cols[2]:
        metric_card("Events", row.get("unique_events", "—"), caption="Unique event names", status="info")
    with cols[3]:
        metric_card("Missing Events", len(missing_events) if not missing_events.empty else 0, caption="Discovery artifact", status="warning" if not missing_events.empty else "success")
    with cols[4]:
        metric_card("Staged Rows", len(staged) if not staged.empty else 0, caption="Profiled master rows", status="info" if not staged.empty else "neutral")
    with cols[5]:
        gate_status = "success" if append_ready and final_pass else "danger"
        gate_value = "Ready" if append_ready and final_pass else "Blocked"
        metric_card("Append Gate", gate_value, caption="Requires precheck + final review", status=gate_status)

    section_heading(
        "Consolidated Data Maintenance Flow",
        "Following the current architecture: Dataset Health → Event Discovery → Final Staged Review → Audit History.",
    )

    render_dataset_health()
    render_event_discovery()
    render_final_review()
    render_audit_history()
