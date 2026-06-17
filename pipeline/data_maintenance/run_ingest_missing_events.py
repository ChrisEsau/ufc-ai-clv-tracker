from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    APPEND_AUDIT_PATH,
    APPEND_PRECHECK_PATH,
    AUDITS_DIR,
    MISSING_EVENTS_PATH,
    STAGED_FINAL_REVIEW_PATH,
    ensure_data_dirs,
)
from pipeline.data_maintenance.run_append_staged_to_master import run_append_staged_to_master
from pipeline.data_maintenance.run_ingest_single_event import run_ingest_single_event
from pipeline.data_maintenance.run_ufcstats_event_check import run_ufcstats_event_check

INGEST_MISSING_EVENTS_AUDIT_PATH = AUDITS_DIR / "ufc_missing_event_ingestion_audit.parquet"

AUDIT_COLUMNS = [
    "run_id",
    "run_timestamp",
    "event_index",
    "event_id",
    "event_name",
    "event_date",
    "auto_append",
    "stage_status",
    "append_ready",
    "final_review_pass",
    "append_status",
    "row_count_pass",
    "staged_rows",
    "rows_appended",
    "error",
]


def _normalize_max_events(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "all"}:
        return None
    parsed = int(text)
    if parsed < 1:
        raise ValueError("--max-events must be a positive integer or 'all'.")
    return parsed


def _load_missing_events(max_events: int | None) -> pd.DataFrame:
    if not MISSING_EVENTS_PATH.exists():
        return pd.DataFrame()

    missing = pd.read_parquet(MISSING_EVENTS_PATH)
    if missing.empty:
        return missing

    if "ufcstats_event_id" not in missing.columns:
        raise ValueError(f"Missing ufcstats_event_id column in {MISSING_EVENTS_PATH}")

    missing = missing.dropna(subset=["ufcstats_event_id"]).copy()
    missing["ufcstats_event_id"] = missing["ufcstats_event_id"].astype(str).str.strip()
    missing = missing[missing["ufcstats_event_id"] != ""].reset_index(drop=True)

    if max_events is not None:
        missing = missing.head(max_events).reset_index(drop=True)

    return missing


def _append_gate_passed() -> tuple[bool, bool, int]:
    append_ready = False
    final_review_pass = False
    staged_rows = 0

    if APPEND_PRECHECK_PATH.exists():
        precheck = pd.read_parquet(APPEND_PRECHECK_PATH)
        if not precheck.empty:
            if "append_ready" in precheck.columns:
                append_ready = bool(precheck["append_ready"].iloc[0])
            if "staged_rows" in precheck.columns:
                staged_rows = int(precheck["staged_rows"].iloc[0])

    if STAGED_FINAL_REVIEW_PATH.exists():
        final_review = pd.read_parquet(STAGED_FINAL_REVIEW_PATH)
        if not final_review.empty and "final_review_pass" in final_review.columns:
            final_review_pass = bool(final_review["final_review_pass"].iloc[0])

    return append_ready, final_review_pass, staged_rows


def _latest_append_audit() -> tuple[str, bool, int]:
    if not APPEND_AUDIT_PATH.exists():
        return "missing_audit", False, 0

    audit = pd.read_parquet(APPEND_AUDIT_PATH)
    if audit.empty:
        return "empty_audit", False, 0

    row = audit.iloc[-1]
    append_status = str(row.get("append_status", "unknown"))
    row_count_pass = bool(row.get("row_count_pass", False))
    rows_appended = int(row.get("staged_rows", 0) or 0)
    return append_status, row_count_pass, rows_appended


def _event_value(row: pd.Series, column: str) -> str | None:
    value = row.get(column)
    if pd.isna(value):
        return None
    return str(value)


def _write_audit(rows: list[dict]) -> pd.DataFrame:
    audit = pd.DataFrame(rows)
    for column in AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = pd.NA
    INGEST_MISSING_EVENTS_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit[AUDIT_COLUMNS].to_parquet(INGEST_MISSING_EVENTS_AUDIT_PATH, index=False)
    return audit[AUDIT_COLUMNS]


def run_ingest_missing_events(
    *,
    max_events: int | None = 1,
    auto_append: bool = False,
    continue_on_failure: bool = False,
) -> pd.DataFrame:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 80)
    print("UFC INGEST MISSING EVENTS")
    print("=" * 80)
    print("Run ID:", run_id)
    print("Max events:", "all" if max_events is None else max_events)
    print("Auto append:", auto_append)
    print("Continue on failure:", continue_on_failure)

    ensure_data_dirs()
    run_ufcstats_event_check()
    missing_events = _load_missing_events(max_events)

    print("Missing events selected:", len(missing_events))

    audit_rows: list[dict] = []

    if missing_events.empty:
        audit_rows.append(
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "event_index": 0,
                "event_id": None,
                "event_name": None,
                "event_date": None,
                "auto_append": auto_append,
                "stage_status": "skipped_no_missing_events",
                "append_ready": False,
                "final_review_pass": False,
                "append_status": "skipped",
                "row_count_pass": False,
                "staged_rows": 0,
                "rows_appended": 0,
                "error": None,
            }
        )
        return _write_audit(audit_rows)

    for idx, row in missing_events.iterrows():
        event_id = _event_value(row, "ufcstats_event_id")
        event_name = _event_value(row, "ufcstats_event_name")
        event_date = _event_value(row, "ufcstats_event_date")

        audit_row = {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "event_index": idx + 1,
            "event_id": event_id,
            "event_name": event_name,
            "event_date": event_date,
            "auto_append": auto_append,
            "stage_status": "not_started",
            "append_ready": False,
            "final_review_pass": False,
            "append_status": "not_started",
            "row_count_pass": False,
            "staged_rows": 0,
            "rows_appended": 0,
            "error": None,
        }

        print()
        print(f"[{idx + 1}/{len(missing_events)}] Event ID: {event_id}")
        print("Event:", event_name)

        try:
            if not event_id:
                raise ValueError("Missing event_id in missing-events artifact.")

            append_ready, final_review_pass = run_ingest_single_event(event_id=event_id)
            append_ready_from_artifact, final_review_from_artifact, staged_rows = _append_gate_passed()

            append_ready = bool(append_ready and append_ready_from_artifact)
            final_review_pass = bool(final_review_pass and final_review_from_artifact)

            audit_row["stage_status"] = "complete"
            audit_row["append_ready"] = append_ready
            audit_row["final_review_pass"] = final_review_pass
            audit_row["staged_rows"] = staged_rows

            if not (append_ready and final_review_pass):
                audit_row["append_status"] = "blocked_by_validation"
                raise RuntimeError(
                    f"Validation gates failed for event_id={event_id}. "
                    f"append_ready={append_ready}, final_review_pass={final_review_pass}"
                )

            if auto_append:
                append_audit, appended = run_append_staged_to_master()
                append_status, row_count_pass, rows_appended = _latest_append_audit()
                audit_row["append_status"] = append_status
                audit_row["row_count_pass"] = row_count_pass
                audit_row["rows_appended"] = rows_appended

                if not appended or append_audit is None or append_audit.empty or not row_count_pass:
                    raise RuntimeError(f"Append failed for event_id={event_id}. append_status={append_status}")
            else:
                audit_row["append_status"] = "ready_not_appended"

        except Exception as exc:
            audit_row["error"] = str(exc)
            if audit_row["stage_status"] == "not_started":
                audit_row["stage_status"] = "failed"
            audit_rows.append(audit_row)
            _write_audit(audit_rows)
            print("FAILED:", exc)
            if not continue_on_failure:
                raise
            continue

        audit_rows.append(audit_row)
        _write_audit(audit_rows)

    audit = _write_audit(audit_rows)

    print()
    print("========== MISSING EVENT INGESTION SUMMARY ==========")
    print(audit[["event_id", "event_name", "stage_status", "append_ready", "final_review_pass", "append_status", "rows_appended", "error"]].to_string(index=False))
    print("Saved:", INGEST_MISSING_EVENTS_AUDIT_PATH)

    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest missing completed UFCStats events with optional gated auto-append.")
    parser.add_argument("--max-events", default="1")
    parser.add_argument("--auto-append", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ingest_missing_events(
        max_events=_normalize_max_events(args.max_events),
        auto_append=bool(args.auto_append),
        continue_on_failure=bool(args.continue_on_failure),
    )


if __name__ == "__main__":
    main()
