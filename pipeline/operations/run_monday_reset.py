from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from pipeline.common.paths import (
    APPEND_AUDIT_PATH,
    APPEND_PRECHECK_PATH,
    BANKROLL_SNAPSHOTS_PATH,
    CLV_RESULTS_PATH,
    DATASET_STATUS_PATH,
    MISSING_EVENTS_PATH,
    STAGED_FINAL_REVIEW_PATH,
    ensure_data_dirs,
)
from pipeline.data_maintenance.run_dataset_status import run_dataset_status
from pipeline.data_maintenance.run_ufcstats_event_check import run_ufcstats_event_check


@dataclass(frozen=True)
class StepResult:
    step_id: str
    name: str
    status: str
    message: str = ""
    outputs: list[str] = field(default_factory=list)


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _run_command(command: Sequence[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(list(command), check=True)


def _path_exists(path: Path | str) -> bool:
    return Path(path).exists()


def _required_outputs(paths: Sequence[Path | str]) -> list[str]:
    return [str(path) for path in paths if _path_exists(path)]


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


def _load_missing_event_ids(max_events: int | None) -> list[str]:
    if not MISSING_EVENTS_PATH.exists():
        return []
    missing = pd.read_parquet(MISSING_EVENTS_PATH)
    if missing.empty or "ufcstats_event_id" not in missing.columns:
        return []

    event_ids = (
        missing["ufcstats_event_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    event_ids = [event_id for event_id in event_ids if event_id]
    if max_events is not None:
        event_ids = event_ids[:max_events]
    return event_ids


def _append_gate_passed() -> tuple[bool, bool]:
    append_ready = False
    final_review_pass = False

    if APPEND_PRECHECK_PATH.exists():
        precheck = pd.read_parquet(APPEND_PRECHECK_PATH)
        if not precheck.empty and "append_ready" in precheck.columns:
            append_ready = bool(precheck["append_ready"].iloc[0])

    if STAGED_FINAL_REVIEW_PATH.exists():
        final_review = pd.read_parquet(STAGED_FINAL_REVIEW_PATH)
        if not final_review.empty and "final_review_pass" in final_review.columns:
            final_review_pass = bool(final_review["final_review_pass"].iloc[0])

    return append_ready, final_review_pass


def _append_audit_passed() -> bool:
    if not APPEND_AUDIT_PATH.exists():
        return False
    audit = pd.read_parquet(APPEND_AUDIT_PATH)
    if audit.empty or "row_count_pass" not in audit.columns:
        return False
    return bool(audit["row_count_pass"].iloc[0])


def _record(results: list[StepResult], step_id: str, name: str, status: str, message: str = "", outputs: Sequence[Path | str] = ()) -> None:
    results.append(
        StepResult(
            step_id=step_id,
            name=name,
            status=status,
            message=message,
            outputs=_required_outputs(outputs),
        )
    )


def _run_command_step(results: list[StepResult], step_id: str, name: str, command: Sequence[str], outputs: Sequence[Path | str]) -> None:
    print()
    print(f"========== {name.upper()} ==========")
    _run_command(command)
    _record(results, step_id, name, "complete", outputs=outputs)


def _run_function_step(results: list[StepResult], step_id: str, name: str, fn: Callable[[], object], outputs: Sequence[Path | str]) -> None:
    print()
    print(f"========== {name.upper()} ==========")
    fn()
    _record(results, step_id, name, "complete", outputs=outputs)


def run_monday_reset(*, mode: str, max_events: int | None, auto_append: bool, run_bankroll: bool, run_clv: bool) -> list[StepResult]:
    print("=" * 80)
    print("UFC MONDAY RESET ORCHESTRATOR")
    print("=" * 80)
    print("Mode:", mode)
    print("Max events:", "all" if max_events is None else max_events)
    print("Auto append:", auto_append)
    print("Run bankroll:", run_bankroll)
    print("Run CLV:", run_clv)

    ensure_data_dirs()
    results: list[StepResult] = []

    _run_function_step(
        results,
        "discover_completed_results",
        "Discover Completed Results",
        run_ufcstats_event_check,
        [MISSING_EVENTS_PATH],
    )

    event_ids = _load_missing_event_ids(max_events)
    if not event_ids:
        _record(
            results,
            "ingest_completed_events",
            "Ingest Completed Events",
            "skipped",
            "No missing completed events found.",
            [MISSING_EVENTS_PATH],
        )
    else:
        print()
        print("========== INGEST COMPLETED EVENTS ==========")
        print("Events selected:", len(event_ids))

        for idx, event_id in enumerate(event_ids, start=1):
            print()
            print(f"[{idx}/{len(event_ids)}] Event ID: {event_id}")
            _run_command(_python_module("pipeline.data_maintenance.run_ingest_single_event", "--event-id", event_id))
            append_ready, final_review_pass = _append_gate_passed()
            if not (append_ready and final_review_pass):
                raise RuntimeError(
                    "Stopping Monday Reset before append because staged ingestion gates failed "
                    f"for event_id={event_id}. append_ready={append_ready}, final_review_pass={final_review_pass}"
                )

            if auto_append:
                _run_command(_python_module("pipeline.data_maintenance.run_append_staged_to_master"))
                if not _append_audit_passed():
                    raise RuntimeError(f"Append audit row_count_pass failed for event_id={event_id}.")
            else:
                print("Auto append disabled. Staged event was validated but not appended.")

        _record(
            results,
            "ingest_completed_events",
            "Ingest Completed Events",
            "complete",
            f"Processed {len(event_ids)} event(s).",
            [APPEND_PRECHECK_PATH, STAGED_FINAL_REVIEW_PATH],
        )

        append_ready, final_review_pass = _append_gate_passed()
        if auto_append:
            _record(
                results,
                "append_results_to_master",
                "Append Results To Master",
                "complete",
                "Auto append completed for all selected events.",
                [APPEND_AUDIT_PATH],
            )
        else:
            _record(
                results,
                "append_results_to_master",
                "Append Results To Master",
                "skipped",
                f"Auto append disabled. append_ready={append_ready}, final_review_pass={final_review_pass}",
                [APPEND_PRECHECK_PATH, STAGED_FINAL_REVIEW_PATH],
            )

    _run_function_step(
        results,
        "refresh_dataset_status",
        "Refresh Dataset Status",
        run_dataset_status,
        [DATASET_STATUS_PATH],
    )

    if run_bankroll:
        _run_command_step(
            results,
            "refresh_bankroll_status",
            "Refresh Bankroll Status",
            _python_module("pipeline.bankroll.run_bankroll_status"),
            [BANKROLL_SNAPSHOTS_PATH],
        )
    else:
        _record(results, "refresh_bankroll_status", "Refresh Bankroll Status", "skipped", "run_bankroll=false")

    if run_clv:
        _run_command_step(
            results,
            "run_clv_tracker",
            "Run CLV Tracker",
            _python_module("pipeline.clv.run_clv_pipeline"),
            [CLV_RESULTS_PATH],
        )
    else:
        _record(results, "run_clv_tracker", "Run CLV Tracker", "skipped", "run_clv=false")

    _record(
        results,
        "model_snapshot_performance",
        "Model / Snapshot Performance",
        "planned",
        "Performance runner not implemented yet.",
    )
    _record(
        results,
        "archive_reset_week",
        "Archive / Reset Week",
        "planned",
        "Weekly archive/reset runner not implemented yet.",
    )

    print()
    print("========== MONDAY RESET SUMMARY ==========")
    for result in results:
        suffix = f" - {result.message}" if result.message else ""
        print(f"{result.status.upper():9} {result.name}{suffix}")

    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Monday Reset post-event orchestration flow.")
    parser.add_argument("--mode", choices=["test", "production"], default="test")
    parser.add_argument(
        "--max-events",
        default="1",
        help="Maximum missing completed events to ingest. Use 'all' for every missing event.",
    )
    parser.add_argument(
        "--auto-append",
        action="store_true",
        help="Append staged rows to master only after append precheck and final review both pass.",
    )
    parser.add_argument("--skip-bankroll", action="store_true")
    parser.add_argument("--skip-clv", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    max_events = _normalize_max_events(args.max_events)
    run_monday_reset(
        mode=args.mode,
        max_events=max_events,
        auto_append=bool(args.auto_append),
        run_bankroll=not bool(args.skip_bankroll),
        run_clv=not bool(args.skip_clv),
    )


if __name__ == "__main__":
    main()
