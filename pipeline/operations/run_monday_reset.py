from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from pipeline.common.paths import (
    AUDITS_DIR,
    BANKROLL_SNAPSHOTS_PATH,
    CLV_RESULTS_PATH,
    DATASET_STATUS_PATH,
    SELECTED_LIVE_CARD_EVENT_PATH,
    ensure_data_dirs,
)
from pipeline.data_maintenance.run_dataset_status import run_dataset_status
from pipeline.data_maintenance.run_ingest_missing_events import run_ingest_missing_events
from pipeline.events.run_set_target_event import run_set_target_event
from utils.operations_status_writer import (
    complete_runbook,
    complete_step,
    fail_runbook,
    start_runbook,
    start_step,
)

RUNBOOK_ID = "monday_reset_v1"
INGEST_MISSING_EVENTS_AUDIT_PATH = AUDITS_DIR / "ufc_missing_event_ingestion_audit.parquet"
BET_SETTLEMENT_AUDIT_PATH = AUDITS_DIR / "ufc_bet_settlement_audit.parquet"


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


def _run_function_step(results: list[StepResult], step_id: str, name: str, fn: Callable[[], object], outputs: Sequence[Path | str]) -> object:
    print()
    print(f"========== {name.upper()} ==========")
    value = fn()
    _record(results, step_id, name, "complete", outputs=outputs)
    return value


def _record_step_status(step_id: str, step_name: str, step_index: int, step_total: int) -> None:
    start_step(
        step_id=step_id,
        step_name=step_name,
        step_index=step_index,
        step_total=step_total,
        substep_total=1,
        runbook_id=RUNBOOK_ID,
    )


def _complete_step_status(message: str) -> None:
    complete_step(message, runbook_id=RUNBOOK_ID)


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
    step_total = 5
    start_runbook(
        runbook_id=RUNBOOK_ID,
        mode=mode,
        step_total=step_total,
        message=f"Monday Reset started in {mode} mode",
    )

    try:
        _record_step_status("process_completed_events", "Process Completed Events", 1, step_total)
        ingestion_audit = _run_function_step(
            results,
            "process_completed_events",
            "Process Completed Events",
            lambda: run_ingest_missing_events(
                max_events=max_events,
                auto_append=auto_append,
                continue_on_failure=False,
            ),
            [INGEST_MISSING_EVENTS_AUDIT_PATH],
        )
        _complete_step_status("Completed Process Completed Events")

        _record_step_status("update_master_dataset", "Update Master Dataset", 2, step_total)
        if hasattr(ingestion_audit, "empty") and not ingestion_audit.empty:
            appended_count = int((ingestion_audit.get("append_status") == "success").sum()) if "append_status" in ingestion_audit.columns else 0
            ready_not_appended_count = int((ingestion_audit.get("append_status") == "ready_not_appended").sum()) if "append_status" in ingestion_audit.columns else 0
            skipped_count = int((ingestion_audit.get("stage_status") == "skipped_no_missing_events").sum()) if "stage_status" in ingestion_audit.columns else 0
            _record(
                results,
                "update_master_dataset",
                "Update Master Dataset",
                "complete" if auto_append and appended_count > 0 else "skipped",
                f"auto_append={auto_append}, appended_events={appended_count}, ready_not_appended={ready_not_appended_count}, skipped={skipped_count}",
                [INGEST_MISSING_EVENTS_AUDIT_PATH],
            )
            _complete_step_status("Completed Update Master Dataset")
        else:
            _record(results, "update_master_dataset", "Update Master Dataset", "skipped", "No ingestion audit rows were produced.")
            _complete_step_status("Skipped Update Master Dataset")

        _record_step_status("refresh_platform_status", "Refresh Platform Status", 3, step_total)
        _run_function_step(
            results,
            "refresh_platform_status",
            "Refresh Platform Status",
            run_dataset_status,
            [DATASET_STATUS_PATH],
        )
        _run_function_step(
            results,
            "set_target_event",
            "Set Target Event",
            lambda: run_set_target_event(refresh_upcoming=True, max_events=1),
            [SELECTED_LIVE_CARD_EVENT_PATH],
        )
        _complete_step_status("Completed Refresh Platform Status")

        _record_step_status("reconcile_performance", "Reconcile Performance", 4, step_total)
        _run_command_step(
            results,
            "settle_open_bets",
            "Settle Open Bets",
            _python_module("pipeline.bankroll.run_settle_open_bets"),
            [BET_SETTLEMENT_AUDIT_PATH],
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
        _complete_step_status("Completed Reconcile Performance")

        _record_step_status("prepare_next_week", "Prepare Next Week", 5, step_total)
        _record(
            results,
            "prepare_next_week",
            "Prepare Next Week",
            "planned",
            "Weekly archive/reset and model performance runners are not implemented yet.",
        )
        _complete_step_status("Completed Prepare Next Week")

        complete_runbook("Monday Reset completed", runbook_id=RUNBOOK_ID)
    except Exception as exc:
        fail_runbook(str(exc), runbook_id=RUNBOOK_ID)
        raise

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
