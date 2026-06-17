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
    ensure_data_dirs,
)
from pipeline.data_maintenance.run_dataset_status import run_dataset_status
from pipeline.data_maintenance.run_ingest_missing_events import run_ingest_missing_events

INGEST_MISSING_EVENTS_AUDIT_PATH = AUDITS_DIR / "ufc_missing_event_ingestion_audit.parquet"


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

    ingestion_audit = _run_function_step(
        results,
        "ingest_missing_completed_events",
        "Ingest Missing Completed Events",
        lambda: run_ingest_missing_events(
            max_events=max_events,
            auto_append=auto_append,
            continue_on_failure=False,
        ),
        [INGEST_MISSING_EVENTS_AUDIT_PATH],
    )

    if hasattr(ingestion_audit, "empty") and not ingestion_audit.empty:
        appended_count = int((ingestion_audit.get("append_status") == "success").sum()) if "append_status" in ingestion_audit.columns else 0
        ready_not_appended_count = int((ingestion_audit.get("append_status") == "ready_not_appended").sum()) if "append_status" in ingestion_audit.columns else 0
        skipped_count = int((ingestion_audit.get("stage_status") == "skipped_no_missing_events").sum()) if "stage_status" in ingestion_audit.columns else 0
        _record(
            results,
            "append_results_to_master",
            "Append Results To Master",
            "complete" if auto_append and appended_count > 0 else "skipped",
            f"auto_append={auto_append}, appended_events={appended_count}, ready_not_appended={ready_not_appended_count}, skipped={skipped_count}",
            [INGEST_MISSING_EVENTS_AUDIT_PATH],
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
