from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


QUEUE_PATH = Path("data/status/ufc_round_stats_backfill_queue.parquet")
ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
VALIDATION_PATH = Path("data/audits/ufc_round_stats_validation.parquet")


def _run(command: list[str]) -> None:
    print()
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append-style completed-fight round-stats ingestion. "
            "Builds the master-minus-round-stats queue, scrapes a batch, "
            "and validates the round-stats dataset."
        )
    )
    parser.add_argument("--start-date", default="2026-03-26")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-fights", default="25")
    parser.add_argument("--sleep-seconds", default="10")
    parser.add_argument("--jitter-seconds", default="5")
    parser.add_argument("--max-failures", default="3")
    parser.add_argument("--reset-status", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--build-rfs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("INGEST COMPLETED ROUND STATS")
    print("=" * 80)
    print("Start date     :", args.start_date)
    print("End date       :", args.end_date or "today")
    print("Max fights     :", args.max_fights)
    print("Reset status   :", bool(args.reset_status))
    print("Build RFS      :", bool(args.build_rfs))

    queue_args = [
        "--start-date",
        args.start_date,
    ]

    if args.end_date:
        queue_args.extend(["--end-date", args.end_date])

    if args.reset_status:
        queue_args.append("--reset-status")

    _run(_python_module("pipeline.round_stats.build_round_stats_backfill_queue", *queue_args))

    _run(
        _python_module(
            "pipeline.round_stats.run_round_stats_backfill_batch",
            "--max-fights",
            str(args.max_fights),
            "--sleep-seconds",
            str(args.sleep_seconds),
            "--jitter-seconds",
            str(args.jitter_seconds),
            "--max-failures",
            str(args.max_failures),
        )
    )

    if not args.skip_validation:
        _run(_python_module("pipeline.round_stats.validate_round_stats_dataset"))

    if args.build_rfs:
        _run(_python_module("pipeline.round_stats.build_round_fighter_state"))
        _run(_python_module("pipeline.round_stats.validate_round_fighter_state"))

    print()
    print("=" * 80)
    print("COMPLETED ROUND-STATS INGEST FINISHED")
    print("=" * 80)
    print("Queue path      :", QUEUE_PATH)
    print("Round stats path:", ROUND_STATS_PATH)
    print("Validation path :", VALIDATION_PATH)


if __name__ == "__main__":
    main()
