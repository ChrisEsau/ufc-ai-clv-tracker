from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from utils.operations_status_writer import (
    complete_runbook,
    complete_step,
    complete_substep,
    fail_runbook,
    start_runbook,
    start_step,
    start_substep,
)


@dataclass(frozen=True)
class Substep:
    substep_id: str
    name: str
    command: list[str]
    expected_outputs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Step:
    step_id: str
    name: str
    substeps: list[Substep]


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _validate_outputs(paths: Sequence[str]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing expected output(s): " + ", ".join(missing))


def _run_command(command: Sequence[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(list(command), check=True)


def build_steps(args: argparse.Namespace) -> list[Step]:
    refresh_command = _python_module("pipeline.prediction.run_refresh_upcoming_events")
    if args.max_upcoming_events:
        refresh_command.extend(["--max-events", str(args.max_upcoming_events)])

    matched_discovery_command = _python_module(
        "pipeline.market.run_draftkings_matched_discovery",
        "--sleep-seconds",
        str(args.draftkings_sleep_seconds),
    )
    if args.max_draftkings_events:
        matched_discovery_command.extend(["--max-events", str(args.max_draftkings_events)])

    return [
        Step(
            step_id="refresh_upcoming_events",
            name="Refresh Upcoming Events",
            substeps=[
                Substep(
                    substep_id="refresh_ufcstats_upcoming_events",
                    name="Refresh UFCStats Upcoming Events",
                    command=refresh_command,
                    expected_outputs=[
                        "data/cards/ufcstats_upcoming_events.parquet",
                        "data/cards/ufcstats_upcoming_fights.parquet",
                    ],
                )
            ],
        ),
        Step(
            step_id="run_predictions",
            name="Run Predictions",
            substeps=[
                Substep(
                    substep_id="build_fighter_state",
                    name="Build Fighter State V2",
                    command=_python_module("pipeline.features.run_build_fighter_state"),
                    expected_outputs=[
                        "data/features/fighter_state_history.parquet",
                        "data/features/latest_fighter_state.parquet",
                    ],
                ),
                Substep(
                    substep_id="build_feature_view",
                    name="Build Feature View V2",
                    command=_python_module(
                        "pipeline.features.run_build_feature_view",
                        "--config",
                        args.feature_view_config,
                    ),
                    expected_outputs=[args.feature_view_output],
                ),
                Substep(
                    substep_id="build_live_card",
                    name="Build Target Live Card",
                    command=_python_module("pipeline.prediction.run_build_target_live_card"),
                    expected_outputs=[
                        "data/predictions/ufc_live_card.parquet",
                        "data/cards/ufc_selected_live_card_event.parquet",
                    ],
                ),
                Substep(
                    substep_id="run_prediction_v2",
                    name="Run Prediction V2",
                    command=_python_module(
                        "pipeline.modeling.run_prediction",
                        "--model-family",
                        args.model_family,
                        "--model-id",
                        args.model_id,
                    ),
                    expected_outputs=[
                        "data/predictions/model_outcomes.parquet",
                        f"data/predictions/by_model/{args.model_id}/model_outcomes.parquet",
                    ],
                ),
            ],
        ),
        Step(
            step_id="get_market_odds",
            name="Get Market Odds",
            substeps=[
                Substep(
                    substep_id="draftkings_event_index",
                    name="Build DraftKings Event Index",
                    command=_python_module("pipeline.market.run_draftkings_event_index"),
                    expected_outputs=["data/market/draftkings_event_index.parquet"],
                ),
                Substep(
                    substep_id="draftkings_card_filter",
                    name="Match DraftKings Events To Card",
                    command=_python_module(
                        "pipeline.market.run_draftkings_card_filter",
                        "--min-match-score",
                        str(args.draftkings_card_min_match_score),
                        "--min-single-score",
                        str(args.draftkings_card_min_single_score),
                    ),
                    expected_outputs=["data/market/draftkings_event_card_matches.parquet"],
                ),
                Substep(
                    substep_id="draftkings_matched_discovery",
                    name="Discover DraftKings Markets",
                    command=matched_discovery_command,
                    expected_outputs=[
                        "data/market/draftkings_market_diagnostic.parquet",
                        "data/market/draftkings_raw_index.parquet",
                    ],
                ),
                Substep(
                    substep_id="draftkings_normalize_markets",
                    name="Normalize DraftKings Markets",
                    command=_python_module("pipeline.market.run_normalize_provider_markets", "--provider", "draftkings"),
                    expected_outputs=["data/market/canonical_market_catalog.parquet"],
                ),
                Substep(
                    substep_id="draftkings_match_markets",
                    name="Match DraftKings Markets To Live Card",
                    command=_python_module(
                        "pipeline.market.run_market_matching",
                        "--registry-path",
                        args.draftkings_registry_path,
                        "--min-match-score",
                        str(args.market_min_match_score),
                    ),
                    expected_outputs=[
                        "data/market/market_outcomes.parquet",
                        "data/audits/ufc_market_match_audit_v2.parquet",
                    ],
                ),
            ],
        ),
        Step(
            step_id="build_betting_outcomes",
            name="Build Betting Outcomes",
            substeps=[
                Substep(
                    substep_id="run_betting_outcomes_v2",
                    name="Run Betting Outcomes V2",
                    command=_python_module(
                        "pipeline.betting.run_betting_outcomes_v2",
                        "--model-mode",
                        args.model_mode,
                    ),
                    expected_outputs=["data/predictions/betting_outcomes.parquet"],
                ),
                Substep(
                    substep_id="run_betting_join_key_diagnostic",
                    name="Run Betting Join Key Diagnostic",
                    command=_python_module(
                        "pipeline.betting.run_betting_join_key_diagnostic",
                        "--model-mode",
                        args.model_mode,
                    ),
                    expected_outputs=["data/audits/ufc_betting_join_key_diagnostic.parquet"],
                ),
            ],
        ),
        Step(
            step_id="capture_snapshots",
            name="Capture Snapshots",
            substeps=[
                Substep(
                    substep_id="capture_model_market_snapshots",
                    name="Capture Model-Market Snapshots",
                    command=_python_module(
                        "pipeline.snapshots.run_capture_model_market_snapshots",
                        "--model-mode",
                        args.snapshot_model_mode,
                    ),
                    expected_outputs=[
                        "data/snapshots/model_market_snapshots.parquet",
                        "data/audits/model_market_snapshot_audit.parquet",
                    ],
                )
            ],
        ),
    ]


def run_market_refresh(args: argparse.Namespace) -> None:
    steps = build_steps(args)
    start_runbook(
        runbook_id="market_refresh_v2",
        mode=args.mode,
        step_total=len(steps),
        message=f"Market Refresh started in {args.mode} mode",
    )

    try:
        for step_index, step in enumerate(steps, start=1):
            start_step(
                step_id=step.step_id,
                step_name=step.name,
                step_index=step_index,
                step_total=len(steps),
                substep_total=len(step.substeps),
            )
            for substep_index, substep in enumerate(step.substeps, start=1):
                start_substep(
                    substep_id=substep.substep_id,
                    substep_name=substep.name,
                    substep_index=substep_index,
                    substep_total=len(step.substeps),
                    message=f"Running {substep.name}",
                )
                _run_command(substep.command)
                _validate_outputs(substep.expected_outputs)
                complete_substep(f"Completed {substep.name}")
            complete_step(f"Completed {step.name}")
        complete_runbook("Market Refresh completed")
    except Exception as exc:
        fail_runbook(str(exc))
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Market Refresh production runbook.")
    parser.add_argument("--mode", choices=["test", "production"], default="test")
    parser.add_argument("--max-upcoming-events", default="", help="Optional max UFCStats upcoming events to refresh.")
    parser.add_argument("--max-draftkings-events", default="5", help="Optional max matched DraftKings events; blank means all.")
    parser.add_argument("--draftkings-sleep-seconds", default="3")
    parser.add_argument("--draftkings-card-min-match-score", default="80")
    parser.add_argument("--draftkings-card-min-single-score", default="70")
    parser.add_argument("--market-min-match-score", default="65")
    parser.add_argument("--draftkings-registry-path", default="configs/market/providers/draftkings_ufc_registry.yaml")
    parser.add_argument("--feature-view-config", default="configs/feature_views/moneyline_base.yaml")
    parser.add_argument("--feature-view-output", default="data/features/moneyline_feature_view.parquet")
    parser.add_argument("--model-family", default="moneyline")
    parser.add_argument("--model-id", default="moneyline_xgboost_v5")
    parser.add_argument("--model-mode", choices=["production", "all", "single"], default="production")
    parser.add_argument(
        "--snapshot-model-mode",
        choices=["production", "all", "single"],
        default="all",
        help="Models included in append-only model-market snapshots. 'all' includes draft artifacts when present.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_market_refresh(args)


if __name__ == "__main__":
    main()
