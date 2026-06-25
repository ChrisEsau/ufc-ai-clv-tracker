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


def _is_uncapped(value: object) -> bool:
    return str(value or "").strip().lower() in {"", "all", "none", "null"}


def _validate_outputs(paths: Sequence[str]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing expected output(s): " + ", ".join(missing))


def _run_command(command: Sequence[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(list(command), check=True)


def build_steps(args: argparse.Namespace) -> list[Step]:
    matched_discovery_command = _python_module(
        "pipeline.market.run_draftkings_matched_discovery",
        "--sleep-seconds",
        str(args.draftkings_sleep_seconds),
    )
    if not _is_uncapped(args.max_draftkings_events):
        matched_discovery_command.extend(["--max-events", str(args.max_draftkings_events)])

    fanduel_matched_discovery_command = _python_module(
        "pipeline.market.run_fanduel_matched_discovery",
        "--sleep-seconds",
        str(args.fanduel_sleep_seconds),
    )
    if not _is_uncapped(args.max_fanduel_events):
        fanduel_matched_discovery_command.extend(["--max-events", str(args.max_fanduel_events)])

    production_prediction_command = _python_module(
        "pipeline.modeling.run_production_predictions",
        "--registry-path",
        args.model_registry_path,
    )
    if bool(args.prefer_raw_model):
        production_prediction_command.append("--prefer-raw-model")

    return [
        Step(
            step_id="run_predictions",
            name="Run All Production Predictions",
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
                    substep_id="run_production_predictions",
                    name="Run Production Registry Predictions",
                    command=production_prediction_command,
                    expected_outputs=["data/audits/production_prediction_audit.parquet"],
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
                    expected_outputs=["data/market/draftkings_market_catalog.parquet"],
                ),
                Substep(
                    substep_id="fanduel_event_index",
                    name="Build FanDuel Event Index",
                    command=_python_module("pipeline.market.run_fanduel_event_index"),
                    expected_outputs=["data/market/fanduel_event_index.parquet"],
                ),
                Substep(
                    substep_id="fanduel_card_filter",
                    name="Match FanDuel Events To Card",
                    command=_python_module(
                        "pipeline.market.run_fanduel_card_filter",
                        "--min-match-score",
                        str(args.draftkings_card_min_match_score),
                        "--min-single-score",
                        str(args.draftkings_card_min_single_score),
                    ),
                    expected_outputs=["data/market/fanduel_event_card_matches.parquet"],
                ),
                Substep(
                    substep_id="fanduel_matched_discovery",
                    name="Discover FanDuel Markets",
                    command=fanduel_matched_discovery_command,
                    expected_outputs=["data/market/fanduel_market_diagnostic.parquet"],
                ),
                Substep(
                    substep_id="fanduel_normalize_markets",
                    name="Normalize FanDuel Markets",
                    command=_python_module("pipeline.market.run_normalize_provider_markets", "--provider", "fanduel"),
                    expected_outputs=["data/market/fanduel_market_catalog.parquet"],
                ),
                Substep(
                    substep_id="merge_market_catalogs",
                    name="Merge Provider Market Catalogs",
                    command=_python_module("pipeline.market.run_merge_market_catalogs"),
                    expected_outputs=["data/market/canonical_market_catalog.parquet"],
                ),
                Substep(
                    substep_id="update_target_event_commence_time",
                    name="Update Target Event Commence Time",
                    command=_python_module("pipeline.market.run_update_target_event_commence_time"),
                    expected_outputs=["data/cards/ufc_selected_live_card_event.parquet"],
                ),
                Substep(
                    substep_id="match_markets",
                    name="Match Merged Markets To Live Card",
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
                ),
                Substep(
                    substep_id="capture_market_intelligence_history",
                    name="Capture Market Intelligence History",
                    command=_python_module("pipeline.market.run_capture_market_intelligence_history"),
                    expected_outputs=[
                        "data/market/market_intelligence_history.parquet",
                        "data/audits/market_intelligence_history_audit.parquet",
                    ],
                ),
                Substep(
                    substep_id="build_market_signals",
                    name="Build Market Signals",
                    command=_python_module("pipeline.market.run_build_market_signals"),
                    expected_outputs=[
                        "data/market/market_signals.parquet",
                        "data/audits/market_signals_audit.parquet",
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
    parser.add_argument("--max-draftkings-events", default="all", help="Optional max matched DraftKings events; use all/blank for every matched event.")
    parser.add_argument("--max-fanduel-events", default="all", help="Optional max matched FanDuel events; use all/blank for every matched event.")
    parser.add_argument("--draftkings-sleep-seconds", default="3")
    parser.add_argument("--fanduel-sleep-seconds", default="1")
    parser.add_argument("--draftkings-card-min-match-score", default="80")
    parser.add_argument("--draftkings-card-min-single-score", default="70")
    parser.add_argument("--market-min-match-score", default="65")
    parser.add_argument("--draftkings-registry-path", default="configs/market/providers/draftkings_ufc_registry.yaml")
    parser.add_argument("--model-registry-path", default="configs/models/model_registry.yaml")
    parser.add_argument("--feature-view-config", default="configs/feature_views/moneyline_base.yaml")
    parser.add_argument("--feature-view-output", default="data/features/moneyline_feature_view.parquet")
    parser.add_argument("--model-mode", choices=["production", "all", "single"], default="production")
    parser.add_argument("--prefer-raw-model", action="store_true")
    parser.add_argument(
        "--snapshot-model-mode",
        choices=["production", "all", "single"],
        default="production",
        help="Models included in append-only model-market snapshots.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_market_refresh(args)


if __name__ == "__main__":
    main()
