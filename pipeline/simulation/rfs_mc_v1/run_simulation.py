"""Command-line runner for RFS Monte Carlo V1.

Example:

python -m pipeline.simulation.rfs_mc_v1.run_simulation \
  --red-fighter-id RED_ID \
  --blue-fighter-id BLUE_ID \
  --target-date 2026-08-10 \
  --weight-class Lightweight \
  --gender male \
  --scheduled-rounds 3 \
  --paths 1000 \
  --seed 42

This remains a shadow-only, uncalibrated V0 simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.simulation.rfs_mc_v1.contracts import (
    MatchupSimulationRequest,
)
from pipeline.simulation.rfs_mc_v1.parameter_registry import (
    LAST3_PROFILE_PARAMETER_DEFINITIONS,
    PROFILE_PARAMETER_DEFINITIONS,
)
from pipeline.simulation.rfs_mc_v1.offensive_power import (
    augment_profile_with_offensive_power,
)
from pipeline.simulation.rfs_mc_v1.profile_builder import (
    build_composite_profile_from_histories,
    load_default_rfs_histories,
)
from pipeline.simulation.rfs_mc_v1.runner import (
    simulate_scored_paths,
    summarize_scored_paths,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the shadow-only RFS Monte Carlo V1 simulator."
        )
    )

    parser.add_argument(
        "--red-fighter-id",
        required=True,
    )
    parser.add_argument(
        "--blue-fighter-id",
        required=True,
    )
    parser.add_argument(
        "--target-date",
        required=True,
        help="Fight date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--weight-class",
        required=True,
    )
    parser.add_argument(
        "--gender",
        required=True,
    )
    parser.add_argument(
        "--scheduled-rounds",
        type=int,
        choices=(3, 5),
        default=3,
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--feature-root",
        default="data/features",
    )
    parser.add_argument(
        "--profile-source",
        choices=("ewm", "last3"),
        default="ewm",
        help=(
            "RFS profile aggregation source. Last-3 uses fighter EWM "
            "as a fallback when a Last-3 value is unavailable."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build profiles, run paths, and return a serializable result."""

    histories = load_default_rfs_histories(
        feature_root=args.feature_root,
    )

    parameter_definitions = (
        LAST3_PROFILE_PARAMETER_DEFINITIONS
        if args.profile_source == "last3"
        else PROFILE_PARAMETER_DEFINITIONS
    )
    round_stats = pd.read_parquet(
        "data/fight_details/ufc_round_stats.parquet"
    )

    red_profile = build_composite_profile_from_histories(
        histories,
        fighter_id=args.red_fighter_id,
        target_date=args.target_date,
        scheduled_rounds=args.scheduled_rounds,
        weight_class=args.weight_class,
        gender=args.gender,
        parameter_definitions=parameter_definitions,
    )

    blue_profile = build_composite_profile_from_histories(
        histories,
        fighter_id=args.blue_fighter_id,
        target_date=args.target_date,
        scheduled_rounds=args.scheduled_rounds,
        weight_class=args.weight_class,
        gender=args.gender,
        parameter_definitions=parameter_definitions,
    )

    red_profile = augment_profile_with_offensive_power(
        red_profile,
        round_stats,
    )
    blue_profile = augment_profile_with_offensive_power(
        blue_profile,
        round_stats,
    )

    request = MatchupSimulationRequest(
        red_profile=red_profile,
        blue_profile=blue_profile,
        path_count=args.paths,
        seed=args.seed,
        simulator_version="rfs_mc_v1",
        calibration_version="uncalibrated_v0",
    )

    scored_paths = simulate_scored_paths(request)
    simulation_summary = summarize_scored_paths(
        scored_paths
    )

    result = {
        "warning": (
            "Uncalibrated V0 shadow simulation. "
            "Do not use as a betting forecast."
        ),
        "request": {
            "red_fighter_id": red_profile.fighter_id,
            "red_fighter_name": red_profile.fighter_name,
            "blue_fighter_id": blue_profile.fighter_id,
            "blue_fighter_name": blue_profile.fighter_name,
            "target_date": args.target_date,
            "weight_class": args.weight_class,
            "gender": args.gender,
            "scheduled_rounds": args.scheduled_rounds,
            "path_count": args.paths,
            "seed": args.seed,
        },
        "profile_metadata": {
            "red_prior_fight_count": (
                red_profile.prior_fight_count
            ),
            "blue_prior_fight_count": (
                blue_profile.prior_fight_count
            ),
            "red_valid_round_fight_count": (
                red_profile.valid_round_fight_count
            ),
            "blue_valid_round_fight_count": (
                blue_profile.valid_round_fight_count
            ),
            "red_low_experience": (
                red_profile.is_low_experience
            ),
            "blue_low_experience": (
                blue_profile.is_low_experience
            ),
        },
        "simulation": simulation_summary,
    }

    return result


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    if args.paths <= 0:
        parser.error("--paths must be positive")

    result = run_from_args(args)

    rendered = json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )

    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output.write_text(
            rendered + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
