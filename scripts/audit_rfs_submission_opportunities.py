"""Audit submission opportunities and attempts for one simulated matchup.

This uses activity-only paths so early KO/TKO finishes do not hide the
underlying ground and submission-attempt generation rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.rfs_mc_v1.contracts import (
    FightPhase,
    MatchupSimulationRequest,
)
from pipeline.simulation.rfs_mc_v1.parameter_registry import (
    LAST3_PROFILE_PARAMETER_DEFINITIONS,
    PROFILE_PARAMETER_DEFINITIONS,
)
from pipeline.simulation.rfs_mc_v1.profile_builder import (
    build_composite_profile_from_histories,
    load_default_rfs_histories,
)
from pipeline.simulation.rfs_mc_v1.runner import (
    simulate_activity_paths,
)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit ground opportunities and submission attempts "
            "for one RFS Monte Carlo matchup."
        )
    )

    parser.add_argument("--red-fighter-id", required=True)
    parser.add_argument("--blue-fighter-id", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--weight-class", required=True)
    parser.add_argument("--gender", required=True)

    parser.add_argument(
        "--scheduled-rounds",
        type=int,
        choices=(3, 5),
        default=3,
    )
    parser.add_argument(
        "--profile-source",
        choices=("ewm", "last3"),
        default="last3",
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
        "--output-root",
        type=Path,
        default=Path(
            "data/simulation/submission_opportunity_audits"
        ),
    )

    return parser


def summarize_values(
    values: list[float],
) -> dict[str, float]:
    """Return compact distribution statistics."""

    array = np.asarray(values, dtype=float)

    return {
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def main() -> None:
    """Run the opportunity audit."""

    args = build_parser().parse_args()

    if args.paths <= 0:
        raise SystemExit("--paths must be positive")

    histories = load_default_rfs_histories(
        feature_root=args.feature_root,
    )

    parameter_definitions = (
        LAST3_PROFILE_PARAMETER_DEFINITIONS
        if args.profile_source == "last3"
        else PROFILE_PARAMETER_DEFINITIONS
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

    request = MatchupSimulationRequest(
        red_profile=red_profile,
        blue_profile=blue_profile,
        path_count=args.paths,
        seed=args.seed,
        simulator_version="rfs_mc_v1",
        calibration_version=(
            "submission_opportunity_audit_v1"
        ),
    )

    simulation = simulate_activity_paths(request)

    path_rows: list[dict[str, object]] = []

    for path in simulation.paths:
        row: dict[str, object] = {
            "path_index": path.path_index,
            "seed": path.seed,
        }

        for side in ("red", "blue"):
            activities = [
                getattr(segment, side)
                for segment in path.segments
            ]

            ground_segments = sum(
                activity.phase == FightPhase.GROUND
                for activity in activities
            )

            control_segments = sum(
                activity.control_seconds > 0
                for activity in activities
            )

            opportunity_segments = sum(
                (
                    activity.phase == FightPhase.GROUND
                    or activity.control_seconds > 0
                )
                for activity in activities
            )

            attempt_segments = sum(
                activity.submission_attempts > 0
                for activity in activities
            )

            total_attempts = sum(
                int(activity.submission_attempts)
                for activity in activities
            )

            eligible_segments = sum(
                (
                    activity.submission_attempts > 0
                    and (
                        activity.phase == FightPhase.GROUND
                        or activity.control_seconds > 0
                    )
                )
                for activity in activities
            )

            outside_opportunity = sum(
                (
                    activity.submission_attempts > 0
                    and activity.phase != FightPhase.GROUND
                    and activity.control_seconds <= 0
                )
                for activity in activities
            )

            row[f"{side}_ground_segments"] = (
                ground_segments
            )
            row[f"{side}_control_segments"] = (
                control_segments
            )
            row[f"{side}_opportunity_segments"] = (
                opportunity_segments
            )
            row[f"{side}_attempt_segments"] = (
                attempt_segments
            )
            row[f"{side}_submission_attempts"] = (
                total_attempts
            )
            row[f"{side}_eligible_segments"] = (
                eligible_segments
            )
            row[
                f"{side}_attempts_outside_opportunity"
            ] = outside_opportunity

        path_rows.append(row)

    path_df = pd.DataFrame(path_rows)

    side_summaries: dict[str, object] = {}

    for side, profile in (
        ("red", red_profile),
        ("blue", blue_profile),
    ):
        opportunity_column = (
            f"{side}_opportunity_segments"
        )
        attempt_segment_column = (
            f"{side}_attempt_segments"
        )
        attempt_column = (
            f"{side}_submission_attempts"
        )

        total_opportunities = int(
            path_df[opportunity_column].sum()
        )
        total_attempt_segments = int(
            path_df[attempt_segment_column].sum()
        )
        total_attempts = int(
            path_df[attempt_column].sum()
        )

        side_summaries[side] = {
            "fighter_id": profile.fighter_id,
            "fighter_name": profile.fighter_name,
            "ground_segments_per_path": summarize_values(
                path_df[
                    f"{side}_ground_segments"
                ].tolist()
            ),
            "control_segments_per_path": summarize_values(
                path_df[
                    f"{side}_control_segments"
                ].tolist()
            ),
            "opportunity_segments_per_path": (
                summarize_values(
                    path_df[
                        opportunity_column
                    ].tolist()
                )
            ),
            "attempt_segments_per_path": summarize_values(
                path_df[
                    attempt_segment_column
                ].tolist()
            ),
            "submission_attempts_per_path": (
                summarize_values(
                    path_df[
                        attempt_column
                    ].tolist()
                )
            ),
            "paths_with_opportunity_pct": float(
                100.0
                * (
                    path_df[opportunity_column] > 0
                ).mean()
            ),
            "paths_with_attempt_pct": float(
                100.0
                * (
                    path_df[attempt_column] > 0
                ).mean()
            ),
            "attempt_segment_rate_per_opportunity": (
                0.0
                if total_opportunities == 0
                else float(
                    total_attempt_segments
                    / total_opportunities
                )
            ),
            "attempts_per_opportunity": (
                0.0
                if total_opportunities == 0
                else float(
                    total_attempts
                    / total_opportunities
                )
            ),
            "attempts_outside_opportunity": int(
                path_df[
                    f"{side}_attempts_outside_opportunity"
                ].sum()
            ),
        }

    summary = {
        "audit_version": (
            "submission_opportunity_audit_v1"
        ),
        "profile_source": args.profile_source,
        "target_date": args.target_date,
        "scheduled_rounds": args.scheduled_rounds,
        "path_count": args.paths,
        "seed": args.seed,
        "red": side_summaries["red"],
        "blue": side_summaries["blue"],
    }

    slug = (
        f"{red_profile.fighter_id}_vs_"
        f"{blue_profile.fighter_id}_"
        f"{args.target_date}_"
        f"{args.profile_source}"
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = args.output_root / f"{slug}_paths.csv"
    json_path = args.output_root / f"{slug}_summary.json"

    path_df.to_csv(csv_path, index=False)

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("SUBMISSION OPPORTUNITY AUDIT")
    print("=" * 78)
    print(
        f"{red_profile.fighter_name} vs "
        f"{blue_profile.fighter_name}"
    )
    print(
        f"Paths: {args.paths} | "
        f"Rounds: {args.scheduled_rounds} | "
        f"Profile: {args.profile_source}"
    )

    for side in ("red", "blue"):
        fighter = summary[side]

        print(
            f"\n{side.upper()} — "
            f"{fighter['fighter_name']}"
        )
        print(
            "Ground segments/path:       "
            f"{fighter['ground_segments_per_path']['mean']:.3f}"
        )
        print(
            "Control segments/path:      "
            f"{fighter['control_segments_per_path']['mean']:.3f}"
        )
        print(
            "Opportunity segments/path:  "
            f"{fighter['opportunity_segments_per_path']['mean']:.3f}"
        )
        print(
            "Attempt segments/path:      "
            f"{fighter['attempt_segments_per_path']['mean']:.3f}"
        )
        print(
            "Submission attempts/path:   "
            f"{fighter['submission_attempts_per_path']['mean']:.3f}"
        )
        print(
            "Paths with an opportunity:  "
            f"{fighter['paths_with_opportunity_pct']:.1f}%"
        )
        print(
            "Paths with an attempt:      "
            f"{fighter['paths_with_attempt_pct']:.1f}%"
        )
        print(
            "Attempt-segment rate:       "
            f"{100.0 * fighter['attempt_segment_rate_per_opportunity']:.2f}%"
        )
        print(
            "Attempts outside eligible "
            "position: "
            f"{fighter['attempts_outside_opportunity']}"
        )

    print(f"\nPath detail: {csv_path}")
    print(f"Locked summary: {json_path}")


if __name__ == "__main__":
    main()
