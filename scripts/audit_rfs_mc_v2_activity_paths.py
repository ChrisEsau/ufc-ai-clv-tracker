"""Population audit for RFS Monte Carlo V2 static activity paths.

This audit verifies that generated phase activity converges toward the
configured fighter parameters across many simulated paths.

It also checks structural legality:

- distance activity occurs only at distance
- only the clinch owner receives clinch control
- only the ground owner generates ground offense
- only the ground defender generates escape, reversal, and scramble attempts
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.activity_path_runner import (
    run_static_activity_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


@dataclass(frozen=True)
class AuditCheck:
    """One population-level calibration check."""

    name: str
    observed: float
    expected: float
    tolerance: float

    @property
    def passed(self) -> bool:
        """Return whether the observed value is inside tolerance."""

        return (
            math.isfinite(self.observed)
            and abs(self.observed - self.expected)
            <= self.tolerance
        )


def safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Return a safe ratio or NaN when the denominator is zero."""

    if denominator == 0:
        return math.nan

    return float(numerator) / float(denominator)


def neutral_transition_parameters() -> FighterTransitionParameters:
    """Return the neutral transition profile used by the path audit."""

    return FighterTransitionParameters(
        distance_retention=0.50,
        clinch_entry_tendency=0.50,
        clinch_entry_resistance=0.50,
        takedown_entry_tendency=0.50,
        takedown_completion_ability=0.50,
        takedown_resistance=0.50,
        takedown_persistence=0.50,
        failed_takedown_persistence=0.50,
        clinch_retention=0.50,
        clinch_escape_ability=0.50,
        ground_retention=0.50,
        ground_escape_ability=0.50,
        reversal_ability=0.50,
        phase_imposition=0.50,
        phase_resistance=0.50,
    )


def red_phase_parameters() -> FighterPhaseParameters:
    """Return an intentionally asymmetric red activity profile."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=3.20,
            sig_strike_accuracy=0.48,
            knockdown_probability_per_landed=0.025,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=1.70,
            clinch_strike_accuracy=0.52,
            control_seconds_mean=10.0,
            damaging_clinch_probability=0.09,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=2.50,
            ground_strike_accuracy=0.55,
            control_seconds_mean=17.0,
            submission_attempt_rate=0.25,
            position_advancement_probability=0.30,
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=0.22,
            reversal_attempt_rate=0.10,
            scramble_attempt_rate=0.16,
            submission_defense=0.78,
        ),
    )


def blue_phase_parameters() -> FighterPhaseParameters:
    """Return an intentionally asymmetric blue activity profile."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=2.60,
            sig_strike_accuracy=0.42,
            knockdown_probability_per_landed=0.018,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=1.20,
            clinch_strike_accuracy=0.46,
            control_seconds_mean=6.0,
            damaging_clinch_probability=0.06,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=1.80,
            ground_strike_accuracy=0.48,
            control_seconds_mean=12.0,
            submission_attempt_rate=0.14,
            position_advancement_probability=0.20,
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=0.18,
            reversal_attempt_rate=0.07,
            scramble_attempt_rate=0.12,
            submission_defense=0.68,
        ),
    )


def add_fighter_counter(
    counters: dict[str, Counter[str]],
    side: str,
    **values: int,
) -> None:
    """Add multiple integer values to one fighter counter."""

    counters[side].update(values)


def run_audit(
    *,
    path_count: int,
    scheduled_rounds: int,
    seed_start: int,
) -> int:
    """Run the population activity audit."""

    if path_count <= 0:
        raise ValueError("path_count must be positive")

    red_transition = neutral_transition_parameters()
    blue_transition = neutral_transition_parameters()

    red_phase = red_phase_parameters()
    blue_phase = blue_phase_parameters()

    phase_segments: Counter[FightPhase] = Counter()
    clinch_owner_segments: Counter[FighterSide] = Counter()
    ground_owner_segments: Counter[FighterSide] = Counter()
    ground_defender_segments: Counter[FighterSide] = Counter()

    distance = {
        "red": Counter(),
        "blue": Counter(),
    }
    clinch = {
        "red": Counter(),
        "blue": Counter(),
    }
    ground_owner = {
        "red": Counter(),
        "blue": Counter(),
    }
    ground_defender = {
        "red": Counter(),
        "blue": Counter(),
    }

    violations: Counter[str] = Counter()

    for path_index in range(path_count):
        path = run_static_activity_path(
            red_transition,
            blue_transition,
            red_phase,
            blue_phase,
            scheduled_rounds=scheduled_rounds,
            seed=seed_start + path_index,
        )

        for record in path.segments:
            phase_segments[record.state.phase] += 1
            activity = record.activity

            if record.state.phase is FightPhase.DISTANCE:
                if not isinstance(
                    activity,
                    DistanceSegmentActivity,
                ):
                    violations["distance_activity_type"] += 1
                    continue

                add_fighter_counter(
                    distance,
                    "red",
                    segments=1,
                    attempted=activity.red.sig_str_attempted,
                    landed=activity.red.sig_str_landed,
                    knockdowns=activity.red.knockdowns,
                )
                add_fighter_counter(
                    distance,
                    "blue",
                    segments=1,
                    attempted=activity.blue.sig_str_attempted,
                    landed=activity.blue.sig_str_landed,
                    knockdowns=activity.blue.knockdowns,
                )

            elif record.state.phase is FightPhase.CLINCH:
                if not isinstance(
                    activity,
                    ClinchSegmentActivity,
                ):
                    violations["clinch_activity_type"] += 1
                    continue

                add_fighter_counter(
                    clinch,
                    "red",
                    segments=1,
                    attempted=activity.red.clinch_str_attempted,
                    landed=activity.red.clinch_str_landed,
                    damaging=(
                        activity.red.damaging_clinch_strikes
                    ),
                )
                add_fighter_counter(
                    clinch,
                    "blue",
                    segments=1,
                    attempted=activity.blue.clinch_str_attempted,
                    landed=activity.blue.clinch_str_landed,
                    damaging=(
                        activity.blue.damaging_clinch_strikes
                    ),
                )

                if record.state.phase_owner is FighterSide.RED:
                    clinch_owner_segments[FighterSide.RED] += 1
                    clinch["red"]["owner_control"] += (
                        activity.red.control_seconds
                    )

                    if activity.blue.control_seconds != 0:
                        violations[
                            "blue_clinch_defender_control"
                        ] += 1

                elif record.state.phase_owner is FighterSide.BLUE:
                    clinch_owner_segments[FighterSide.BLUE] += 1
                    clinch["blue"]["owner_control"] += (
                        activity.blue.control_seconds
                    )

                    if activity.red.control_seconds != 0:
                        violations[
                            "red_clinch_defender_control"
                        ] += 1

                else:
                    violations["clinch_missing_owner"] += 1

            elif record.state.phase is FightPhase.GROUND:
                if not isinstance(
                    activity,
                    GroundSegmentActivity,
                ):
                    violations["ground_activity_type"] += 1
                    continue

                if record.state.phase_owner is FighterSide.RED:
                    owner_side = FighterSide.RED
                    defender_side = FighterSide.BLUE
                    owner_name = "red"
                    defender_name = "blue"
                    owner_activity = activity.red
                    defender_activity = activity.blue

                elif record.state.phase_owner is FighterSide.BLUE:
                    owner_side = FighterSide.BLUE
                    defender_side = FighterSide.RED
                    owner_name = "blue"
                    defender_name = "red"
                    owner_activity = activity.blue
                    defender_activity = activity.red

                else:
                    violations["ground_missing_owner"] += 1
                    continue

                ground_owner_segments[owner_side] += 1
                ground_defender_segments[defender_side] += 1

                add_fighter_counter(
                    ground_owner,
                    owner_name,
                    segments=1,
                    attempted=owner_activity.ground_str_attempted,
                    landed=owner_activity.ground_str_landed,
                    control=owner_activity.control_seconds,
                    submissions=owner_activity.submission_attempts,
                    advancements=(
                        owner_activity.position_advancements
                    ),
                )

                add_fighter_counter(
                    ground_defender,
                    defender_name,
                    segments=1,
                    escapes=defender_activity.escape_attempts,
                    reversals=defender_activity.reversal_attempts,
                    scrambles=defender_activity.scramble_attempts,
                )

                if (
                    owner_activity.escape_attempts != 0
                    or owner_activity.reversal_attempts != 0
                    or owner_activity.scramble_attempts != 0
                ):
                    violations[
                        "ground_owner_defensive_activity"
                    ] += 1

                if (
                    defender_activity.ground_str_attempted != 0
                    or defender_activity.ground_str_landed != 0
                    or defender_activity.control_seconds != 0
                    or defender_activity.submission_attempts != 0
                    or defender_activity.position_advancements != 0
                ):
                    violations[
                        "ground_defender_offensive_activity"
                    ] += 1

    total_segments = sum(phase_segments.values())

    checks = [
        AuditCheck(
            "red distance attempts / segment",
            safe_ratio(
                distance["red"]["attempted"],
                distance["red"]["segments"],
            ),
            red_phase.distance.sig_strike_attempt_rate,
            0.05,
        ),
        AuditCheck(
            "blue distance attempts / segment",
            safe_ratio(
                distance["blue"]["attempted"],
                distance["blue"]["segments"],
            ),
            blue_phase.distance.sig_strike_attempt_rate,
            0.05,
        ),
        AuditCheck(
            "red distance accuracy",
            safe_ratio(
                distance["red"]["landed"],
                distance["red"]["attempted"],
            ),
            red_phase.distance.sig_strike_accuracy,
            0.01,
        ),
        AuditCheck(
            "blue distance accuracy",
            safe_ratio(
                distance["blue"]["landed"],
                distance["blue"]["attempted"],
            ),
            blue_phase.distance.sig_strike_accuracy,
            0.01,
        ),
        AuditCheck(
            "red knockdowns / landed",
            safe_ratio(
                distance["red"]["knockdowns"],
                distance["red"]["landed"],
            ),
            red_phase.distance.knockdown_probability_per_landed,
            0.005,
        ),
        AuditCheck(
            "blue knockdowns / landed",
            safe_ratio(
                distance["blue"]["knockdowns"],
                distance["blue"]["landed"],
            ),
            blue_phase.distance.knockdown_probability_per_landed,
            0.005,
        ),
        AuditCheck(
            "red clinch attempts / segment",
            safe_ratio(
                clinch["red"]["attempted"],
                clinch["red"]["segments"],
            ),
            red_phase.clinch.clinch_strike_attempt_rate,
            0.05,
        ),
        AuditCheck(
            "blue clinch attempts / segment",
            safe_ratio(
                clinch["blue"]["attempted"],
                clinch["blue"]["segments"],
            ),
            blue_phase.clinch.clinch_strike_attempt_rate,
            0.05,
        ),
        AuditCheck(
            "red clinch accuracy",
            safe_ratio(
                clinch["red"]["landed"],
                clinch["red"]["attempted"],
            ),
            red_phase.clinch.clinch_strike_accuracy,
            0.015,
        ),
        AuditCheck(
            "blue clinch accuracy",
            safe_ratio(
                clinch["blue"]["landed"],
                clinch["blue"]["attempted"],
            ),
            blue_phase.clinch.clinch_strike_accuracy,
            0.015,
        ),
        AuditCheck(
            "red damaging clinch / landed",
            safe_ratio(
                clinch["red"]["damaging"],
                clinch["red"]["landed"],
            ),
            red_phase.clinch.damaging_clinch_probability,
            0.015,
        ),
        AuditCheck(
            "blue damaging clinch / landed",
            safe_ratio(
                clinch["blue"]["damaging"],
                clinch["blue"]["landed"],
            ),
            blue_phase.clinch.damaging_clinch_probability,
            0.015,
        ),
        AuditCheck(
            "red clinch owner control / segment",
            safe_ratio(
                clinch["red"]["owner_control"],
                clinch_owner_segments[FighterSide.RED],
            ),
            red_phase.clinch.control_seconds_mean,
            0.20,
        ),
        AuditCheck(
            "blue clinch owner control / segment",
            safe_ratio(
                clinch["blue"]["owner_control"],
                clinch_owner_segments[FighterSide.BLUE],
            ),
            blue_phase.clinch.control_seconds_mean,
            0.20,
        ),
        AuditCheck(
            "red ground attempts / owner segment",
            safe_ratio(
                ground_owner["red"]["attempted"],
                ground_owner["red"]["segments"],
            ),
            red_phase.ground_owner.ground_strike_attempt_rate,
            0.08,
        ),
        AuditCheck(
            "blue ground attempts / owner segment",
            safe_ratio(
                ground_owner["blue"]["attempted"],
                ground_owner["blue"]["segments"],
            ),
            blue_phase.ground_owner.ground_strike_attempt_rate,
            0.08,
        ),
        AuditCheck(
            "red ground accuracy",
            safe_ratio(
                ground_owner["red"]["landed"],
                ground_owner["red"]["attempted"],
            ),
            red_phase.ground_owner.ground_strike_accuracy,
            0.015,
        ),
        AuditCheck(
            "blue ground accuracy",
            safe_ratio(
                ground_owner["blue"]["landed"],
                ground_owner["blue"]["attempted"],
            ),
            blue_phase.ground_owner.ground_strike_accuracy,
            0.015,
        ),
        AuditCheck(
            "red ground control / owner segment",
            safe_ratio(
                ground_owner["red"]["control"],
                ground_owner["red"]["segments"],
            ),
            red_phase.ground_owner.control_seconds_mean,
            0.25,
        ),
        AuditCheck(
            "blue ground control / owner segment",
            safe_ratio(
                ground_owner["blue"]["control"],
                ground_owner["blue"]["segments"],
            ),
            blue_phase.ground_owner.control_seconds_mean,
            0.25,
        ),
        AuditCheck(
            "red submissions / owner segment",
            safe_ratio(
                ground_owner["red"]["submissions"],
                ground_owner["red"]["segments"],
            ),
            red_phase.ground_owner.submission_attempt_rate,
            0.03,
        ),
        AuditCheck(
            "blue submissions / owner segment",
            safe_ratio(
                ground_owner["blue"]["submissions"],
                ground_owner["blue"]["segments"],
            ),
            blue_phase.ground_owner.submission_attempt_rate,
            0.03,
        ),
        AuditCheck(
            "red advancements / owner segment",
            safe_ratio(
                ground_owner["red"]["advancements"],
                ground_owner["red"]["segments"],
            ),
            (
                red_phase.ground_owner
                .position_advancement_probability
            ),
            0.02,
        ),
        AuditCheck(
            "blue advancements / owner segment",
            safe_ratio(
                ground_owner["blue"]["advancements"],
                ground_owner["blue"]["segments"],
            ),
            (
                blue_phase.ground_owner
                .position_advancement_probability
            ),
            0.02,
        ),
        AuditCheck(
            "red escapes / defender segment",
            safe_ratio(
                ground_defender["red"]["escapes"],
                ground_defender["red"]["segments"],
            ),
            red_phase.ground_defender.escape_attempt_rate,
            0.03,
        ),
        AuditCheck(
            "blue escapes / defender segment",
            safe_ratio(
                ground_defender["blue"]["escapes"],
                ground_defender["blue"]["segments"],
            ),
            blue_phase.ground_defender.escape_attempt_rate,
            0.03,
        ),
        AuditCheck(
            "red reversals / defender segment",
            safe_ratio(
                ground_defender["red"]["reversals"],
                ground_defender["red"]["segments"],
            ),
            red_phase.ground_defender.reversal_attempt_rate,
            0.02,
        ),
        AuditCheck(
            "blue reversals / defender segment",
            safe_ratio(
                ground_defender["blue"]["reversals"],
                ground_defender["blue"]["segments"],
            ),
            blue_phase.ground_defender.reversal_attempt_rate,
            0.02,
        ),
        AuditCheck(
            "red scrambles / defender segment",
            safe_ratio(
                ground_defender["red"]["scrambles"],
                ground_defender["red"]["segments"],
            ),
            red_phase.ground_defender.scramble_attempt_rate,
            0.025,
        ),
        AuditCheck(
            "blue scrambles / defender segment",
            safe_ratio(
                ground_defender["blue"]["scrambles"],
                ground_defender["blue"]["segments"],
            ),
            blue_phase.ground_defender.scramble_attempt_rate,
            0.025,
        ),
        AuditCheck(
            "structural legality violations",
            float(sum(violations.values())),
            0.0,
            0.0,
        ),
    ]

    print("=" * 80)
    print("RFS MONTE CARLO V2 STATIC ACTIVITY PATH AUDIT")
    print("=" * 80)
    print(f"Paths:             {path_count:,}")
    print(f"Scheduled rounds:  {scheduled_rounds}")
    print(f"Segments:          {total_segments:,}")
    print()

    print("PHASE OCCUPANCY")
    for phase in (
        FightPhase.DISTANCE,
        FightPhase.CLINCH,
        FightPhase.GROUND,
    ):
        count = phase_segments[phase]
        share = safe_ratio(count, total_segments)

        print(
            f"  {phase.name:<10}"
            f"{count:>10,} segments  "
            f"{share:>7.2%}"
        )

    print()
    print("OWNERSHIP SAMPLE COUNTS")
    print(
        "  Clinch red owner: "
        f"{clinch_owner_segments[FighterSide.RED]:,}"
    )
    print(
        "  Clinch blue owner: "
        f"{clinch_owner_segments[FighterSide.BLUE]:,}"
    )
    print(
        "  Ground red owner: "
        f"{ground_owner_segments[FighterSide.RED]:,}"
    )
    print(
        "  Ground blue owner: "
        f"{ground_owner_segments[FighterSide.BLUE]:,}"
    )

    print()
    print("CALIBRATION CHECKS")

    all_passed = True

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        all_passed = all_passed and check.passed

        print(
            f"  [{status}] {check.name:<43}"
            f" observed={check.observed:>8.4f}"
            f" expected={check.expected:>8.4f}"
            f" tolerance={check.tolerance:>7.4f}"
        )

    if violations:
        print()
        print("STRUCTURAL VIOLATIONS")

        for name, count in sorted(violations.items()):
            print(f"  {name}: {count:,}")

    print()
    print("=" * 80)
    print(
        "AUDIT PASS"
        if all_passed
        else "AUDIT FAIL"
    )
    print("=" * 80)

    return 0 if all_passed else 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit V2 static activity generation across "
            "many simulated fight paths."
        )
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=10_000,
        help="Number of simulated paths.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        choices=(3, 5),
        default=3,
        help="Scheduled rounds per path.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First deterministic path seed.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the command-line audit."""

    args = parse_args()

    return run_audit(
        path_count=args.paths,
        scheduled_rounds=args.rounds,
        seed_start=args.seed_start,
    )


if __name__ == "__main__":
    raise SystemExit(main())
