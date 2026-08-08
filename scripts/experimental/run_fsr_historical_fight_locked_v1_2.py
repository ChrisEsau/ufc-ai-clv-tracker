"""Run population-centered locked FSR V1.1 with corrected phase activity rates.

This is a shadow-only calibration checkpoint. It does NOT change the Monte
Carlo engine or any locked FSR observation/evidence equations.

V1.2 preserves:
- FSR locked-family V1.1 population-centered ratings;
- the existing wrestling/control/striking/submission/durability mappings;
- MC transition, dynamic-state, finish, scoring, and judging engines.

V1.2 changes only the adapter's conversion of historical whole-round activity
rates into the phase-specific 30-second rates consumed by MC V2.

The MC contracts define strike/submission activity rates as expected events in
ONE ACTIVE 30-second phase segment. Historical baselines are whole fighter-round
rates. Dividing a whole-round rate by 10 is therefore incorrect when an event is
legal only in one phase.

We estimate a deterministic neutral, no-finish shared-state phase exposure and
use it as the population reference denominator:

    distance_rate_per_active_segment
        = historical_distance_attempts_per_fighter_round
          / neutral_distance_segments_per_round

    submission_rate_per_ground_owner_segment
        = historical_sub_attempts_per_fighter_round
          / neutral_ground_owner_segments_per_fighter_round

This keeps population-average physical activity on the historical scale while
still allowing matchup-specific transition behavior to change total exposure to
each phase.

Shadow/research only.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    run_shared_state_path,
)

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1 as locked_v1
from scripts.experimental import run_fsr_historical_fight_locked_v1_1 as locked_v1_1


REFERENCE_PATHS = 5000
REFERENCE_ROUNDS = 3
REFERENCE_SEED_START = 2026080800

# Capture before any monkey-patching.
_original_population_baselines = base.population_baselines


@lru_cache(maxsize=1)
def neutral_phase_exposure() -> dict[str, float]:
    """Estimate neutral phase exposure with the static no-finish state engine.

    The reference is intentionally independent of the target matchup and of
    activity/finish/dynamic-state randomness. Both fighters use 50-centered
    wrestling/control transition traits, and every simulated round is complete.
    """

    neutral_card = {
        "wrestling_entry": 50.0,
        "wrestling_conversion": 50.0,
        "td_defense": 50.0,
        "control_imposition": 50.0,
        "control_resistance": 50.0,
    }

    neutral_transition = locked_v1.build_transition(neutral_card)

    distance_segments = 0
    clinch_segments = 0
    ground_segments = 0
    red_ground_owner_segments = 0
    blue_ground_owner_segments = 0

    for index in range(REFERENCE_PATHS):
        path = run_shared_state_path(
            neutral_transition,
            neutral_transition,
            scheduled_rounds=REFERENCE_ROUNDS,
            seed=REFERENCE_SEED_START + index,
        )

        for segment in path.segments:
            state = segment.state

            if state.phase is FightPhase.DISTANCE:
                distance_segments += 1
            elif state.phase is FightPhase.CLINCH:
                clinch_segments += 1
            elif state.phase is FightPhase.GROUND:
                ground_segments += 1

                if state.phase_owner is FighterSide.RED:
                    red_ground_owner_segments += 1
                elif state.phase_owner is FighterSide.BLUE:
                    blue_ground_owner_segments += 1
                else:
                    raise RuntimeError(
                        "Ground reference segment has no authoritative owner."
                    )

    total_rounds = float(
        REFERENCE_PATHS * REFERENCE_ROUNDS
    )

    distance_per_round = distance_segments / total_rounds
    clinch_per_round = clinch_segments / total_rounds
    ground_per_round = ground_segments / total_rounds

    ground_owner_per_fighter_round = (
        red_ground_owner_segments
        + blue_ground_owner_segments
    ) / (2.0 * total_rounds)

    total_segments_per_round = (
        distance_per_round
        + clinch_per_round
        + ground_per_round
    )

    if abs(total_segments_per_round - 10.0) > 1e-9:
        raise RuntimeError(
            "Neutral static exposure does not sum to 10 segments per round: "
            f"{total_segments_per_round}"
        )

    if distance_per_round <= 0.0:
        raise RuntimeError(
            "Neutral reference generated no distance exposure."
        )

    if ground_owner_per_fighter_round <= 0.0:
        raise RuntimeError(
            "Neutral reference generated no ground-owner exposure."
        )

    return {
        "reference_distance_segments_per_round": distance_per_round,
        "reference_clinch_segments_per_round": clinch_per_round,
        "reference_ground_segments_per_round": ground_per_round,
        "reference_ground_owner_segments_per_fighter_round": (
            ground_owner_per_fighter_round
        ),
    }


def population_baselines(
    rounds,
    target_date,
) -> dict[str, float]:
    """Add phase-conditioned segment rates to the existing historical means."""

    result = dict(
        _original_population_baselines(
            rounds,
            target_date,
        )
    )

    exposure = neutral_phase_exposure()
    result.update(exposure)

    result["distance_attempt_rate_per_distance_segment"] = (
        result["distance_attempts_per_round"]
        / exposure["reference_distance_segments_per_round"]
    )

    result["sub_attempt_rate_per_ground_owner_segment"] = (
        result["sub_attempts_per_round"]
        / exposure[
            "reference_ground_owner_segments_per_fighter_round"
        ]
    )

    return result


def build_phase(
    fighter: dict[str, float],
    opponent: dict[str, float],
    baselines: dict[str, float],
):
    """Preserve V1 adapter logic except for phase-conditional activity rates."""

    # Start from the already directionally-tested locked V1 mapping.
    phase = locked_v1.build_phase(
        fighter,
        opponent,
        baselines,
    )

    distance_attempt_rate = max(
        0.0,
        baselines[
            "distance_attempt_rate_per_distance_segment"
        ],
    )

    # Submission Pressure remains the offensive attempt-generation skill and
    # Submission Resistance remains the opponent adjustment. Only the physical
    # baseline unit changes from whole-round/10 to ground-owner segment.
    submission_attempt_rate = base.matchup_rate(
        baseline=baselines[
            "sub_attempt_rate_per_ground_owner_segment"
        ],
        offense_rating=fighter[
            "submission_pressure"
        ],
        defense_rating=opponent[
            "submission_resistance"
        ],
    )

    return replace(
        phase,
        distance=replace(
            phase.distance,
            sig_strike_attempt_rate=(
                distance_attempt_rate
            ),
        ),
        ground_owner=replace(
            phase.ground_owner,
            submission_attempt_rate=(
                submission_attempt_rate
            ),
        ),
    )


def install_overrides() -> None:
    """Install only shadow-runner overrides; the MC engine remains untouched."""

    # First install population-centered cards and the already-tested V1
    # mappings for every other parameter family.
    locked_v1_1.install_overrides()

    # Then replace only the historical-rate conversion and phase builder.
    base.population_baselines = population_baselines
    base.build_phase = build_phase


if __name__ == "__main__":
    install_overrides()
    base.main()
