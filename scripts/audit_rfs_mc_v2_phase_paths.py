"""Audit shared-phase behavior across many RFS Monte Carlo V2 paths."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from statistics import mean, median

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    run_shared_state_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


SEGMENT_SECONDS = 30


def neutral_profile(**overrides: float) -> FighterTransitionParameters:
    """Build a neutral transition profile with optional overrides."""

    neutral = FighterTransitionParameters(
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

    return replace(neutral, **overrides)


def scenarios() -> dict[
    str,
    tuple[FighterTransitionParameters, FighterTransitionParameters],
]:
    """Return the provisional transition scenarios to audit."""

    return {
        "NEUTRAL": (
            neutral_profile(),
            neutral_profile(),
        ),
        "RED WRESTLING ADVANTAGE": (
            neutral_profile(
                clinch_entry_tendency=0.90,
                takedown_entry_tendency=0.95,
                takedown_completion_ability=0.90,
                takedown_persistence=0.90,
                failed_takedown_persistence=0.85,
                clinch_retention=0.85,
                ground_retention=0.90,
                phase_imposition=0.90,
                phase_resistance=0.75,
            ),
            neutral_profile(
                clinch_entry_resistance=0.15,
                takedown_resistance=0.10,
                clinch_escape_ability=0.25,
                ground_escape_ability=0.20,
                reversal_ability=0.20,
                phase_resistance=0.15,
            ),
        ),
        "DISTANCE SPECIALISTS": (
            neutral_profile(
                distance_retention=0.95,
                clinch_entry_resistance=0.90,
                takedown_resistance=0.90,
                clinch_escape_ability=0.85,
                ground_escape_ability=0.80,
                phase_resistance=0.95,
            ),
            neutral_profile(
                distance_retention=0.95,
                clinch_entry_resistance=0.90,
                takedown_resistance=0.90,
                clinch_escape_ability=0.85,
                ground_escape_ability=0.80,
                phase_resistance=0.95,
            ),
        ),
    }


def record_phase_spells(
    phase_sequence: list[FightPhase],
    spell_lengths: dict[FightPhase, list[int]],
) -> None:
    """Record uninterrupted phase spells within one round."""

    if not phase_sequence:
        return

    current_phase = phase_sequence[0]
    current_length = 1

    for phase in phase_sequence[1:]:
        if phase is current_phase:
            current_length += 1
            continue

        spell_lengths[current_phase].append(current_length)
        current_phase = phase
        current_length = 1

    spell_lengths[current_phase].append(current_length)


def audit_scenario(
    name: str,
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    path_count: int,
    scheduled_rounds: int,
    seed: int,
) -> None:
    """Run and print one multi-path shared-state audit."""

    phase_counts: Counter[FightPhase] = Counter()
    owner_counts: Counter[tuple[FightPhase, FighterSide]] = Counter()
    transition_counts: Counter[tuple[str, str]] = Counter()
    transition_source_counts: Counter[FightPhase] = Counter()
    phase_spells: dict[FightPhase, list[int]] = defaultdict(list)

    for path_index in range(path_count):
        path = run_shared_state_path(
            red,
            blue,
            scheduled_rounds=scheduled_rounds,
            seed=seed + path_index,
        )

        round_phases: list[FightPhase] = []
        current_round = 1

        for record in path.segments:
            state = record.state

            if state.round_number != current_round:
                record_phase_spells(
                    round_phases,
                    phase_spells,
                )
                round_phases = []
                current_round = state.round_number

            round_phases.append(state.phase)
            phase_counts[state.phase] += 1

            if state.phase_owner is not None:
                owner_counts[
                    (state.phase, state.phase_owner)
                ] += 1

            if record.transition is not None:
                transition = record.transition
                transition_source_counts[state.phase] += 1

                actor = (
                    transition.actor.value
                    if transition.actor is not None
                    else "neutral"
                )

                transition_counts[
                    (transition.event.value, actor)
                ] += 1

        record_phase_spells(
            round_phases,
            phase_spells,
        )

    total_segments = sum(phase_counts.values())

    print(f"\n{name}")
    print("=" * len(name))
    print(f"Paths: {path_count:,}")
    print(f"Rounds per path: {scheduled_rounds}")
    print(f"Segments: {total_segments:,}")

    print("\nPHASE OCCUPANCY")
    for phase in FightPhase:
        count = phase_counts[phase]
        print(
            f"{phase.value:10s} "
            f"{count / total_segments:7.2%} "
            f"({count:,} segments)"
        )

    print("\nPHASE OWNERSHIP")
    for phase in (FightPhase.CLINCH, FightPhase.GROUND):
        phase_total = phase_counts[phase]

        for side in (FighterSide.RED, FighterSide.BLUE):
            count = owner_counts[(phase, side)]
            share = count / phase_total if phase_total else 0.0

            print(
                f"{phase.value:10s} "
                f"{side.value:5s} "
                f"{share:7.2%}"
            )

    print("\nPHASE SPELL LENGTH")
    for phase in FightPhase:
        lengths = phase_spells[phase]

        if not lengths:
            continue

        print(
            f"{phase.value:10s} "
            f"mean={mean(lengths):5.2f} segments "
            f"median={median(lengths):4.1f} "
            f"mean_seconds={mean(lengths) * SEGMENT_SECONDS:6.1f}"
        )

    print("\nTRANSITIONS")
    for (event, actor), count in transition_counts.most_common():
        print(
            f"{event:20s} "
            f"{actor:8s} "
            f"{count:8,d}"
        )

    if name == "NEUTRAL":
        red_owned = sum(
            owner_counts[(phase, FighterSide.RED)]
            for phase in (FightPhase.CLINCH, FightPhase.GROUND)
        )
        blue_owned = sum(
            owner_counts[(phase, FighterSide.BLUE)]
            for phase in (FightPhase.CLINCH, FightPhase.GROUND)
        )
        owned_total = red_owned + blue_owned

        red_share = red_owned / owned_total if owned_total else 0.0
        blue_share = blue_owned / owned_total if owned_total else 0.0

        print("\nNEUTRAL SYMMETRY")
        print(f"Red owned share:  {red_share:7.2%}")
        print(f"Blue owned share: {blue_share:7.2%}")
        print(
            f"Absolute gap:     "
            f"{abs(red_share - blue_share):7.2%}"
        )


def main() -> None:
    """Run all provisional shared-state scenarios."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        choices=(3, 5),
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    for scenario_index, (
        name,
        (red, blue),
    ) in enumerate(scenarios().items()):
        audit_scenario(
            name,
            red,
            blue,
            path_count=args.paths,
            scheduled_rounds=args.rounds,
            seed=args.seed + scenario_index * 1_000_000,
        )


if __name__ == "__main__":
    main()
