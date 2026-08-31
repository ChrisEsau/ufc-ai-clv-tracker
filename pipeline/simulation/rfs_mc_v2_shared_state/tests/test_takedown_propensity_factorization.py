"""Regression tests for two-stage and chain-wrestling takedown semantics."""

from dataclasses import replace

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
    TransitionProbability,
    build_clinch_transition_distribution,
    build_distance_transition_distribution,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_sampler import (
    apply_transition_option,
)


def neutral_parameters() -> FighterTransitionParameters:
    """Return the 0.50 neutral transition profile used by calibration docs."""

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


def sequence_probability(distribution, side: FighterSide) -> float:
    """Return terminal success + failure probability for an initiated chain."""

    return (
        distribution.probability(
            TransitionEvent.TAKEDOWN,
            side,
        )
        + distribution.probability(
            TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
            side,
        )
    )


def expected_attempts_from_distribution(
    distribution,
    side: FighterSide,
) -> float:
    """Return unconditional expected attempts contributed by one fighter."""

    return sum(
        option.probability * option.attempt_count
        for option in distribution.options
        if (
            option.actor is side
            and option.event in {
                TransitionEvent.TAKEDOWN,
                TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
            }
        )
    )


def test_neutral_distance_sequence_mass_is_preserved() -> None:
    """Chain expansion must preserve the old 0.75 initiation weight."""

    distribution = build_distance_transition_distribution(
        neutral_parameters(),
        neutral_parameters(),
    )

    total_weight = 6.0 + 1.0 + 1.0 + 0.75 + 0.75
    expected_sequence = 0.75 / total_weight

    assert sequence_probability(
        distribution,
        FighterSide.RED,
    ) == pytest.approx(expected_sequence)
    assert sequence_probability(
        distribution,
        FighterSide.BLUE,
    ) == pytest.approx(expected_sequence)

    # Multiple-attempt chains make expected attempts larger than sequence
    # initiation probability even though initiation mass itself is unchanged.
    assert expected_attempts_from_distribution(
        distribution,
        FighterSide.RED,
    ) > expected_sequence


def test_neutral_clinch_sequence_mass_is_preserved() -> None:
    """Owner/defender neutral sequence masses remain 15% and 5%."""

    distribution = build_clinch_transition_distribution(
        neutral_parameters(),
        neutral_parameters(),
        current_owner=FighterSide.RED,
    )

    assert sequence_probability(
        distribution,
        FighterSide.RED,
    ) == pytest.approx(0.15)
    assert sequence_probability(
        distribution,
        FighterSide.BLUE,
    ) == pytest.approx(0.05)


def zero_propensity_elite_converter() -> FighterTransitionParameters:
    """Strong success skills cannot create a wrestling sequence."""

    return replace(
        neutral_parameters(),
        takedown_entry_tendency=0.0,
        takedown_persistence=1.0,
        failed_takedown_persistence=1.0,
        takedown_completion_ability=1.0,
        phase_imposition=1.0,
        clinch_retention=1.0,
    )


def test_zero_propensity_has_zero_distance_sequence_probability() -> None:
    distribution = build_distance_transition_distribution(
        zero_propensity_elite_converter(),
        neutral_parameters(),
    )

    assert sequence_probability(
        distribution,
        FighterSide.RED,
    ) == pytest.approx(0.0)


def test_zero_propensity_has_zero_owner_clinch_sequence_probability() -> None:
    distribution = build_clinch_transition_distribution(
        zero_propensity_elite_converter(),
        neutral_parameters(),
        current_owner=FighterSide.RED,
    )

    assert sequence_probability(
        distribution,
        FighterSide.RED,
    ) == pytest.approx(0.0)


def test_more_propensity_increases_sequence_initiation() -> None:
    neutral = neutral_parameters()
    low = replace(
        neutral,
        takedown_entry_tendency=0.10,
    )
    high = replace(
        neutral,
        takedown_entry_tendency=0.90,
    )

    low_distribution = build_distance_transition_distribution(
        low,
        neutral,
    )
    high_distribution = build_distance_transition_distribution(
        high,
        neutral,
    )

    assert sequence_probability(
        high_distribution,
        FighterSide.RED,
    ) > sequence_probability(
        low_distribution,
        FighterSide.RED,
    )


def test_conversion_skill_changes_success_not_sequence_frequency() -> None:
    """At fixed propensity, skill only changes terminal success/failure mix."""

    neutral = neutral_parameters()
    low = replace(
        neutral,
        takedown_completion_ability=0.10,
    )
    high = replace(
        neutral,
        takedown_completion_ability=0.90,
    )

    low_distribution = build_distance_transition_distribution(
        low,
        neutral,
    )
    high_distribution = build_distance_transition_distribution(
        high,
        neutral,
    )

    assert sequence_probability(
        high_distribution,
        FighterSide.RED,
    ) == pytest.approx(
        sequence_probability(
            low_distribution,
            FighterSide.RED,
        )
    )
    assert high_distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) > low_distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    )


def test_failed_shot_persistence_changes_attempts_not_sequence_frequency() -> None:
    """Chain persistence creates repeat shots without creating new initiations."""

    neutral = neutral_parameters()
    low = replace(
        neutral,
        takedown_persistence=0.05,
        failed_takedown_persistence=0.05,
    )
    high = replace(
        neutral,
        takedown_persistence=0.95,
        failed_takedown_persistence=0.95,
    )

    low_distribution = build_distance_transition_distribution(
        low,
        neutral,
    )
    high_distribution = build_distance_transition_distribution(
        high,
        neutral,
    )

    assert sequence_probability(
        high_distribution,
        FighterSide.RED,
    ) == pytest.approx(
        sequence_probability(
            low_distribution,
            FighterSide.RED,
        )
    )
    assert expected_attempts_from_distribution(
        high_distribution,
        FighterSide.RED,
    ) > expected_attempts_from_distribution(
        low_distribution,
        FighterSide.RED,
    )


def test_chain_distribution_contains_multi_attempt_terminal_outcomes() -> None:
    """Strong persistence must expose attempt-count-specific outcomes."""

    attacker = replace(
        neutral_parameters(),
        takedown_persistence=0.95,
        failed_takedown_persistence=0.95,
    )
    distribution = build_distance_transition_distribution(
        attacker,
        neutral_parameters(),
    )

    attempt_counts = {
        option.attempt_count
        for option in distribution.options
        if (
            option.actor is FighterSide.RED
            and option.event in {
                TransitionEvent.TAKEDOWN,
                TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
            }
        )
    }

    assert {1, 2, 3, 4}.issubset(attempt_counts)


def test_successful_third_shot_carries_three_attempts_to_transition() -> None:
    state = SharedFightState(
        phase=FightPhase.DISTANCE,
        phase_owner=None,
        phase_age_segments=3,
        position_quality=0.0,
        round_number=1,
        segment_number=4,
    )
    option = TransitionProbability(
        event=TransitionEvent.TAKEDOWN,
        actor=FighterSide.RED,
        probability=1.0,
        attempt_count=3,
    )

    transition = apply_transition_option(
        state,
        option,
    )

    assert transition.attempt_count == 3
    assert transition.next_state.phase is FightPhase.GROUND
    assert transition.next_state.phase_owner is FighterSide.RED


def test_failed_four_shot_chain_stays_at_distance_and_carries_count() -> None:
    state = SharedFightState(
        phase=FightPhase.DISTANCE,
        phase_owner=None,
        phase_age_segments=3,
        position_quality=0.0,
        round_number=1,
        segment_number=4,
    )
    option = TransitionProbability(
        event=TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
        actor=FighterSide.RED,
        probability=1.0,
        attempt_count=4,
    )

    transition = apply_transition_option(
        state,
        option,
    )

    assert transition.attempt_count == 4
    assert transition.next_state.phase is FightPhase.DISTANCE
    assert transition.next_state.phase_owner is None
    assert transition.next_state.phase_age_segments == 0


def test_failed_clinch_chain_preserves_existing_owner() -> None:
    state = SharedFightState(
        phase=FightPhase.CLINCH,
        phase_owner=FighterSide.RED,
        phase_age_segments=2,
        position_quality=0.4,
        round_number=1,
        segment_number=4,
    )
    option = TransitionProbability(
        event=TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
        actor=FighterSide.BLUE,
        probability=1.0,
        attempt_count=2,
    )

    transition = apply_transition_option(
        state,
        option,
    )

    assert transition.attempt_count == 2
    assert transition.next_state.phase is FightPhase.CLINCH
    assert transition.next_state.phase_owner is FighterSide.RED
    assert transition.next_state.position_quality == pytest.approx(0.4)
    assert transition.next_state.phase_age_segments == 0
