"""Regression tests for two-stage takedown transition semantics."""

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


def attempt_probability(distribution, side: FighterSide) -> float:
    """Return success + failure probability for one fighter's TD attempt."""

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


def test_neutral_distance_attempt_mass_is_preserved() -> None:
    """Neutral initiation keeps the pre-split 0.75 relative attempt weight."""

    distribution = build_distance_transition_distribution(
        neutral_parameters(),
        neutral_parameters(),
    )

    total_weight = 6.0 + 1.0 + 1.0 + 0.75 + 0.75
    expected_attempt = 0.75 / total_weight

    assert attempt_probability(
        distribution,
        FighterSide.RED,
    ) == pytest.approx(expected_attempt)
    assert attempt_probability(
        distribution,
        FighterSide.BLUE,
    ) == pytest.approx(expected_attempt)

    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) == pytest.approx(expected_attempt * 0.36)
    assert distribution.probability(
        TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
        FighterSide.RED,
    ) == pytest.approx(expected_attempt * 0.64)


def test_neutral_clinch_attempt_mass_is_preserved() -> None:
    """Owner/defender neutral attempt masses remain 15% and 5%."""

    distribution = build_clinch_transition_distribution(
        neutral_parameters(),
        neutral_parameters(),
        current_owner=FighterSide.RED,
    )

    assert attempt_probability(
        distribution,
        FighterSide.RED,
    ) == pytest.approx(0.15)
    assert attempt_probability(
        distribution,
        FighterSide.BLUE,
    ) == pytest.approx(0.05)

    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) == pytest.approx(0.15 * 0.36)
    assert distribution.probability(
        TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
        FighterSide.RED,
    ) == pytest.approx(0.15 * 0.64)


def zero_propensity_elite_converter() -> FighterTransitionParameters:
    """Strong success skills cannot create a shot without initiation."""

    return replace(
        neutral_parameters(),
        takedown_entry_tendency=0.0,
        takedown_persistence=0.0,
        failed_takedown_persistence=0.0,
        takedown_completion_ability=1.0,
        phase_imposition=1.0,
        clinch_retention=1.0,
    )


def test_zero_propensity_has_zero_distance_attempt_probability() -> None:
    distribution = build_distance_transition_distribution(
        zero_propensity_elite_converter(),
        neutral_parameters(),
    )

    assert attempt_probability(
        distribution,
        FighterSide.RED,
    ) == pytest.approx(0.0)


def test_zero_propensity_has_zero_owner_clinch_attempt_probability() -> None:
    distribution = build_clinch_transition_distribution(
        zero_propensity_elite_converter(),
        neutral_parameters(),
        current_owner=FighterSide.RED,
    )

    assert attempt_probability(
        distribution,
        FighterSide.RED,
    ) == pytest.approx(0.0)


def test_more_propensity_increases_attempts() -> None:
    neutral = neutral_parameters()
    low = replace(
        neutral,
        takedown_entry_tendency=0.10,
        takedown_persistence=0.10,
        failed_takedown_persistence=0.10,
    )
    high = replace(
        neutral,
        takedown_entry_tendency=0.90,
        takedown_persistence=0.90,
        failed_takedown_persistence=0.90,
    )

    low_distribution = build_distance_transition_distribution(
        low,
        neutral,
    )
    high_distribution = build_distance_transition_distribution(
        high,
        neutral,
    )

    assert attempt_probability(
        high_distribution,
        FighterSide.RED,
    ) > attempt_probability(
        low_distribution,
        FighterSide.RED,
    )


def test_conversion_skill_changes_success_not_attempt_frequency() -> None:
    """At fixed propensity, skill only changes the success/failure split."""

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

    assert attempt_probability(
        high_distribution,
        FighterSide.RED,
    ) == pytest.approx(
        attempt_probability(
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
    assert high_distribution.probability(
        TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
        FighterSide.RED,
    ) < low_distribution.probability(
        TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
        FighterSide.RED,
    )


def test_failed_distance_shot_stays_at_distance_and_resets_age() -> None:
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
    )

    transition = apply_transition_option(
        state,
        option,
    )

    assert transition.next_state.phase is FightPhase.DISTANCE
    assert transition.next_state.phase_owner is None
    assert transition.next_state.phase_age_segments == 0


def test_failed_clinch_shot_preserves_existing_owner() -> None:
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
    )

    transition = apply_transition_option(
        state,
        option,
    )

    assert transition.next_state.phase is FightPhase.CLINCH
    assert transition.next_state.phase_owner is FighterSide.RED
    assert transition.next_state.position_quality == pytest.approx(0.4)
    assert transition.next_state.phase_age_segments == 0
