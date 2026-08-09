"""Regression tests for propensity-gated completed-takedown transitions."""

from dataclasses import replace

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
    build_clinch_transition_distribution,
    build_distance_transition_distribution,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
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


def test_neutral_distance_calibration_is_preserved_exactly() -> None:
    """Factorization must not move the documented neutral base distribution."""

    distribution = build_distance_transition_distribution(
        neutral_parameters(),
        neutral_parameters(),
    )

    total_weight = 6.0 + 1.0 + 1.0 + 0.75 + 0.75

    assert distribution.probability(
        TransitionEvent.STAY,
        None,
    ) == pytest.approx(6.0 / total_weight)
    assert distribution.probability(
        TransitionEvent.CLINCH_ENTRY,
        FighterSide.RED,
    ) == pytest.approx(1.0 / total_weight)
    assert distribution.probability(
        TransitionEvent.CLINCH_ENTRY,
        FighterSide.BLUE,
    ) == pytest.approx(1.0 / total_weight)
    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) == pytest.approx(0.75 / total_weight)
    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.BLUE,
    ) == pytest.approx(0.75 / total_weight)


def test_neutral_clinch_calibration_is_preserved_exactly() -> None:
    """Neutral owner/defender takedown weights remain 15% and 5%."""

    distribution = build_clinch_transition_distribution(
        neutral_parameters(),
        neutral_parameters(),
        current_owner=FighterSide.RED,
    )

    assert distribution.probability(
        TransitionEvent.STAY,
        None,
    ) == pytest.approx(0.45)
    assert distribution.probability(
        TransitionEvent.CLINCH_BREAK,
        None,
    ) == pytest.approx(0.25)
    assert distribution.probability(
        TransitionEvent.OWNERSHIP_CHANGE,
        FighterSide.BLUE,
    ) == pytest.approx(0.10)
    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) == pytest.approx(0.15)
    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.BLUE,
    ) == pytest.approx(0.05)


def zero_propensity_elite_converter() -> FighterTransitionParameters:
    """Strong success skills cannot create a takedown without initiation."""

    return replace(
        neutral_parameters(),
        takedown_entry_tendency=0.0,
        takedown_persistence=0.0,
        failed_takedown_persistence=0.0,
        takedown_completion_ability=1.0,
        phase_imposition=1.0,
        clinch_retention=1.0,
    )


def test_zero_propensity_has_zero_distance_takedown_probability() -> None:
    distribution = build_distance_transition_distribution(
        zero_propensity_elite_converter(),
        neutral_parameters(),
    )

    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) == pytest.approx(0.0)


def test_zero_propensity_has_zero_owner_clinch_takedown_probability() -> None:
    distribution = build_clinch_transition_distribution(
        zero_propensity_elite_converter(),
        neutral_parameters(),
        current_owner=FighterSide.RED,
    )

    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) == pytest.approx(0.0)


def test_zero_propensity_has_zero_defender_clinch_takedown_probability() -> None:
    distribution = build_clinch_transition_distribution(
        neutral_parameters(),
        zero_propensity_elite_converter(),
        current_owner=FighterSide.RED,
    )

    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.BLUE,
    ) == pytest.approx(0.0)


def test_more_propensity_increases_completed_takedown_probability() -> None:
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

    assert high_distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) > low_distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    )


def test_more_conversion_skill_increases_takedown_when_propensity_exists() -> None:
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

    assert high_distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) > low_distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    )
