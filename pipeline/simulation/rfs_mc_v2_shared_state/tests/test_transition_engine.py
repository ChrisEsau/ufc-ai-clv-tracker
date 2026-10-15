"""Tests for V2 shared transition-probability generation."""

from dataclasses import replace
from math import fsum

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
    DistanceTransitionCalibration,
    build_distance_transition_distribution,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


def parameters(
    **overrides: float,
) -> FighterTransitionParameters:
    """Build a neutral transition profile with optional changes."""

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


def test_distance_probabilities_sum_to_one() -> None:
    distribution = build_distance_transition_distribution(
        parameters(),
        parameters(),
    )

    total = fsum(
        option.probability
        for option in distribution.options
    )

    assert total == pytest.approx(1.0, abs=1e-12)


def test_neutral_matchup_is_symmetric() -> None:
    distribution = build_distance_transition_distribution(
        parameters(),
        parameters(),
    )

    assert distribution.probability(
        TransitionEvent.CLINCH_ENTRY,
        FighterSide.RED,
    ) == pytest.approx(
        distribution.probability(
            TransitionEvent.CLINCH_ENTRY,
            FighterSide.BLUE,
        )
    )

    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) == pytest.approx(
        distribution.probability(
            TransitionEvent.TAKEDOWN,
            FighterSide.BLUE,
        )
    )


def test_neutral_matchup_most_often_stays_at_distance() -> None:
    distribution = build_distance_transition_distribution(
        parameters(),
        parameters(),
    )

    stay = distribution.probability(
        TransitionEvent.STAY,
        None,
    )
    red_clinch = distribution.probability(
        TransitionEvent.CLINCH_ENTRY,
        FighterSide.RED,
    )
    red_takedown = distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    )

    assert stay > red_clinch
    assert red_clinch > red_takedown


def test_stronger_red_clinch_matchup_increases_entry_probability() -> None:
    baseline = build_distance_transition_distribution(
        parameters(),
        parameters(),
    )

    favorable = build_distance_transition_distribution(
        parameters(
            clinch_entry_tendency=0.95,
            phase_imposition=0.90,
        ),
        parameters(
            clinch_entry_resistance=0.10,
            phase_resistance=0.10,
        ),
    )

    assert favorable.probability(
        TransitionEvent.CLINCH_ENTRY,
        FighterSide.RED,
    ) > baseline.probability(
        TransitionEvent.CLINCH_ENTRY,
        FighterSide.RED,
    )


def test_stronger_red_takedown_matchup_increases_probability() -> None:
    baseline = build_distance_transition_distribution(
        parameters(),
        parameters(),
    )

    favorable = build_distance_transition_distribution(
        parameters(
            takedown_entry_tendency=0.95,
            takedown_completion_ability=0.90,
            takedown_persistence=0.90,
            failed_takedown_persistence=0.85,
            phase_imposition=0.90,
        ),
        parameters(
            takedown_resistance=0.10,
            phase_resistance=0.10,
        ),
    )

    assert favorable.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) > baseline.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    )


def test_distance_traits_increase_stay_probability() -> None:
    baseline = build_distance_transition_distribution(
        parameters(),
        parameters(),
    )

    distance_fight = build_distance_transition_distribution(
        parameters(
            distance_retention=0.95,
            phase_resistance=0.95,
        ),
        parameters(
            distance_retention=0.95,
            phase_resistance=0.95,
        ),
    )

    assert distance_fight.probability(
        TransitionEvent.STAY,
        None,
    ) > baseline.probability(
        TransitionEvent.STAY,
        None,
    )


def test_swapping_fighters_swaps_directional_probabilities() -> None:
    red = parameters(
        clinch_entry_tendency=0.80,
        takedown_entry_tendency=0.70,
        phase_imposition=0.75,
    )
    blue = parameters(
        clinch_entry_resistance=0.65,
        takedown_resistance=0.70,
        phase_resistance=0.60,
    )

    original = build_distance_transition_distribution(
        red,
        blue,
    )
    swapped = build_distance_transition_distribution(
        blue,
        red,
    )

    assert original.probability(
        TransitionEvent.CLINCH_ENTRY,
        FighterSide.RED,
    ) == pytest.approx(
        swapped.probability(
            TransitionEvent.CLINCH_ENTRY,
            FighterSide.BLUE,
        )
    )

    assert original.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) == pytest.approx(
        swapped.probability(
            TransitionEvent.TAKEDOWN,
            FighterSide.BLUE,
        )
    )

    assert original.probability(
        TransitionEvent.STAY,
        None,
    ) == pytest.approx(
        swapped.probability(
            TransitionEvent.STAY,
            None,
        )
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("stay_base_weight", 0.0),
        ("clinch_entry_base_weight", -1.0),
        ("takedown_base_weight", float("nan")),
        ("matchup_effect_strength", -0.01),
    ],
)
def test_distance_calibration_is_validated(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        replace(
            DistanceTransitionCalibration(),
            **{field_name: value},
        )


@pytest.mark.parametrize(
    "current_owner",
    [
        FighterSide.RED,
        FighterSide.BLUE,
    ],
)
def test_clinch_probabilities_sum_to_one(
    current_owner: FighterSide,
) -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_clinch_transition_distribution,
    )

    distribution = build_clinch_transition_distribution(
        parameters(),
        parameters(),
        current_owner=current_owner,
    )

    assert fsum(
        option.probability
        for option in distribution.options
    ) == pytest.approx(1.0, abs=1e-12)


def test_neutral_clinch_favors_owner_takedown() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_clinch_transition_distribution,
    )

    distribution = build_clinch_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) > distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.BLUE,
    )


def test_neutral_clinch_most_often_stays_with_owner() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_clinch_transition_distribution,
    )

    distribution = build_clinch_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    stay = distribution.probability(
        TransitionEvent.STAY,
        None,
    )

    assert stay == max(
        option.probability
        for option in distribution.options
    )


def test_owner_retention_increases_clinch_stay() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_clinch_transition_distribution,
    )

    baseline = build_clinch_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    favorable = build_clinch_transition_distribution(
        parameters(
            clinch_retention=0.95,
            phase_imposition=0.90,
        ),
        parameters(
            clinch_escape_ability=0.10,
            phase_resistance=0.15,
        ),
        current_owner=FighterSide.RED,
    )

    assert favorable.probability(
        TransitionEvent.STAY,
        None,
    ) > baseline.probability(
        TransitionEvent.STAY,
        None,
    )


def test_defender_escape_increases_clinch_break() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_clinch_transition_distribution,
    )

    baseline = build_clinch_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    favorable = build_clinch_transition_distribution(
        parameters(
            clinch_retention=0.10,
            phase_imposition=0.20,
        ),
        parameters(
            clinch_escape_ability=0.95,
            phase_resistance=0.90,
        ),
        current_owner=FighterSide.RED,
    )

    assert favorable.probability(
        TransitionEvent.CLINCH_BREAK,
        None,
    ) > baseline.probability(
        TransitionEvent.CLINCH_BREAK,
        None,
    )


def test_defender_imposition_increases_ownership_change() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_clinch_transition_distribution,
    )

    baseline = build_clinch_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    favorable = build_clinch_transition_distribution(
        parameters(
            clinch_retention=0.10,
            phase_resistance=0.10,
        ),
        parameters(
            phase_imposition=0.95,
            clinch_retention=0.90,
        ),
        current_owner=FighterSide.RED,
    )

    assert favorable.probability(
        TransitionEvent.OWNERSHIP_CHANGE,
        FighterSide.BLUE,
    ) > baseline.probability(
        TransitionEvent.OWNERSHIP_CHANGE,
        FighterSide.BLUE,
    )


def test_owner_takedown_traits_increase_clinch_takedown() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_clinch_transition_distribution,
    )

    baseline = build_clinch_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    favorable = build_clinch_transition_distribution(
        parameters(
            takedown_entry_tendency=0.95,
            takedown_completion_ability=0.95,
            takedown_persistence=0.90,
            failed_takedown_persistence=0.90,
            clinch_retention=0.90,
            phase_imposition=0.90,
        ),
        parameters(
            takedown_resistance=0.10,
            phase_resistance=0.15,
        ),
        current_owner=FighterSide.RED,
    )

    assert favorable.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) > baseline.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    )


def test_swapping_clinch_owner_preserves_symmetry() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_clinch_transition_distribution,
    )

    red = parameters(
        clinch_retention=0.80,
        takedown_completion_ability=0.75,
        phase_imposition=0.70,
    )
    blue = parameters(
        clinch_escape_ability=0.65,
        takedown_resistance=0.70,
        phase_resistance=0.60,
    )

    original = build_clinch_transition_distribution(
        red,
        blue,
        current_owner=FighterSide.RED,
    )
    swapped = build_clinch_transition_distribution(
        blue,
        red,
        current_owner=FighterSide.BLUE,
    )

    assert original.probability(
        TransitionEvent.STAY,
        None,
    ) == pytest.approx(
        swapped.probability(
            TransitionEvent.STAY,
            None,
        )
    )

    assert original.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.RED,
    ) == pytest.approx(
        swapped.probability(
            TransitionEvent.TAKEDOWN,
            FighterSide.BLUE,
        )
    )

    assert original.probability(
        TransitionEvent.OWNERSHIP_CHANGE,
        FighterSide.BLUE,
    ) == pytest.approx(
        swapped.probability(
            TransitionEvent.OWNERSHIP_CHANGE,
            FighterSide.RED,
        )
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("stay_base_weight", 0.0),
        ("break_base_weight", -1.0),
        ("ownership_change_base_weight", float("nan")),
        ("owner_takedown_base_weight", 0.0),
        ("defender_takedown_base_weight", -0.01),
        ("matchup_effect_strength", -0.01),
    ],
)
def test_clinch_calibration_is_validated(
    field_name: str,
    value: float,
) -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        ClinchTransitionCalibration,
    )

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        replace(
            ClinchTransitionCalibration(),
            **{field_name: value},
        )


@pytest.mark.parametrize(
    "current_owner",
    [
        FighterSide.RED,
        FighterSide.BLUE,
    ],
)
def test_ground_probabilities_sum_to_one(
    current_owner: FighterSide,
) -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_ground_transition_distribution,
    )

    distribution = build_ground_transition_distribution(
        parameters(),
        parameters(),
        current_owner=current_owner,
    )

    assert fsum(
        option.probability
        for option in distribution.options
    ) == pytest.approx(1.0, abs=1e-12)


def test_neutral_ground_most_often_stays_with_owner() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_ground_transition_distribution,
    )

    distribution = build_ground_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    stay = distribution.probability(
        TransitionEvent.STAY,
        None,
    )

    assert stay == max(
        option.probability
        for option in distribution.options
    )


def test_neutral_ground_defensive_outcomes_are_ordered() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_ground_transition_distribution,
    )

    distribution = build_ground_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    escape = distribution.probability(
        TransitionEvent.GROUND_ESCAPE,
        FighterSide.BLUE,
    )
    scramble = distribution.probability(
        TransitionEvent.SCRAMBLE_TO_CLINCH,
        FighterSide.BLUE,
    )
    reversal = distribution.probability(
        TransitionEvent.REVERSAL,
        FighterSide.BLUE,
    )

    assert escape > scramble > reversal


def test_owner_retention_increases_ground_stay() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_ground_transition_distribution,
    )

    baseline = build_ground_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    favorable = build_ground_transition_distribution(
        parameters(
            ground_retention=0.95,
            phase_imposition=0.90,
            phase_resistance=0.90,
        ),
        parameters(
            ground_escape_ability=0.10,
            reversal_ability=0.10,
        ),
        current_owner=FighterSide.RED,
    )

    assert favorable.probability(
        TransitionEvent.STAY,
        None,
    ) > baseline.probability(
        TransitionEvent.STAY,
        None,
    )


def test_defender_escape_increases_ground_escape() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_ground_transition_distribution,
    )

    baseline = build_ground_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    favorable = build_ground_transition_distribution(
        parameters(
            ground_retention=0.10,
            phase_imposition=0.15,
        ),
        parameters(
            ground_escape_ability=0.95,
            phase_resistance=0.90,
        ),
        current_owner=FighterSide.RED,
    )

    assert favorable.probability(
        TransitionEvent.GROUND_ESCAPE,
        FighterSide.BLUE,
    ) > baseline.probability(
        TransitionEvent.GROUND_ESCAPE,
        FighterSide.BLUE,
    )


def test_defender_scramble_traits_increase_scramble() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_ground_transition_distribution,
    )

    baseline = build_ground_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    favorable = build_ground_transition_distribution(
        parameters(
            ground_retention=0.10,
            phase_resistance=0.15,
        ),
        parameters(
            ground_escape_ability=0.90,
            phase_imposition=0.95,
        ),
        current_owner=FighterSide.RED,
    )

    assert favorable.probability(
        TransitionEvent.SCRAMBLE_TO_CLINCH,
        FighterSide.BLUE,
    ) > baseline.probability(
        TransitionEvent.SCRAMBLE_TO_CLINCH,
        FighterSide.BLUE,
    )


def test_defender_reversal_ability_increases_reversal() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_ground_transition_distribution,
    )

    baseline = build_ground_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    favorable = build_ground_transition_distribution(
        parameters(
            ground_retention=0.10,
            phase_resistance=0.10,
        ),
        parameters(
            reversal_ability=0.95,
            phase_imposition=0.90,
        ),
        current_owner=FighterSide.RED,
    )

    assert favorable.probability(
        TransitionEvent.REVERSAL,
        FighterSide.BLUE,
    ) > baseline.probability(
        TransitionEvent.REVERSAL,
        FighterSide.BLUE,
    )


def test_ground_defensive_actions_belong_to_defender() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_ground_transition_distribution,
    )

    distribution = build_ground_transition_distribution(
        parameters(),
        parameters(),
        current_owner=FighterSide.RED,
    )

    for event in (
        TransitionEvent.GROUND_ESCAPE,
        TransitionEvent.SCRAMBLE_TO_CLINCH,
        TransitionEvent.REVERSAL,
    ):
        assert distribution.probability(
            event,
            FighterSide.BLUE,
        ) > 0.0


def test_swapping_ground_owner_preserves_symmetry() -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        build_ground_transition_distribution,
    )

    red = parameters(
        ground_retention=0.85,
        phase_imposition=0.75,
        phase_resistance=0.70,
    )
    blue = parameters(
        ground_escape_ability=0.70,
        reversal_ability=0.65,
        phase_resistance=0.60,
    )

    original = build_ground_transition_distribution(
        red,
        blue,
        current_owner=FighterSide.RED,
    )
    swapped = build_ground_transition_distribution(
        blue,
        red,
        current_owner=FighterSide.BLUE,
    )

    assert original.probability(
        TransitionEvent.STAY,
        None,
    ) == pytest.approx(
        swapped.probability(
            TransitionEvent.STAY,
            None,
        )
    )

    for event in (
        TransitionEvent.GROUND_ESCAPE,
        TransitionEvent.SCRAMBLE_TO_CLINCH,
        TransitionEvent.REVERSAL,
    ):
        assert original.probability(
            event,
            FighterSide.BLUE,
        ) == pytest.approx(
            swapped.probability(
                event,
                FighterSide.RED,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("stay_base_weight", 0.0),
        ("escape_base_weight", -1.0),
        ("scramble_base_weight", float("nan")),
        ("reversal_base_weight", 0.0),
        ("matchup_effect_strength", -0.01),
    ],
)
def test_ground_calibration_is_validated(
    field_name: str,
    value: float,
) -> None:
    from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
        GroundTransitionCalibration,
    )

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        replace(
            GroundTransitionCalibration(),
            **{field_name: value},
        )
