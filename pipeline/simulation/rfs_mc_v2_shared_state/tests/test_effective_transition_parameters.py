"""Tests for V2 temporary effective transition parameters."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    StatePenaltyWeights,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FighterDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_transition_effect_calibration import (
    DynamicTransitionEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.effective_transition_parameters import (
    TransitionCapabilityMultipliers,
    build_effective_transition_parameters,
    calculate_transition_capability_multiplier,
    calculate_transition_capability_multipliers,
    calculate_transition_fatigue_effect_multiplier,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


MULTIPLIER_FIELDS = [
    "entry",
    "completion",
    "retention",
    "escape",
    "reversal",
    "persistence",
    "imposition",
    "resistance",
]


def dynamic_parameters(
    **overrides: float,
) -> FighterDynamicParameters:
    """Build valid fighter dynamic-response parameters."""

    values = {
        "fatigue_accumulation_resistance": 0.50,
        "fatigue_performance_resilience": 0.50,
        "recovery_ability": 0.50,
        "damage_resistance": 0.50,
        "acute_stress_resistance": 0.50,
        "acute_stress_recovery": 0.50,
    }
    values.update(overrides)

    return FighterDynamicParameters(**values)


def effect_calibration() -> DynamicTransitionEffectCalibration:
    """Build a controlled transition-effect calibration."""

    return DynamicTransitionEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_transition_multiplier=0.15,
        entry=StatePenaltyWeights(
            fatigue=0.45,
            damage=0.20,
            acute_stress=0.30,
        ),
        completion=StatePenaltyWeights(
            fatigue=0.55,
            damage=0.30,
            acute_stress=0.25,
        ),
        retention=StatePenaltyWeights(
            fatigue=0.50,
            damage=0.25,
            acute_stress=0.20,
        ),
        escape=StatePenaltyWeights(
            fatigue=0.45,
            damage=0.40,
            acute_stress=0.30,
        ),
        reversal=StatePenaltyWeights(
            fatigue=0.55,
            damage=0.35,
            acute_stress=0.25,
        ),
        persistence=StatePenaltyWeights(
            fatigue=0.60,
            damage=0.20,
            acute_stress=0.25,
        ),
        imposition=StatePenaltyWeights(
            fatigue=0.50,
            damage=0.25,
            acute_stress=0.30,
        ),
        resistance=StatePenaltyWeights(
            fatigue=0.40,
            damage=0.45,
            acute_stress=0.35,
        ),
    )


def baseline_transition_parameters() -> FighterTransitionParameters:
    """Build transition parameters with easy-to-audit values."""

    return FighterTransitionParameters(
        distance_retention=0.80,
        clinch_entry_tendency=0.70,
        clinch_entry_resistance=0.60,
        takedown_entry_tendency=0.50,
        takedown_completion_ability=0.40,
        takedown_resistance=0.90,
        takedown_persistence=0.80,
        failed_takedown_persistence=0.70,
        clinch_retention=0.60,
        clinch_escape_ability=0.50,
        ground_retention=0.40,
        ground_escape_ability=0.30,
        reversal_ability=0.20,
        phase_imposition=0.90,
        phase_resistance=0.80,
    )


def dynamic_state(
    *,
    fatigue: float = 0.50,
    damage: float = 0.40,
    acute_stress: float = 0.30,
) -> FighterDynamicState:
    """Build a controlled fighter dynamic state."""

    return FighterDynamicState(
        fatigue=fatigue,
        damage=damage,
        acute_stress=acute_stress,
    )


def transition_multipliers(
    **overrides: float,
) -> TransitionCapabilityMultipliers:
    """Build valid transition capability multipliers."""

    values = {
        "entry": 0.80,
        "completion": 0.75,
        "retention": 0.70,
        "escape": 0.65,
        "reversal": 0.60,
        "persistence": 0.55,
        "imposition": 0.50,
        "resistance": 0.45,
    }
    values.update(overrides)

    return TransitionCapabilityMultipliers(**values)


def test_valid_transition_multipliers_are_retained() -> None:
    multipliers = transition_multipliers()

    assert multipliers.entry == 0.80
    assert multipliers.completion == 0.75
    assert multipliers.retention == 0.70
    assert multipliers.escape == 0.65
    assert multipliers.reversal == 0.60
    assert multipliers.persistence == 0.55
    assert multipliers.imposition == 0.50
    assert multipliers.resistance == 0.45


def test_transition_multiplier_boundaries_are_allowed() -> None:
    low = TransitionCapabilityMultipliers(
        entry=0.0,
        completion=0.0,
        retention=0.0,
        escape=0.0,
        reversal=0.0,
        persistence=0.0,
        imposition=0.0,
        resistance=0.0,
    )
    high = TransitionCapabilityMultipliers(
        entry=1.0,
        completion=1.0,
        retention=1.0,
        escape=1.0,
        reversal=1.0,
        persistence=1.0,
        imposition=1.0,
        resistance=1.0,
    )

    assert low.entry == 0.0
    assert high.entry == 1.0


@pytest.mark.parametrize(
    "field_name",
    MULTIPLIER_FIELDS,
)
def test_transition_multipliers_must_be_numeric(
    field_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        transition_multipliers(
            **{
                field_name: "invalid",
            }
        )


@pytest.mark.parametrize(
    "field_name",
    MULTIPLIER_FIELDS,
)
def test_transition_multipliers_must_be_finite(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        transition_multipliers(
            **{
                field_name: float("nan"),
            }
        )


@pytest.mark.parametrize(
    "field_name",
    MULTIPLIER_FIELDS,
)
def test_transition_multipliers_cannot_be_negative(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        transition_multipliers(
            **{
                field_name: -0.01,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    MULTIPLIER_FIELDS,
)
def test_transition_multipliers_cannot_exceed_one(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        transition_multipliers(
            **{
                field_name: 1.01,
            }
        )


def test_transition_multipliers_are_immutable() -> None:
    multipliers = transition_multipliers()

    with pytest.raises(FrozenInstanceError):
        multipliers.entry = 0.10


def test_zero_resilience_receives_full_fatigue_effect() -> None:
    result = calculate_transition_fatigue_effect_multiplier(
        dynamic_parameters(
            fatigue_performance_resilience=0.0,
        ),
        effect_calibration(),
    )

    assert result == pytest.approx(1.0)


def test_max_resilience_receives_minimum_fatigue_effect() -> None:
    result = calculate_transition_fatigue_effect_multiplier(
        dynamic_parameters(
            fatigue_performance_resilience=1.0,
        ),
        effect_calibration(),
    )

    assert result == pytest.approx(0.20)


def test_midpoint_resilience_interpolates_linearly() -> None:
    result = calculate_transition_fatigue_effect_multiplier(
        dynamic_parameters(
            fatigue_performance_resilience=0.50,
        ),
        effect_calibration(),
    )

    assert result == pytest.approx(0.60)


@pytest.mark.parametrize(
    (
        "selected_parameters",
        "selected_calibration",
        "expected_message",
    ),
    [
        (
            "invalid",
            effect_calibration(),
            "parameters must be FighterDynamicParameters",
        ),
        (
            dynamic_parameters(),
            "invalid",
            (
                "calibration must be "
                "DynamicTransitionEffectCalibration"
            ),
        ),
    ],
)
def test_transition_fatigue_effect_requires_correct_types(
    selected_parameters: object,
    selected_calibration: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        calculate_transition_fatigue_effect_multiplier(
            selected_parameters,
            selected_calibration,
        )


def test_transition_multiplier_uses_exact_penalty_arithmetic() -> None:
    result = calculate_transition_capability_multiplier(
        dynamic_state(),
        dynamic_parameters(),
        effect_calibration().entry,
        effect_calibration(),
    )

    # Fatigue effect:
    # 1 - 0.50 resilience * (1 - 0.20 floor) = 0.60
    #
    # Fatigue penalty:
    # 0.50 fatigue * 0.60 effect * 0.45 weight = 0.135
    #
    # Damage penalty:
    # 0.40 damage * 0.20 weight = 0.080
    #
    # Acute-stress penalty:
    # 0.30 stress * 0.30 weight = 0.090
    #
    # Transition multiplier:
    # 1 - (0.135 + 0.080 + 0.090) = 0.695

    assert result == pytest.approx(0.695)


def test_fresh_state_returns_full_transition_capability() -> None:
    result = calculate_transition_capability_multiplier(
        FighterDynamicState.opening_state(),
        dynamic_parameters(),
        effect_calibration().entry,
        effect_calibration(),
    )

    assert result == pytest.approx(1.0)


def test_transition_multiplier_respects_floor() -> None:
    calibration = effect_calibration()

    result = calculate_transition_capability_multiplier(
        FighterDynamicState(
            fatigue=1.0,
            damage=1.0,
            acute_stress=1.0,
        ),
        dynamic_parameters(
            fatigue_performance_resilience=0.0,
        ),
        StatePenaltyWeights(
            fatigue=1.0,
            damage=1.0,
            acute_stress=1.0,
        ),
        calibration,
    )

    assert result == pytest.approx(
        calibration.minimum_effective_transition_multiplier
    )


def test_higher_resilience_reduces_transition_fatigue_penalty() -> None:
    state = dynamic_state(
        fatigue=1.0,
        damage=0.0,
        acute_stress=0.0,
    )
    weights = StatePenaltyWeights(
        fatigue=1.0,
        damage=0.0,
        acute_stress=0.0,
    )

    low_resilience = calculate_transition_capability_multiplier(
        state,
        dynamic_parameters(
            fatigue_performance_resilience=0.0,
        ),
        weights,
        effect_calibration(),
    )
    high_resilience = calculate_transition_capability_multiplier(
        state,
        dynamic_parameters(
            fatigue_performance_resilience=1.0,
        ),
        weights,
        effect_calibration(),
    )

    assert low_resilience == pytest.approx(0.15)
    assert high_resilience == pytest.approx(0.80)
    assert high_resilience > low_resilience


def test_resilience_does_not_change_transition_damage_penalty() -> None:
    state = dynamic_state(
        fatigue=0.0,
        damage=0.50,
        acute_stress=0.0,
    )
    weights = StatePenaltyWeights(
        fatigue=0.0,
        damage=0.50,
        acute_stress=0.0,
    )

    low_resilience = calculate_transition_capability_multiplier(
        state,
        dynamic_parameters(
            fatigue_performance_resilience=0.0,
        ),
        weights,
        effect_calibration(),
    )
    high_resilience = calculate_transition_capability_multiplier(
        state,
        dynamic_parameters(
            fatigue_performance_resilience=1.0,
        ),
        weights,
        effect_calibration(),
    )

    assert low_resilience == pytest.approx(0.75)
    assert high_resilience == pytest.approx(0.75)


@pytest.mark.parametrize(
    (
        "selected_state",
        "selected_parameters",
        "selected_weights",
        "selected_calibration",
        "expected_message",
    ),
    [
        (
            "invalid",
            dynamic_parameters(),
            effect_calibration().entry,
            effect_calibration(),
            "state must be FighterDynamicState",
        ),
        (
            dynamic_state(),
            "invalid",
            effect_calibration().entry,
            effect_calibration(),
            "parameters must be FighterDynamicParameters",
        ),
        (
            dynamic_state(),
            dynamic_parameters(),
            "invalid",
            effect_calibration(),
            "weights must be StatePenaltyWeights",
        ),
        (
            dynamic_state(),
            dynamic_parameters(),
            effect_calibration().entry,
            "invalid",
            (
                "calibration must be "
                "DynamicTransitionEffectCalibration"
            ),
        ),
    ],
)
def test_transition_capability_calculation_requires_types(
    selected_state: object,
    selected_parameters: object,
    selected_weights: object,
    selected_calibration: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        calculate_transition_capability_multiplier(
            selected_state,
            selected_parameters,
            selected_weights,
            selected_calibration,
        )


def test_all_transition_families_use_their_own_weights() -> None:
    multipliers = calculate_transition_capability_multipliers(
        dynamic_state(),
        dynamic_parameters(),
        effect_calibration(),
    )

    assert multipliers.entry == pytest.approx(0.695)
    assert multipliers.completion == pytest.approx(0.640)
    assert multipliers.retention == pytest.approx(0.690)
    assert multipliers.escape == pytest.approx(0.615)
    assert multipliers.reversal == pytest.approx(0.620)
    assert multipliers.persistence == pytest.approx(0.665)
    assert multipliers.imposition == pytest.approx(0.660)
    assert multipliers.resistance == pytest.approx(0.595)


def test_fresh_state_preserves_transition_baseline() -> None:
    baseline = baseline_transition_parameters()

    effective = build_effective_transition_parameters(
        baseline,
        FighterDynamicState.opening_state(),
        dynamic_parameters(),
        effect_calibration(),
    )

    assert effective == baseline


def test_effective_transition_parameters_use_correct_families() -> None:
    baseline = baseline_transition_parameters()

    effective = build_effective_transition_parameters(
        baseline,
        dynamic_state(),
        dynamic_parameters(),
        effect_calibration(),
    )

    assert effective.distance_retention == pytest.approx(
        0.80 * 0.690
    )

    assert effective.clinch_entry_tendency == pytest.approx(
        0.70 * 0.695
    )
    assert effective.clinch_entry_resistance == pytest.approx(
        0.60 * 0.595
    )

    assert effective.takedown_entry_tendency == pytest.approx(
        0.50 * 0.695
    )
    assert effective.takedown_completion_ability == pytest.approx(
        0.40 * 0.640
    )
    assert effective.takedown_resistance == pytest.approx(
        0.90 * 0.595
    )

    assert effective.takedown_persistence == pytest.approx(
        0.80 * 0.665
    )
    assert effective.failed_takedown_persistence == pytest.approx(
        0.70 * 0.665
    )

    assert effective.clinch_retention == pytest.approx(
        0.60 * 0.690
    )
    assert effective.clinch_escape_ability == pytest.approx(
        0.50 * 0.615
    )

    assert effective.ground_retention == pytest.approx(
        0.40 * 0.690
    )
    assert effective.ground_escape_ability == pytest.approx(
        0.30 * 0.615
    )
    assert effective.reversal_ability == pytest.approx(
        0.20 * 0.620
    )

    assert effective.phase_imposition == pytest.approx(
        0.90 * 0.660
    )
    assert effective.phase_resistance == pytest.approx(
        0.80 * 0.595
    )


def test_effective_transition_generation_does_not_mutate_baseline() -> None:
    baseline = baseline_transition_parameters()
    original = baseline_transition_parameters()

    effective = build_effective_transition_parameters(
        baseline,
        dynamic_state(),
        dynamic_parameters(),
        effect_calibration(),
    )

    assert baseline == original
    assert effective is not baseline
    assert effective != baseline


def test_severe_state_reduces_every_transition_capability() -> None:
    baseline = baseline_transition_parameters()

    fresh = build_effective_transition_parameters(
        baseline,
        FighterDynamicState.opening_state(),
        dynamic_parameters(),
        effect_calibration(),
    )
    impaired = build_effective_transition_parameters(
        baseline,
        FighterDynamicState(
            fatigue=1.0,
            damage=1.0,
            acute_stress=1.0,
        ),
        dynamic_parameters(),
        effect_calibration(),
    )

    for field_name in baseline.__dataclass_fields__:
        assert (
            getattr(impaired, field_name)
            < getattr(fresh, field_name)
        )


def test_effective_generation_requires_transition_baseline() -> None:
    with pytest.raises(
        TypeError,
        match="baseline must be FighterTransitionParameters",
    ):
        build_effective_transition_parameters(
            "invalid",
            dynamic_state(),
            dynamic_parameters(),
            effect_calibration(),
        )


@pytest.mark.parametrize(
    (
        "selected_state",
        "selected_parameters",
        "selected_calibration",
        "expected_message",
    ),
    [
        (
            "invalid",
            dynamic_parameters(),
            effect_calibration(),
            "state must be FighterDynamicState",
        ),
        (
            dynamic_state(),
            "invalid",
            effect_calibration(),
            "parameters must be FighterDynamicParameters",
        ),
        (
            dynamic_state(),
            dynamic_parameters(),
            "invalid",
            (
                "calibration must be "
                "DynamicTransitionEffectCalibration"
            ),
        ),
    ],
)
def test_all_transition_multiplier_calculation_requires_types(
    selected_state: object,
    selected_parameters: object,
    selected_calibration: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        calculate_transition_capability_multipliers(
            selected_state,
            selected_parameters,
            selected_calibration,
        )
