"""Tests for V2 temporary effective phase parameters."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
    StatePenaltyWeights,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FighterDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.effective_phase_parameters import (
    CapabilityMultipliers,
    build_effective_phase_parameters,
    calculate_capability_multiplier,
    calculate_capability_multipliers,
    calculate_fatigue_effect_multiplier,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)


MULTIPLIER_FIELDS = [
    "output",
    "accuracy",
    "power",
    "control",
    "grappling",
    "defense",
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


def effect_calibration() -> DynamicEffectCalibration:
    """Build a controlled dynamic performance calibration."""

    return DynamicEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_capability_multiplier=0.15,
        output=StatePenaltyWeights(
            fatigue=0.60,
            damage=0.20,
            acute_stress=0.30,
        ),
        accuracy=StatePenaltyWeights(
            fatigue=0.30,
            damage=0.40,
            acute_stress=0.50,
        ),
        power=StatePenaltyWeights(
            fatigue=0.40,
            damage=0.50,
            acute_stress=0.20,
        ),
        control=StatePenaltyWeights(
            fatigue=0.50,
            damage=0.30,
            acute_stress=0.20,
        ),
        grappling=StatePenaltyWeights(
            fatigue=0.45,
            damage=0.35,
            acute_stress=0.25,
        ),
        defense=StatePenaltyWeights(
            fatigue=0.25,
            damage=0.55,
            acute_stress=0.40,
        ),
    )


def baseline_phase_parameters() -> FighterPhaseParameters:
    """Build phase parameters with easy-to-audit values."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=4.0,
            sig_strike_accuracy=0.50,
            knockdown_probability_per_landed=0.10,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=2.0,
            clinch_strike_accuracy=0.60,
            control_seconds_mean=10.0,
            damaging_clinch_probability=0.20,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=3.0,
            ground_strike_accuracy=0.55,
            control_seconds_mean=15.0,
            submission_attempt_rate=0.40,
            position_advancement_probability=0.30,
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=0.50,
            reversal_attempt_rate=0.20,
            scramble_attempt_rate=0.40,
            submission_defense=0.80,
        ),
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


def capability_multipliers(
    **overrides: float,
) -> CapabilityMultipliers:
    """Build valid capability multipliers."""

    values = {
        "output": 0.80,
        "accuracy": 0.75,
        "power": 0.70,
        "control": 0.65,
        "grappling": 0.60,
        "defense": 0.55,
    }
    values.update(overrides)

    return CapabilityMultipliers(**values)


def test_valid_capability_multipliers_are_retained() -> None:
    multipliers = capability_multipliers()

    assert multipliers.output == 0.80
    assert multipliers.accuracy == 0.75
    assert multipliers.power == 0.70
    assert multipliers.control == 0.65
    assert multipliers.grappling == 0.60
    assert multipliers.defense == 0.55


def test_capability_multiplier_boundaries_are_allowed() -> None:
    low = CapabilityMultipliers(
        output=0.0,
        accuracy=0.0,
        power=0.0,
        control=0.0,
        grappling=0.0,
        defense=0.0,
    )
    high = CapabilityMultipliers(
        output=1.0,
        accuracy=1.0,
        power=1.0,
        control=1.0,
        grappling=1.0,
        defense=1.0,
    )

    assert low.output == 0.0
    assert high.output == 1.0


@pytest.mark.parametrize(
    "field_name",
    MULTIPLIER_FIELDS,
)
def test_capability_multipliers_must_be_numeric(
    field_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        capability_multipliers(
            **{
                field_name: "invalid",
            }
        )


@pytest.mark.parametrize(
    "field_name",
    MULTIPLIER_FIELDS,
)
def test_capability_multipliers_must_be_finite(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        capability_multipliers(
            **{
                field_name: float("nan"),
            }
        )


@pytest.mark.parametrize(
    "field_name",
    MULTIPLIER_FIELDS,
)
def test_capability_multipliers_cannot_be_negative(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        capability_multipliers(
            **{
                field_name: -0.01,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    MULTIPLIER_FIELDS,
)
def test_capability_multipliers_cannot_exceed_one(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        capability_multipliers(
            **{
                field_name: 1.01,
            }
        )


def test_capability_multipliers_are_immutable() -> None:
    multipliers = capability_multipliers()

    with pytest.raises(FrozenInstanceError):
        multipliers.output = 0.10


def test_zero_resilience_receives_full_fatigue_effect() -> None:
    result = calculate_fatigue_effect_multiplier(
        dynamic_parameters(
            fatigue_performance_resilience=0.0,
        ),
        effect_calibration(),
    )

    assert result == pytest.approx(1.0)


def test_max_resilience_receives_minimum_fatigue_effect() -> None:
    result = calculate_fatigue_effect_multiplier(
        dynamic_parameters(
            fatigue_performance_resilience=1.0,
        ),
        effect_calibration(),
    )

    assert result == pytest.approx(0.20)


def test_midpoint_resilience_interpolates_linearly() -> None:
    result = calculate_fatigue_effect_multiplier(
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
            "calibration must be DynamicEffectCalibration",
        ),
    ],
)
def test_fatigue_effect_requires_correct_types(
    selected_parameters: object,
    selected_calibration: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        calculate_fatigue_effect_multiplier(
            selected_parameters,
            selected_calibration,
        )


def test_capability_multiplier_uses_exact_penalty_arithmetic() -> None:
    result = calculate_capability_multiplier(
        dynamic_state(),
        dynamic_parameters(),
        effect_calibration().output,
        effect_calibration(),
    )

    # Fatigue effect:
    # 1 - 0.50 resilience * (1 - 0.20 floor) = 0.60
    #
    # Fatigue penalty:
    # 0.50 fatigue * 0.60 effect * 0.60 weight = 0.18
    #
    # Damage penalty:
    # 0.40 damage * 0.20 weight = 0.08
    #
    # Acute stress penalty:
    # 0.30 stress * 0.30 weight = 0.09
    #
    # Capability multiplier:
    # 1 - (0.18 + 0.08 + 0.09) = 0.65

    assert result == pytest.approx(0.65)


def test_fresh_state_returns_full_capability() -> None:
    result = calculate_capability_multiplier(
        FighterDynamicState.opening_state(),
        dynamic_parameters(),
        effect_calibration().output,
        effect_calibration(),
    )

    assert result == pytest.approx(1.0)


def test_capability_multiplier_respects_floor() -> None:
    calibration = effect_calibration()

    result = calculate_capability_multiplier(
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
        calibration.minimum_effective_capability_multiplier
    )


def test_higher_resilience_reduces_fatigue_penalty() -> None:
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

    low_resilience = calculate_capability_multiplier(
        state,
        dynamic_parameters(
            fatigue_performance_resilience=0.0,
        ),
        weights,
        effect_calibration(),
    )
    high_resilience = calculate_capability_multiplier(
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


def test_resilience_does_not_change_damage_penalty() -> None:
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

    low_resilience = calculate_capability_multiplier(
        state,
        dynamic_parameters(
            fatigue_performance_resilience=0.0,
        ),
        weights,
        effect_calibration(),
    )
    high_resilience = calculate_capability_multiplier(
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
            effect_calibration().output,
            effect_calibration(),
            "state must be FighterDynamicState",
        ),
        (
            dynamic_state(),
            "invalid",
            effect_calibration().output,
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
            effect_calibration().output,
            "invalid",
            "calibration must be DynamicEffectCalibration",
        ),
    ],
)
def test_capability_calculation_requires_correct_types(
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
        calculate_capability_multiplier(
            selected_state,
            selected_parameters,
            selected_weights,
            selected_calibration,
        )


def test_all_capability_families_use_their_own_weights() -> None:
    multipliers = calculate_capability_multipliers(
        dynamic_state(),
        dynamic_parameters(),
        effect_calibration(),
    )

    assert multipliers.output == pytest.approx(0.650)
    assert multipliers.accuracy == pytest.approx(0.600)
    assert multipliers.power == pytest.approx(0.620)
    assert multipliers.control == pytest.approx(0.670)
    assert multipliers.grappling == pytest.approx(0.650)
    assert multipliers.defense == pytest.approx(0.585)


def test_fresh_state_preserves_all_baseline_phase_parameters() -> None:
    baseline = baseline_phase_parameters()

    effective = build_effective_phase_parameters(
        baseline,
        FighterDynamicState.opening_state(),
        dynamic_parameters(),
        effect_calibration(),
    )

    assert effective == baseline


def test_effective_phase_parameters_use_correct_capability_families() -> None:
    baseline = baseline_phase_parameters()

    effective = build_effective_phase_parameters(
        baseline,
        dynamic_state(),
        dynamic_parameters(),
        effect_calibration(),
    )

    assert effective.distance.sig_strike_attempt_rate == (
        pytest.approx(
            4.0 * 0.650
        )
    )
    assert effective.distance.sig_strike_accuracy == (
        pytest.approx(
            0.50 * 0.600
        )
    )
    assert (
        effective.distance.knockdown_probability_per_landed
        == pytest.approx(
            0.10 * 0.620
        )
    )

    assert effective.clinch.clinch_strike_attempt_rate == (
        pytest.approx(
            2.0 * 0.650
        )
    )
    assert effective.clinch.clinch_strike_accuracy == (
        pytest.approx(
            0.60 * 0.600
        )
    )
    assert effective.clinch.control_seconds_mean == (
        pytest.approx(
            10.0 * 0.670
        )
    )
    assert effective.clinch.damaging_clinch_probability == (
        pytest.approx(
            0.20 * 0.620
        )
    )

    assert effective.ground_owner.ground_strike_attempt_rate == (
        pytest.approx(
            3.0 * 0.650
        )
    )
    assert effective.ground_owner.ground_strike_accuracy == (
        pytest.approx(
            0.55 * 0.600
        )
    )
    assert effective.ground_owner.control_seconds_mean == (
        pytest.approx(
            15.0 * 0.670
        )
    )
    assert effective.ground_owner.submission_attempt_rate == (
        pytest.approx(
            0.40 * 0.650
        )
    )
    assert (
        effective.ground_owner
        .position_advancement_probability
        == pytest.approx(
            0.30 * 0.650
        )
    )

    assert effective.ground_defender.escape_attempt_rate == (
        pytest.approx(
            0.50 * 0.650
        )
    )
    assert effective.ground_defender.reversal_attempt_rate == (
        pytest.approx(
            0.20 * 0.650
        )
    )
    assert effective.ground_defender.scramble_attempt_rate == (
        pytest.approx(
            0.40 * 0.650
        )
    )
    assert effective.ground_defender.submission_defense == (
        pytest.approx(
            0.80 * 0.585
        )
    )


def test_effective_generation_does_not_mutate_baseline() -> None:
    baseline = baseline_phase_parameters()
    original = baseline_phase_parameters()

    effective = build_effective_phase_parameters(
        baseline,
        dynamic_state(),
        dynamic_parameters(),
        effect_calibration(),
    )

    assert baseline == original
    assert effective is not baseline
    assert effective != baseline


def test_more_severe_state_reduces_every_phase_capability() -> None:
    baseline = baseline_phase_parameters()

    fresh = build_effective_phase_parameters(
        baseline,
        FighterDynamicState.opening_state(),
        dynamic_parameters(),
        effect_calibration(),
    )
    impaired = build_effective_phase_parameters(
        baseline,
        FighterDynamicState(
            fatigue=1.0,
            damage=1.0,
            acute_stress=1.0,
        ),
        dynamic_parameters(),
        effect_calibration(),
    )

    assert (
        impaired.distance.sig_strike_attempt_rate
        < fresh.distance.sig_strike_attempt_rate
    )
    assert (
        impaired.distance.sig_strike_accuracy
        < fresh.distance.sig_strike_accuracy
    )
    assert (
        impaired.distance.knockdown_probability_per_landed
        < fresh.distance.knockdown_probability_per_landed
    )

    assert (
        impaired.clinch.control_seconds_mean
        < fresh.clinch.control_seconds_mean
    )
    assert (
        impaired.ground_owner.submission_attempt_rate
        < fresh.ground_owner.submission_attempt_rate
    )
    assert (
        impaired.ground_defender.escape_attempt_rate
        < fresh.ground_defender.escape_attempt_rate
    )
    assert (
        impaired.ground_defender.submission_defense
        < fresh.ground_defender.submission_defense
    )


def test_effective_generation_requires_phase_baseline() -> None:
    with pytest.raises(
        TypeError,
        match="baseline must be FighterPhaseParameters",
    ):
        build_effective_phase_parameters(
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
            "calibration must be DynamicEffectCalibration",
        ),
    ],
)
def test_all_multiplier_calculation_requires_correct_types(
    selected_state: object,
    selected_parameters: object,
    selected_calibration: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        calculate_capability_multipliers(
            selected_state,
            selected_parameters,
            selected_calibration,
        )
