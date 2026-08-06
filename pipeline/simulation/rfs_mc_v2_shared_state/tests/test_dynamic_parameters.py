"""Tests for V2 immutable fighter dynamic-response parameters."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)


PARAMETER_NAMES = [
    "fatigue_accumulation_resistance",
    "fatigue_performance_resilience",
    "recovery_ability",
    "damage_resistance",
    "acute_stress_resistance",
    "acute_stress_recovery",
]


def dynamic_parameters(
    **overrides: float,
) -> FighterDynamicParameters:
    """Build a valid dynamic-response parameter bundle."""

    values = {
        "fatigue_accumulation_resistance": 0.60,
        "fatigue_performance_resilience": 0.70,
        "recovery_ability": 0.55,
        "damage_resistance": 0.65,
        "acute_stress_resistance": 0.50,
        "acute_stress_recovery": 0.75,
    }
    values.update(overrides)

    return FighterDynamicParameters(**values)


def test_valid_dynamic_parameters_are_retained() -> None:
    parameters = dynamic_parameters()

    assert parameters.fatigue_accumulation_resistance == 0.60
    assert parameters.fatigue_performance_resilience == 0.70
    assert parameters.recovery_ability == 0.55
    assert parameters.damage_resistance == 0.65
    assert parameters.acute_stress_resistance == 0.50
    assert parameters.acute_stress_recovery == 0.75


def test_boundary_values_are_allowed() -> None:
    low = FighterDynamicParameters(
        fatigue_accumulation_resistance=0.0,
        fatigue_performance_resilience=0.0,
        recovery_ability=0.0,
        damage_resistance=0.0,
        acute_stress_resistance=0.0,
        acute_stress_recovery=0.0,
    )

    high = FighterDynamicParameters(
        fatigue_accumulation_resistance=1.0,
        fatigue_performance_resilience=1.0,
        recovery_ability=1.0,
        damage_resistance=1.0,
        acute_stress_resistance=1.0,
        acute_stress_recovery=1.0,
    )

    assert low.fatigue_accumulation_resistance == 0.0
    assert high.fatigue_accumulation_resistance == 1.0


@pytest.mark.parametrize(
    "field_name",
    PARAMETER_NAMES,
)
def test_dynamic_parameters_must_be_numeric(
    field_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        dynamic_parameters(
            **{
                field_name: "invalid",
            }
        )


@pytest.mark.parametrize(
    "field_name",
    PARAMETER_NAMES,
)
def test_dynamic_parameters_must_be_finite(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        dynamic_parameters(
            **{
                field_name: float("nan"),
            }
        )


@pytest.mark.parametrize(
    "field_name",
    PARAMETER_NAMES,
)
def test_dynamic_parameters_cannot_be_negative(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        dynamic_parameters(
            **{
                field_name: -0.01,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    PARAMETER_NAMES,
)
def test_dynamic_parameters_cannot_exceed_one(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        dynamic_parameters(
            **{
                field_name: 1.01,
            }
        )


def test_dynamic_parameters_are_immutable() -> None:
    parameters = dynamic_parameters()

    with pytest.raises(FrozenInstanceError):
        parameters.recovery_ability = 0.90
