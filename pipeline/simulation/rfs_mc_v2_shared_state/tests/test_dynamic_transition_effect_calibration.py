"""Tests for V2 dynamic transition-effect calibration."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    StatePenaltyWeights,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_transition_effect_calibration import (
    DynamicTransitionEffectCalibration,
)


SCALAR_FIELDS = [
    "minimum_fatigue_effect_multiplier",
    "minimum_effective_transition_multiplier",
]

CAPABILITY_FIELDS = [
    "entry",
    "completion",
    "retention",
    "escape",
    "reversal",
    "persistence",
    "imposition",
    "resistance",
]


def weights(
    *,
    fatigue: float,
    damage: float,
    acute_stress: float,
) -> StatePenaltyWeights:
    """Build one valid transition penalty-weight bundle."""

    return StatePenaltyWeights(
        fatigue=fatigue,
        damage=damage,
        acute_stress=acute_stress,
    )


def calibration_values() -> dict[str, object]:
    """Build values for a complete transition-effect calibration."""

    return {
        "minimum_fatigue_effect_multiplier": 0.20,
        "minimum_effective_transition_multiplier": 0.15,
        "entry": weights(
            fatigue=0.45,
            damage=0.20,
            acute_stress=0.30,
        ),
        "completion": weights(
            fatigue=0.55,
            damage=0.30,
            acute_stress=0.25,
        ),
        "retention": weights(
            fatigue=0.50,
            damage=0.25,
            acute_stress=0.20,
        ),
        "escape": weights(
            fatigue=0.45,
            damage=0.40,
            acute_stress=0.30,
        ),
        "reversal": weights(
            fatigue=0.55,
            damage=0.35,
            acute_stress=0.25,
        ),
        "persistence": weights(
            fatigue=0.60,
            damage=0.20,
            acute_stress=0.25,
        ),
        "imposition": weights(
            fatigue=0.50,
            damage=0.25,
            acute_stress=0.30,
        ),
        "resistance": weights(
            fatigue=0.40,
            damage=0.45,
            acute_stress=0.35,
        ),
    }


def calibration() -> DynamicTransitionEffectCalibration:
    """Build a complete valid transition-effect calibration."""

    return DynamicTransitionEffectCalibration(
        **calibration_values()
    )


def test_valid_transition_effect_calibration_retains_values() -> None:
    selected = calibration()

    assert selected.minimum_fatigue_effect_multiplier == 0.20
    assert selected.minimum_effective_transition_multiplier == 0.15

    assert selected.entry.fatigue == 0.45
    assert selected.completion.damage == 0.30
    assert selected.retention.acute_stress == 0.20
    assert selected.escape.damage == 0.40
    assert selected.reversal.fatigue == 0.55
    assert selected.persistence.fatigue == 0.60
    assert selected.imposition.acute_stress == 0.30
    assert selected.resistance.damage == 0.45


def test_transition_effect_scalar_boundaries_are_allowed() -> None:
    low_values = calibration_values()
    low_values[
        "minimum_fatigue_effect_multiplier"
    ] = 0.0
    low_values[
        "minimum_effective_transition_multiplier"
    ] = 0.0

    high_values = calibration_values()
    high_values[
        "minimum_fatigue_effect_multiplier"
    ] = 1.0
    high_values[
        "minimum_effective_transition_multiplier"
    ] = 1.0

    low = DynamicTransitionEffectCalibration(
        **low_values
    )
    high = DynamicTransitionEffectCalibration(
        **high_values
    )

    assert low.minimum_fatigue_effect_multiplier == 0.0
    assert high.minimum_fatigue_effect_multiplier == 1.0

    assert low.minimum_effective_transition_multiplier == 0.0
    assert high.minimum_effective_transition_multiplier == 1.0


@pytest.mark.parametrize(
    "field_name",
    SCALAR_FIELDS,
)
def test_transition_effect_scalars_must_be_numeric(
    field_name: str,
) -> None:
    values = calibration_values()
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        DynamicTransitionEffectCalibration(
            **values
        )


@pytest.mark.parametrize(
    "field_name",
    SCALAR_FIELDS,
)
def test_transition_effect_scalars_must_be_finite(
    field_name: str,
) -> None:
    values = calibration_values()
    values[field_name] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        DynamicTransitionEffectCalibration(
            **values
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        (
            "minimum_fatigue_effect_multiplier",
            -0.01,
        ),
        (
            "minimum_fatigue_effect_multiplier",
            1.01,
        ),
        (
            "minimum_effective_transition_multiplier",
            -0.01,
        ),
        (
            "minimum_effective_transition_multiplier",
            1.01,
        ),
    ],
)
def test_transition_effect_scalars_must_be_in_unit_interval(
    field_name: str,
    invalid_value: float,
) -> None:
    values = calibration_values()
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        DynamicTransitionEffectCalibration(
            **values
        )


@pytest.mark.parametrize(
    "field_name",
    CAPABILITY_FIELDS,
)
def test_transition_capability_fields_require_penalty_weights(
    field_name: str,
) -> None:
    values = calibration_values()
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be StatePenaltyWeights",
    ):
        DynamicTransitionEffectCalibration(
            **values
        )


def test_transition_effect_calibration_is_immutable() -> None:
    selected = calibration()

    with pytest.raises(FrozenInstanceError):
        selected.minimum_effective_transition_multiplier = 0.90
