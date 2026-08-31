"""Tests for V2 dynamic performance-effect calibration."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
    StatePenaltyWeights,
)


WEIGHT_FIELDS = [
    "fatigue",
    "damage",
    "acute_stress",
]

SCALAR_FIELDS = [
    "minimum_fatigue_effect_multiplier",
    "minimum_effective_capability_multiplier",
]

CAPABILITY_FIELDS = [
    "output",
    "accuracy",
    "power",
    "control",
    "grappling",
    "defense",
]


def penalty_weights(
    **overrides: float,
) -> StatePenaltyWeights:
    """Build valid dynamic-state penalty weights."""

    values = {
        "fatigue": 0.40,
        "damage": 0.25,
        "acute_stress": 0.20,
    }
    values.update(overrides)

    return StatePenaltyWeights(**values)


def calibration_values() -> dict[str, object]:
    """Build values for a complete effect calibration."""

    return {
        "minimum_fatigue_effect_multiplier": 0.20,
        "minimum_effective_capability_multiplier": 0.15,
        "output": penalty_weights(
            fatigue=0.55,
            damage=0.20,
            acute_stress=0.25,
        ),
        "accuracy": penalty_weights(
            fatigue=0.25,
            damage=0.30,
            acute_stress=0.35,
        ),
        "power": penalty_weights(
            fatigue=0.30,
            damage=0.35,
            acute_stress=0.20,
        ),
        "control": penalty_weights(
            fatigue=0.50,
            damage=0.25,
            acute_stress=0.20,
        ),
        "grappling": penalty_weights(
            fatigue=0.45,
            damage=0.30,
            acute_stress=0.25,
        ),
        "defense": penalty_weights(
            fatigue=0.35,
            damage=0.45,
            acute_stress=0.30,
        ),
    }


def calibration() -> DynamicEffectCalibration:
    """Build a complete valid effect calibration."""

    return DynamicEffectCalibration(
        **calibration_values()
    )


def test_valid_penalty_weights_are_retained() -> None:
    weights = penalty_weights()

    assert weights.fatigue == 0.40
    assert weights.damage == 0.25
    assert weights.acute_stress == 0.20


def test_penalty_weight_boundaries_are_allowed() -> None:
    low = StatePenaltyWeights(
        fatigue=0.0,
        damage=0.0,
        acute_stress=0.0,
    )
    high = StatePenaltyWeights(
        fatigue=1.0,
        damage=1.0,
        acute_stress=1.0,
    )

    assert low.fatigue == 0.0
    assert high.fatigue == 1.0


@pytest.mark.parametrize(
    "field_name",
    WEIGHT_FIELDS,
)
def test_penalty_weights_must_be_numeric(
    field_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        penalty_weights(
            **{
                field_name: "invalid",
            }
        )


@pytest.mark.parametrize(
    "field_name",
    WEIGHT_FIELDS,
)
def test_penalty_weights_must_be_finite(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        penalty_weights(
            **{
                field_name: float("nan"),
            }
        )


@pytest.mark.parametrize(
    "field_name",
    WEIGHT_FIELDS,
)
def test_penalty_weights_cannot_be_negative(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        penalty_weights(
            **{
                field_name: -0.01,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    WEIGHT_FIELDS,
)
def test_penalty_weights_cannot_exceed_one(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        penalty_weights(
            **{
                field_name: 1.01,
            }
        )


def test_penalty_weights_are_immutable() -> None:
    weights = penalty_weights()

    with pytest.raises(FrozenInstanceError):
        weights.fatigue = 0.90


def test_valid_effect_calibration_retains_values() -> None:
    selected = calibration()

    assert selected.minimum_fatigue_effect_multiplier == 0.20
    assert selected.minimum_effective_capability_multiplier == 0.15

    assert selected.output.fatigue == 0.55
    assert selected.accuracy.damage == 0.30
    assert selected.power.acute_stress == 0.20
    assert selected.control.fatigue == 0.50
    assert selected.grappling.damage == 0.30
    assert selected.defense.acute_stress == 0.30


def test_effect_scalar_boundaries_are_allowed() -> None:
    low_values = calibration_values()
    low_values[
        "minimum_fatigue_effect_multiplier"
    ] = 0.0
    low_values[
        "minimum_effective_capability_multiplier"
    ] = 0.0

    high_values = calibration_values()
    high_values[
        "minimum_fatigue_effect_multiplier"
    ] = 1.0
    high_values[
        "minimum_effective_capability_multiplier"
    ] = 1.0

    low = DynamicEffectCalibration(**low_values)
    high = DynamicEffectCalibration(**high_values)

    assert low.minimum_fatigue_effect_multiplier == 0.0
    assert high.minimum_fatigue_effect_multiplier == 1.0

    assert low.minimum_effective_capability_multiplier == 0.0
    assert high.minimum_effective_capability_multiplier == 1.0


@pytest.mark.parametrize(
    "field_name",
    SCALAR_FIELDS,
)
def test_effect_scalars_must_be_numeric(
    field_name: str,
) -> None:
    values = calibration_values()
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        DynamicEffectCalibration(**values)


@pytest.mark.parametrize(
    "field_name",
    SCALAR_FIELDS,
)
def test_effect_scalars_must_be_finite(
    field_name: str,
) -> None:
    values = calibration_values()
    values[field_name] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        DynamicEffectCalibration(**values)


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
            "minimum_effective_capability_multiplier",
            -0.01,
        ),
        (
            "minimum_effective_capability_multiplier",
            1.01,
        ),
    ],
)
def test_effect_scalars_must_be_in_unit_interval(
    field_name: str,
    invalid_value: float,
) -> None:
    values = calibration_values()
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        DynamicEffectCalibration(**values)


@pytest.mark.parametrize(
    "field_name",
    CAPABILITY_FIELDS,
)
def test_capability_fields_require_penalty_weights(
    field_name: str,
) -> None:
    values = calibration_values()
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be StatePenaltyWeights",
    ):
        DynamicEffectCalibration(**values)


def test_effect_calibration_is_immutable() -> None:
    selected = calibration()

    with pytest.raises(FrozenInstanceError):
        selected.minimum_fatigue_effect_multiplier = 0.90
