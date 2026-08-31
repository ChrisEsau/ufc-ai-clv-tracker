"""Tests for RFS Monte Carlo V2 finish-probability calibration."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.finish_calibration import (
    FinishProbabilityCalibration,
    KnockoutFinishCalibration,
    SubmissionFinishCalibration,
)


KNOCKOUT_FIELDS = [
    "distance_landed_probability",
    "distance_knockdown_probability",
    "clinch_landed_probability",
    "damaging_clinch_probability",
    "ground_landed_probability",
    "defender_fatigue_amplifier",
    "defender_damage_amplifier",
    "defender_acute_stress_amplifier",
    "maximum_segment_probability",
]

SUBMISSION_FIELDS = [
    "base_probability_per_attempt",
    "position_quality_amplifier",
    "minimum_submission_defense_effect_multiplier",
    "defender_fatigue_amplifier",
    "defender_damage_amplifier",
    "defender_acute_stress_amplifier",
    "maximum_probability_per_attempt",
    "maximum_segment_probability",
]


def knockout_values() -> dict[str, object]:
    """Build valid KO/TKO calibration values."""

    return {
        "distance_landed_probability": 0.010,
        "distance_knockdown_probability": 0.300,
        "clinch_landed_probability": 0.008,
        "damaging_clinch_probability": 0.120,
        "ground_landed_probability": 0.012,
        "defender_fatigue_amplifier": 0.300,
        "defender_damage_amplifier": 0.600,
        "defender_acute_stress_amplifier": 0.200,
        "maximum_segment_probability": 0.750,
    }


def submission_values() -> dict[str, object]:
    """Build valid submission calibration values."""

    return {
        "base_probability_per_attempt": 0.080,
        "position_quality_amplifier": 0.500,
        "minimum_submission_defense_effect_multiplier": 0.150,
        "defender_fatigue_amplifier": 0.300,
        "defender_damage_amplifier": 0.250,
        "defender_acute_stress_amplifier": 0.200,
        "maximum_probability_per_attempt": 0.500,
        "maximum_segment_probability": 0.800,
    }


def knockout_calibration() -> KnockoutFinishCalibration:
    """Build one valid KO/TKO calibration."""

    return KnockoutFinishCalibration(
        **knockout_values()
    )


def submission_calibration() -> SubmissionFinishCalibration:
    """Build one valid submission calibration."""

    return SubmissionFinishCalibration(
        **submission_values()
    )


def finish_calibration() -> FinishProbabilityCalibration:
    """Build one complete finish-probability calibration."""

    return FinishProbabilityCalibration(
        knockout=knockout_calibration(),
        submission=submission_calibration(),
    )


def test_knockout_calibration_retains_values() -> None:
    selected = knockout_calibration()

    assert selected.distance_landed_probability == 0.010
    assert selected.distance_knockdown_probability == 0.300
    assert selected.clinch_landed_probability == 0.008
    assert selected.damaging_clinch_probability == 0.120
    assert selected.ground_landed_probability == 0.012
    assert selected.defender_fatigue_amplifier == 0.300
    assert selected.defender_damage_amplifier == 0.600
    assert selected.defender_acute_stress_amplifier == 0.200
    assert selected.maximum_segment_probability == 0.750


def test_submission_calibration_retains_values() -> None:
    selected = submission_calibration()

    assert selected.base_probability_per_attempt == 0.080
    assert selected.position_quality_amplifier == 0.500
    assert (
        selected.minimum_submission_defense_effect_multiplier
        == 0.150
    )
    assert selected.defender_fatigue_amplifier == 0.300
    assert selected.defender_damage_amplifier == 0.250
    assert selected.defender_acute_stress_amplifier == 0.200
    assert selected.maximum_probability_per_attempt == 0.500
    assert selected.maximum_segment_probability == 0.800


def test_finish_calibration_retains_nested_contracts() -> None:
    knockout = knockout_calibration()
    submission = submission_calibration()

    selected = FinishProbabilityCalibration(
        knockout=knockout,
        submission=submission,
    )

    assert selected.knockout is knockout
    assert selected.submission is submission


def test_knockout_scalar_boundaries_are_allowed() -> None:
    low_values = {
        field_name: 0.0
        for field_name in KNOCKOUT_FIELDS
    }
    high_values = {
        field_name: 1.0
        for field_name in KNOCKOUT_FIELDS
    }

    low = KnockoutFinishCalibration(
        **low_values
    )
    high = KnockoutFinishCalibration(
        **high_values
    )

    for field_name in KNOCKOUT_FIELDS:
        assert getattr(low, field_name) == 0.0
        assert getattr(high, field_name) == 1.0


def test_submission_scalar_boundaries_are_allowed() -> None:
    low_values = {
        field_name: 0.0
        for field_name in SUBMISSION_FIELDS
    }
    high_values = {
        field_name: 1.0
        for field_name in SUBMISSION_FIELDS
    }

    low = SubmissionFinishCalibration(
        **low_values
    )
    high = SubmissionFinishCalibration(
        **high_values
    )

    for field_name in SUBMISSION_FIELDS:
        assert getattr(low, field_name) == 0.0
        assert getattr(high, field_name) == 1.0


@pytest.mark.parametrize(
    "field_name",
    KNOCKOUT_FIELDS,
)
def test_knockout_values_must_be_numeric(
    field_name: str,
) -> None:
    values = knockout_values()
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        KnockoutFinishCalibration(
            **values
        )


@pytest.mark.parametrize(
    "field_name",
    KNOCKOUT_FIELDS,
)
def test_knockout_values_must_be_finite(
    field_name: str,
) -> None:
    values = knockout_values()
    values[field_name] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        KnockoutFinishCalibration(
            **values
        )


@pytest.mark.parametrize(
    "field_name",
    KNOCKOUT_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
    ],
)
def test_knockout_values_must_be_in_unit_interval(
    field_name: str,
    invalid_value: float,
) -> None:
    values = knockout_values()
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        KnockoutFinishCalibration(
            **values
        )


@pytest.mark.parametrize(
    "field_name",
    SUBMISSION_FIELDS,
)
def test_submission_values_must_be_numeric(
    field_name: str,
) -> None:
    values = submission_values()
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        SubmissionFinishCalibration(
            **values
        )


@pytest.mark.parametrize(
    "field_name",
    SUBMISSION_FIELDS,
)
def test_submission_values_must_be_finite(
    field_name: str,
) -> None:
    values = submission_values()
    values[field_name] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        SubmissionFinishCalibration(
            **values
        )


@pytest.mark.parametrize(
    "field_name",
    SUBMISSION_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
    ],
)
def test_submission_values_must_be_in_unit_interval(
    field_name: str,
    invalid_value: float,
) -> None:
    values = submission_values()
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        SubmissionFinishCalibration(
            **values
        )


def test_finish_calibration_requires_knockout_contract() -> None:
    with pytest.raises(
        TypeError,
        match="knockout must be KnockoutFinishCalibration",
    ):
        FinishProbabilityCalibration(
            knockout="invalid",
            submission=submission_calibration(),
        )


def test_finish_calibration_requires_submission_contract() -> None:
    with pytest.raises(
        TypeError,
        match="submission must be SubmissionFinishCalibration",
    ):
        FinishProbabilityCalibration(
            knockout=knockout_calibration(),
            submission="invalid",
        )


def test_knockout_calibration_is_immutable() -> None:
    selected = knockout_calibration()

    with pytest.raises(FrozenInstanceError):
        selected.maximum_segment_probability = 0.20


def test_submission_calibration_is_immutable() -> None:
    selected = submission_calibration()

    with pytest.raises(FrozenInstanceError):
        selected.maximum_segment_probability = 0.20


def test_finish_calibration_is_immutable() -> None:
    selected = finish_calibration()

    with pytest.raises(FrozenInstanceError):
        selected.knockout = knockout_calibration()
