"""Tests for V2 dynamic-state calibration contracts."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    ActivityWorkloadCalibration,
    AdversityCalibration,
    DynamicStateCalibration,
    PhaseWorkloadCalibration,
    RecoveryCalibration,
    ResistanceScalingCalibration,
)


PHASE_VALUES = {
    "distance": 0.010,
    "clinch_owner": 0.020,
    "clinch_defender": 0.025,
    "ground_owner": 0.025,
    "ground_defender": 0.030,
}

ACTIVITY_VALUES = {
    "strike_attempt": 0.002,
    "control_second": 0.001,
    "submission_attempt": 0.010,
    "position_advancement": 0.008,
    "escape_attempt": 0.006,
    "reversal_attempt": 0.009,
    "scramble_attempt": 0.008,
}

ADVERSITY_VALUES = {
    "distance_landed_damage": 0.003,
    "clinch_landed_damage": 0.002,
    "damaging_clinch_bonus_damage": 0.010,
    "ground_landed_damage": 0.004,
    "knockdown_damage": 0.150,
    "distance_landed_stress": 0.004,
    "clinch_landed_stress": 0.003,
    "damaging_clinch_bonus_stress": 0.020,
    "ground_landed_stress": 0.005,
    "knockdown_stress": 0.250,
    "control_second_received_stress": 0.001,
    "submission_attempt_received_stress": 0.020,
    "position_advancement_received_stress": 0.015,
}

RESISTANCE_VALUES = {
    "minimum_fatigue_accumulation_multiplier": 0.25,
    "minimum_damage_accumulation_multiplier": 0.20,
    "minimum_acute_stress_accumulation_multiplier": 0.15,
}

RECOVERY_VALUES = {
    "low_workload_threshold": 0.10,
    "segment_fatigue_recovery": 0.01,
    "round_break_fatigue_recovery": 0.08,
    "segment_acute_stress_recovery": 0.10,
    "round_break_acute_stress_recovery": 0.35,
}


NONNEGATIVE_CASES = [
    *[
        (
            PhaseWorkloadCalibration,
            PHASE_VALUES,
            field_name,
        )
        for field_name in PHASE_VALUES
    ],
    *[
        (
            ActivityWorkloadCalibration,
            ACTIVITY_VALUES,
            field_name,
        )
        for field_name in ACTIVITY_VALUES
    ],
    *[
        (
            AdversityCalibration,
            ADVERSITY_VALUES,
            field_name,
        )
        for field_name in ADVERSITY_VALUES
    ],
]

UNIT_INTERVAL_CASES = [
    *[
        (
            ResistanceScalingCalibration,
            RESISTANCE_VALUES,
            field_name,
        )
        for field_name in RESISTANCE_VALUES
    ],
    *[
        (
            RecoveryCalibration,
            RECOVERY_VALUES,
            field_name,
        )
        for field_name in RECOVERY_VALUES
    ],
]


def phase_workload() -> PhaseWorkloadCalibration:
    """Build valid phase-role workload calibration."""

    return PhaseWorkloadCalibration(**PHASE_VALUES)


def activity_workload() -> ActivityWorkloadCalibration:
    """Build valid realized-activity workload calibration."""

    return ActivityWorkloadCalibration(**ACTIVITY_VALUES)


def adversity() -> AdversityCalibration:
    """Build valid adversity calibration."""

    return AdversityCalibration(**ADVERSITY_VALUES)


def resistance_scaling() -> ResistanceScalingCalibration:
    """Build valid resistance-scaling calibration."""

    return ResistanceScalingCalibration(**RESISTANCE_VALUES)


def recovery() -> RecoveryCalibration:
    """Build valid recovery calibration."""

    return RecoveryCalibration(**RECOVERY_VALUES)


def calibration_values() -> dict[str, object]:
    """Build values for the complete nested calibration bundle."""

    return {
        "phase_workload": phase_workload(),
        "activity_workload": activity_workload(),
        "adversity": adversity(),
        "resistance_scaling": resistance_scaling(),
        "recovery": recovery(),
    }


def calibration() -> DynamicStateCalibration:
    """Build a complete valid dynamic-state calibration."""

    return DynamicStateCalibration(**calibration_values())


def build_with_override(
    contract_type: type,
    baseline: dict[str, float],
    field_name: str,
    value: object,
) -> object:
    """Construct one scalar contract with a field override."""

    values = dict(baseline)
    values[field_name] = value

    return contract_type(**values)


def test_valid_calibration_retains_nested_contracts() -> None:
    selected = calibration()

    assert selected.phase_workload == phase_workload()
    assert selected.activity_workload == activity_workload()
    assert selected.adversity == adversity()
    assert selected.resistance_scaling == resistance_scaling()
    assert selected.recovery == recovery()


def test_zero_values_are_allowed() -> None:
    phase = PhaseWorkloadCalibration(
        **{
            name: 0.0
            for name in PHASE_VALUES
        }
    )
    activity = ActivityWorkloadCalibration(
        **{
            name: 0.0
            for name in ACTIVITY_VALUES
        }
    )
    selected_adversity = AdversityCalibration(
        **{
            name: 0.0
            for name in ADVERSITY_VALUES
        }
    )
    resistance = ResistanceScalingCalibration(
        **{
            name: 0.0
            for name in RESISTANCE_VALUES
        }
    )
    selected_recovery = RecoveryCalibration(
        **{
            name: 0.0
            for name in RECOVERY_VALUES
        }
    )

    assert phase.distance == 0.0
    assert activity.strike_attempt == 0.0
    assert selected_adversity.knockdown_damage == 0.0
    assert (
        resistance.minimum_fatigue_accumulation_multiplier
        == 0.0
    )
    assert selected_recovery.round_break_fatigue_recovery == 0.0


@pytest.mark.parametrize(
    (
        "contract_type",
        "baseline",
        "field_name",
    ),
    NONNEGATIVE_CASES,
)
def test_nonnegative_costs_must_be_numeric(
    contract_type: type,
    baseline: dict[str, float],
    field_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        build_with_override(
            contract_type,
            baseline,
            field_name,
            "invalid",
        )


@pytest.mark.parametrize(
    (
        "contract_type",
        "baseline",
        "field_name",
    ),
    NONNEGATIVE_CASES,
)
def test_nonnegative_costs_must_be_finite(
    contract_type: type,
    baseline: dict[str, float],
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        build_with_override(
            contract_type,
            baseline,
            field_name,
            float("nan"),
        )


@pytest.mark.parametrize(
    (
        "contract_type",
        "baseline",
        "field_name",
    ),
    NONNEGATIVE_CASES,
)
def test_nonnegative_costs_cannot_be_negative(
    contract_type: type,
    baseline: dict[str, float],
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be negative",
    ):
        build_with_override(
            contract_type,
            baseline,
            field_name,
            -0.01,
        )


@pytest.mark.parametrize(
    (
        "contract_type",
        "baseline",
        "field_name",
    ),
    UNIT_INTERVAL_CASES,
)
def test_normalized_values_cannot_exceed_one(
    contract_type: type,
    baseline: dict[str, float],
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        build_with_override(
            contract_type,
            baseline,
            field_name,
            1.01,
        )


def test_resistance_scaling_accepts_unit_boundaries() -> None:
    low = ResistanceScalingCalibration(
        **{
            name: 0.0
            for name in RESISTANCE_VALUES
        }
    )
    high = ResistanceScalingCalibration(
        **{
            name: 1.0
            for name in RESISTANCE_VALUES
        }
    )

    assert (
        low.minimum_damage_accumulation_multiplier
        == 0.0
    )
    assert (
        high.minimum_damage_accumulation_multiplier
        == 1.0
    )


def test_recovery_accepts_unit_boundaries() -> None:
    low = RecoveryCalibration(
        **{
            name: 0.0
            for name in RECOVERY_VALUES
        }
    )
    high = RecoveryCalibration(
        **{
            name: 1.0
            for name in RECOVERY_VALUES
        }
    )

    assert low.low_workload_threshold == 0.0
    assert high.low_workload_threshold == 1.0


@pytest.mark.parametrize(
    (
        "contract_type",
        "baseline",
        "field_name",
    ),
    [
        (
            PhaseWorkloadCalibration,
            PHASE_VALUES,
            "distance",
        ),
        (
            ActivityWorkloadCalibration,
            ACTIVITY_VALUES,
            "strike_attempt",
        ),
        (
            AdversityCalibration,
            ADVERSITY_VALUES,
            "knockdown_damage",
        ),
    ],
)
def test_unbounded_cost_fields_may_exceed_one(
    contract_type: type,
    baseline: dict[str, float],
    field_name: str,
) -> None:
    selected = build_with_override(
        contract_type,
        baseline,
        field_name,
        2.50,
    )

    assert getattr(selected, field_name) == 2.50


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_type_name",
    ),
    [
        (
            "phase_workload",
            activity_workload(),
            "PhaseWorkloadCalibration",
        ),
        (
            "activity_workload",
            phase_workload(),
            "ActivityWorkloadCalibration",
        ),
        (
            "adversity",
            recovery(),
            "AdversityCalibration",
        ),
        (
            "resistance_scaling",
            recovery(),
            "ResistanceScalingCalibration",
        ),
        (
            "recovery",
            resistance_scaling(),
            "RecoveryCalibration",
        ),
    ],
)
def test_complete_calibration_requires_correct_nested_types(
    field_name: str,
    invalid_value: object,
    expected_type_name: str,
) -> None:
    values = calibration_values()
    values[field_name] = invalid_value

    with pytest.raises(
        TypeError,
        match=(
            f"{field_name} must be "
            f"{expected_type_name}"
        ),
    ):
        DynamicStateCalibration(**values)


@pytest.mark.parametrize(
    (
        "instance",
        "field_name",
        "replacement",
    ),
    [
        (
            phase_workload(),
            "distance",
            0.90,
        ),
        (
            activity_workload(),
            "strike_attempt",
            0.90,
        ),
        (
            adversity(),
            "knockdown_damage",
            0.90,
        ),
        (
            resistance_scaling(),
            "minimum_damage_accumulation_multiplier",
            0.90,
        ),
        (
            recovery(),
            "round_break_fatigue_recovery",
            0.90,
        ),
        (
            calibration(),
            "phase_workload",
            phase_workload(),
        ),
    ],
)
def test_calibration_contracts_are_immutable(
    instance: object,
    field_name: str,
    replacement: object,
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(
            instance,
            field_name,
            replacement,
        )
