"""Tests for V2 segment workload and adversity calculation."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchFighterActivity,
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceFighterActivity,
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    ActivityWorkloadCalibration,
    AdversityCalibration,
    DynamicStateCalibration,
    PhaseWorkloadCalibration,
    RecoveryCalibration,
    ResistanceScalingCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_exposure import (
    FighterSegmentExposure,
    SegmentDynamicExposure,
    calculate_segment_dynamic_exposure,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundFighterActivity,
    GroundSegmentActivity,
)


EXPOSURE_FIELDS = [
    "fatigue_workload",
    "persistent_damage_exposure",
    "acute_stress_exposure",
]


def shared_state(
    phase: FightPhase,
    *,
    owner: FighterSide | None,
) -> SharedFightState:
    """Build a valid shared state for one segment."""

    return SharedFightState(
        phase=phase,
        phase_owner=owner,
        phase_age_segments=1,
        position_quality=(
            0.0
            if phase is FightPhase.DISTANCE
            else 0.50
        ),
        round_number=1,
        segment_number=2,
    )


def calibration() -> DynamicStateCalibration:
    """Build a calibration with easy-to-audit values."""

    return DynamicStateCalibration(
        phase_workload=PhaseWorkloadCalibration(
            distance=0.10,
            clinch_owner=0.20,
            clinch_defender=0.30,
            ground_owner=0.25,
            ground_defender=0.35,
        ),
        activity_workload=ActivityWorkloadCalibration(
            strike_attempt=0.01,
            control_second=0.005,
            submission_attempt=0.04,
            position_advancement=0.03,
            escape_attempt=0.02,
            reversal_attempt=0.05,
            scramble_attempt=0.06,
        ),
        adversity=AdversityCalibration(
            distance_landed_damage=0.02,
            clinch_landed_damage=0.02,
            damaging_clinch_bonus_damage=0.10,
            ground_landed_damage=0.04,
            knockdown_damage=0.20,
            distance_landed_stress=0.03,
            clinch_landed_stress=0.03,
            damaging_clinch_bonus_stress=0.20,
            ground_landed_stress=0.05,
            knockdown_stress=0.30,
            control_second_received_stress=0.004,
            submission_attempt_received_stress=0.08,
            position_advancement_received_stress=0.07,
        ),
        resistance_scaling=ResistanceScalingCalibration(
            minimum_fatigue_accumulation_multiplier=0.25,
            minimum_damage_accumulation_multiplier=0.20,
            minimum_acute_stress_accumulation_multiplier=0.15,
        ),
        recovery=RecoveryCalibration(
            low_workload_threshold=0.10,
            segment_fatigue_recovery=0.01,
            round_break_fatigue_recovery=0.08,
            segment_acute_stress_recovery=0.10,
            round_break_acute_stress_recovery=0.35,
        ),
    )


def zero_calibration() -> DynamicStateCalibration:
    """Build a calibration where every modeled cost is zero."""

    return DynamicStateCalibration(
        phase_workload=PhaseWorkloadCalibration(
            distance=0.0,
            clinch_owner=0.0,
            clinch_defender=0.0,
            ground_owner=0.0,
            ground_defender=0.0,
        ),
        activity_workload=ActivityWorkloadCalibration(
            strike_attempt=0.0,
            control_second=0.0,
            submission_attempt=0.0,
            position_advancement=0.0,
            escape_attempt=0.0,
            reversal_attempt=0.0,
            scramble_attempt=0.0,
        ),
        adversity=AdversityCalibration(
            distance_landed_damage=0.0,
            clinch_landed_damage=0.0,
            damaging_clinch_bonus_damage=0.0,
            ground_landed_damage=0.0,
            knockdown_damage=0.0,
            distance_landed_stress=0.0,
            clinch_landed_stress=0.0,
            damaging_clinch_bonus_stress=0.0,
            ground_landed_stress=0.0,
            knockdown_stress=0.0,
            control_second_received_stress=0.0,
            submission_attempt_received_stress=0.0,
            position_advancement_received_stress=0.0,
        ),
        resistance_scaling=ResistanceScalingCalibration(
            minimum_fatigue_accumulation_multiplier=0.0,
            minimum_damage_accumulation_multiplier=0.0,
            minimum_acute_stress_accumulation_multiplier=0.0,
        ),
        recovery=RecoveryCalibration(
            low_workload_threshold=0.0,
            segment_fatigue_recovery=0.0,
            round_break_fatigue_recovery=0.0,
            segment_acute_stress_recovery=0.0,
            round_break_acute_stress_recovery=0.0,
        ),
    )


def zero_ground_activity() -> GroundFighterActivity:
    """Return an empty ground-activity record."""

    return GroundFighterActivity(
        ground_str_attempted=0,
        ground_str_landed=0,
        control_seconds=0,
        submission_attempts=0,
        position_advancements=0,
        escape_attempts=0,
        reversal_attempts=0,
        scramble_attempts=0,
    )


def test_valid_fighter_exposure_is_retained() -> None:
    exposure = FighterSegmentExposure(
        fatigue_workload=0.20,
        persistent_damage_exposure=0.30,
        acute_stress_exposure=0.40,
    )

    assert exposure.fatigue_workload == 0.20
    assert exposure.persistent_damage_exposure == 0.30
    assert exposure.acute_stress_exposure == 0.40


def test_zero_exposure_values_are_allowed() -> None:
    exposure = FighterSegmentExposure(
        fatigue_workload=0.0,
        persistent_damage_exposure=0.0,
        acute_stress_exposure=0.0,
    )

    assert exposure == FighterSegmentExposure(
        fatigue_workload=0.0,
        persistent_damage_exposure=0.0,
        acute_stress_exposure=0.0,
    )


@pytest.mark.parametrize(
    "field_name",
    EXPOSURE_FIELDS,
)
def test_exposure_values_must_be_numeric(
    field_name: str,
) -> None:
    values = {
        "fatigue_workload": 0.20,
        "persistent_damage_exposure": 0.30,
        "acute_stress_exposure": 0.40,
    }
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        FighterSegmentExposure(**values)


@pytest.mark.parametrize(
    "field_name",
    EXPOSURE_FIELDS,
)
def test_exposure_values_must_be_finite(
    field_name: str,
) -> None:
    values = {
        "fatigue_workload": 0.20,
        "persistent_damage_exposure": 0.30,
        "acute_stress_exposure": 0.40,
    }
    values[field_name] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        FighterSegmentExposure(**values)


@pytest.mark.parametrize(
    "field_name",
    EXPOSURE_FIELDS,
)
def test_exposure_values_cannot_be_negative(
    field_name: str,
) -> None:
    values = {
        "fatigue_workload": 0.20,
        "persistent_damage_exposure": 0.30,
        "acute_stress_exposure": 0.40,
    }
    values[field_name] = -0.01

    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be negative",
    ):
        FighterSegmentExposure(**values)


def test_valid_segment_exposure_is_retained() -> None:
    state = shared_state(
        FightPhase.DISTANCE,
        owner=None,
    )
    red = FighterSegmentExposure(
        fatigue_workload=0.10,
        persistent_damage_exposure=0.20,
        acute_stress_exposure=0.30,
    )
    blue = FighterSegmentExposure(
        fatigue_workload=0.40,
        persistent_damage_exposure=0.50,
        acute_stress_exposure=0.60,
    )

    exposure = SegmentDynamicExposure(
        state=state,
        red=red,
        blue=blue,
    )

    assert exposure.state == state
    assert exposure.red == red
    assert exposure.blue == blue


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "state",
            "invalid",
            "state must be SharedFightState",
        ),
        (
            "red",
            "invalid",
            "red must be FighterSegmentExposure",
        ),
        (
            "blue",
            "invalid",
            "blue must be FighterSegmentExposure",
        ),
    ],
)
def test_segment_exposure_requires_correct_nested_types(
    field_name: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    values = {
        "state": shared_state(
            FightPhase.DISTANCE,
            owner=None,
        ),
        "red": FighterSegmentExposure(
            fatigue_workload=0.10,
            persistent_damage_exposure=0.20,
            acute_stress_exposure=0.30,
        ),
        "blue": FighterSegmentExposure(
            fatigue_workload=0.40,
            persistent_damage_exposure=0.50,
            acute_stress_exposure=0.60,
        ),
    }
    values[field_name] = invalid_value

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        SegmentDynamicExposure(**values)


def test_fighter_exposure_is_immutable() -> None:
    exposure = FighterSegmentExposure(
        fatigue_workload=0.10,
        persistent_damage_exposure=0.20,
        acute_stress_exposure=0.30,
    )

    with pytest.raises(FrozenInstanceError):
        exposure.fatigue_workload = 0.90


def test_segment_exposure_is_immutable() -> None:
    exposure = SegmentDynamicExposure(
        state=shared_state(
            FightPhase.DISTANCE,
            owner=None,
        ),
        red=FighterSegmentExposure(
            fatigue_workload=0.10,
            persistent_damage_exposure=0.20,
            acute_stress_exposure=0.30,
        ),
        blue=FighterSegmentExposure(
            fatigue_workload=0.40,
            persistent_damage_exposure=0.50,
            acute_stress_exposure=0.60,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        exposure.red = exposure.blue


def test_distance_exposure_uses_own_work_and_opponent_adversity() -> None:
    state = shared_state(
        FightPhase.DISTANCE,
        owner=None,
    )

    activity = DistanceSegmentActivity(
        state=state,
        red=DistanceFighterActivity(
            sig_str_attempted=4,
            sig_str_landed=2,
            knockdowns=1,
        ),
        blue=DistanceFighterActivity(
            sig_str_attempted=3,
            sig_str_landed=1,
            knockdowns=0,
        ),
    )

    exposure = calculate_segment_dynamic_exposure(
        activity,
        calibration(),
    )

    assert exposure.state == state

    assert exposure.red.fatigue_workload == pytest.approx(
        0.10 + 4 * 0.01
    )
    assert exposure.blue.fatigue_workload == pytest.approx(
        0.10 + 3 * 0.01
    )

    assert exposure.red.persistent_damage_exposure == (
        pytest.approx(
            1 * 0.02
        )
    )
    assert exposure.blue.persistent_damage_exposure == (
        pytest.approx(
            2 * 0.02
            + 1 * 0.20
        )
    )

    assert exposure.red.acute_stress_exposure == (
        pytest.approx(
            1 * 0.03
        )
    )
    assert exposure.blue.acute_stress_exposure == (
        pytest.approx(
            2 * 0.03
            + 1 * 0.30
        )
    )


def test_red_clinch_owner_receives_owner_workload() -> None:
    state = shared_state(
        FightPhase.CLINCH,
        owner=FighterSide.RED,
    )

    activity = ClinchSegmentActivity(
        state=state,
        red=ClinchFighterActivity(
            clinch_str_attempted=4,
            clinch_str_landed=2,
            damaging_clinch_strikes=1,
            control_seconds=12,
        ),
        blue=ClinchFighterActivity(
            clinch_str_attempted=3,
            clinch_str_landed=1,
            damaging_clinch_strikes=0,
            control_seconds=0,
        ),
    )

    exposure = calculate_segment_dynamic_exposure(
        activity,
        calibration(),
    )

    assert exposure.red.fatigue_workload == pytest.approx(
        0.20
        + 4 * 0.01
        + 12 * 0.005
    )
    assert exposure.blue.fatigue_workload == pytest.approx(
        0.30
        + 3 * 0.01
    )

    assert exposure.red.persistent_damage_exposure == (
        pytest.approx(
            1 * 0.02
        )
    )
    assert exposure.blue.persistent_damage_exposure == (
        pytest.approx(
            2 * 0.02
            + 1 * 0.10
        )
    )

    assert exposure.red.acute_stress_exposure == (
        pytest.approx(
            1 * 0.03
        )
    )
    assert exposure.blue.acute_stress_exposure == (
        pytest.approx(
            2 * 0.03
            + 1 * 0.20
            + 12 * 0.004
        )
    )


def test_blue_clinch_owner_applies_control_stress_to_red() -> None:
    state = shared_state(
        FightPhase.CLINCH,
        owner=FighterSide.BLUE,
    )

    activity = ClinchSegmentActivity(
        state=state,
        red=ClinchFighterActivity(
            clinch_str_attempted=0,
            clinch_str_landed=0,
            damaging_clinch_strikes=0,
            control_seconds=0,
        ),
        blue=ClinchFighterActivity(
            clinch_str_attempted=0,
            clinch_str_landed=0,
            damaging_clinch_strikes=0,
            control_seconds=10,
        ),
    )

    exposure = calculate_segment_dynamic_exposure(
        activity,
        calibration(),
    )

    assert exposure.red.fatigue_workload == pytest.approx(
        0.30
    )
    assert exposure.blue.fatigue_workload == pytest.approx(
        0.20
        + 10 * 0.005
    )

    assert exposure.red.acute_stress_exposure == pytest.approx(
        10 * 0.004
    )
    assert exposure.blue.acute_stress_exposure == 0.0


def test_red_ground_owner_and_blue_defender_use_role_costs() -> None:
    state = shared_state(
        FightPhase.GROUND,
        owner=FighterSide.RED,
    )

    activity = GroundSegmentActivity(
        state=state,
        red=GroundFighterActivity(
            ground_str_attempted=5,
            ground_str_landed=3,
            control_seconds=15,
            submission_attempts=1,
            position_advancements=1,
            escape_attempts=0,
            reversal_attempts=0,
            scramble_attempts=0,
        ),
        blue=GroundFighterActivity(
            ground_str_attempted=0,
            ground_str_landed=0,
            control_seconds=0,
            submission_attempts=0,
            position_advancements=0,
            escape_attempts=2,
            reversal_attempts=1,
            scramble_attempts=1,
        ),
    )

    exposure = calculate_segment_dynamic_exposure(
        activity,
        calibration(),
    )

    assert exposure.red.fatigue_workload == pytest.approx(
        0.25
        + 5 * 0.01
        + 15 * 0.005
        + 1 * 0.04
        + 1 * 0.03
    )
    assert exposure.blue.fatigue_workload == pytest.approx(
        0.35
        + 2 * 0.02
        + 1 * 0.05
        + 1 * 0.06
    )

    assert exposure.red.persistent_damage_exposure == 0.0
    assert exposure.blue.persistent_damage_exposure == (
        pytest.approx(
            3 * 0.04
        )
    )

    assert exposure.red.acute_stress_exposure == 0.0
    assert exposure.blue.acute_stress_exposure == (
        pytest.approx(
            3 * 0.05
            + 15 * 0.004
            + 1 * 0.08
            + 1 * 0.07
        )
    )


def test_blue_ground_owner_applies_ground_adversity_to_red() -> None:
    state = shared_state(
        FightPhase.GROUND,
        owner=FighterSide.BLUE,
    )

    activity = GroundSegmentActivity(
        state=state,
        red=GroundFighterActivity(
            ground_str_attempted=0,
            ground_str_landed=0,
            control_seconds=0,
            submission_attempts=0,
            position_advancements=0,
            escape_attempts=1,
            reversal_attempts=0,
            scramble_attempts=0,
        ),
        blue=GroundFighterActivity(
            ground_str_attempted=2,
            ground_str_landed=2,
            control_seconds=10,
            submission_attempts=1,
            position_advancements=1,
            escape_attempts=0,
            reversal_attempts=0,
            scramble_attempts=0,
        ),
    )

    exposure = calculate_segment_dynamic_exposure(
        activity,
        calibration(),
    )

    assert exposure.red.fatigue_workload == pytest.approx(
        0.35
        + 1 * 0.02
    )
    assert exposure.blue.fatigue_workload == pytest.approx(
        0.25
        + 2 * 0.01
        + 10 * 0.005
        + 1 * 0.04
        + 1 * 0.03
    )

    assert exposure.red.persistent_damage_exposure == (
        pytest.approx(
            2 * 0.04
        )
    )
    assert exposure.blue.persistent_damage_exposure == 0.0

    assert exposure.red.acute_stress_exposure == (
        pytest.approx(
            2 * 0.05
            + 10 * 0.004
            + 1 * 0.08
            + 1 * 0.07
        )
    )
    assert exposure.blue.acute_stress_exposure == 0.0


def test_zero_cost_calibration_produces_zero_exposure() -> None:
    activity = DistanceSegmentActivity(
        state=shared_state(
            FightPhase.DISTANCE,
            owner=None,
        ),
        red=DistanceFighterActivity(
            sig_str_attempted=10,
            sig_str_landed=8,
            knockdowns=2,
        ),
        blue=DistanceFighterActivity(
            sig_str_attempted=12,
            sig_str_landed=9,
            knockdowns=1,
        ),
    )

    exposure = calculate_segment_dynamic_exposure(
        activity,
        zero_calibration(),
    )

    zero = FighterSegmentExposure(
        fatigue_workload=0.0,
        persistent_damage_exposure=0.0,
        acute_stress_exposure=0.0,
    )

    assert exposure.red == zero
    assert exposure.blue == zero


def test_calculation_requires_dynamic_calibration() -> None:
    activity = DistanceSegmentActivity(
        state=shared_state(
            FightPhase.DISTANCE,
            owner=None,
        ),
        red=DistanceFighterActivity(
            sig_str_attempted=0,
            sig_str_landed=0,
            knockdowns=0,
        ),
        blue=DistanceFighterActivity(
            sig_str_attempted=0,
            sig_str_landed=0,
            knockdowns=0,
        ),
    )

    with pytest.raises(
        TypeError,
        match="calibration must be DynamicStateCalibration",
    ):
        calculate_segment_dynamic_exposure(
            activity,
            "invalid",
        )


def test_calculation_rejects_unsupported_activity_type() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "activity must be a supported phase "
            "activity record"
        ),
    ):
        calculate_segment_dynamic_exposure(
            object(),
            calibration(),
        )
