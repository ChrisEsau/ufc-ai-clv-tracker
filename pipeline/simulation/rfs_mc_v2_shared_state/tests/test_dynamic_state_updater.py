"""Tests for V2 dynamic-state update logic."""

from dataclasses import replace

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SharedFightState,
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
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FightDynamicState,
    FighterDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state_updater import (
    apply_fighter_round_break_recovery,
    apply_round_break_recovery,
    calculate_resistance_multiplier,
    update_fight_dynamic_state,
    update_fighter_dynamic_state,
)


def calibration() -> DynamicStateCalibration:
    """Build a controlled dynamic-state calibration."""

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
            minimum_fatigue_accumulation_multiplier=0.25,
            minimum_damage_accumulation_multiplier=0.20,
            minimum_acute_stress_accumulation_multiplier=0.15,
        ),
        recovery=RecoveryCalibration(
            low_workload_threshold=0.10,
            segment_fatigue_recovery=0.08,
            round_break_fatigue_recovery=0.20,
            segment_acute_stress_recovery=0.10,
            round_break_acute_stress_recovery=0.40,
        ),
    )


def parameters(
    **overrides: float,
) -> FighterDynamicParameters:
    """Build valid fighter dynamic-response parameters."""

    baseline = FighterDynamicParameters(
        fatigue_accumulation_resistance=0.50,
        fatigue_performance_resilience=0.50,
        recovery_ability=0.50,
        damage_resistance=0.50,
        acute_stress_resistance=0.50,
        acute_stress_recovery=0.50,
    )

    return replace(
        baseline,
        **overrides,
    )


def fighter_state(
    *,
    fatigue: float = 0.0,
    damage: float = 0.0,
    acute_stress: float = 0.0,
) -> FighterDynamicState:
    """Build a fighter dynamic state."""

    return FighterDynamicState(
        fatigue=fatigue,
        damage=damage,
        acute_stress=acute_stress,
    )


def exposure(
    *,
    fatigue: float = 0.0,
    damage: float = 0.0,
    stress: float = 0.0,
) -> FighterSegmentExposure:
    """Build one fighter's raw segment exposure."""

    return FighterSegmentExposure(
        fatigue_workload=fatigue,
        persistent_damage_exposure=damage,
        acute_stress_exposure=stress,
    )


def segment_exposure(
    red: FighterSegmentExposure | None = None,
    blue: FighterSegmentExposure | None = None,
) -> SegmentDynamicExposure:
    """Build a two-fighter distance-segment exposure."""

    return SegmentDynamicExposure(
        state=SharedFightState(
            phase=FightPhase.DISTANCE,
            phase_owner=None,
            phase_age_segments=0,
            position_quality=0.0,
            round_number=1,
            segment_number=1,
        ),
        red=red if red is not None else exposure(),
        blue=blue if blue is not None else exposure(),
    )


def test_zero_resistance_returns_full_multiplier() -> None:
    assert calculate_resistance_multiplier(
        resistance=0.0,
        minimum_multiplier=0.25,
    ) == pytest.approx(1.0)


def test_max_resistance_returns_minimum_multiplier() -> None:
    assert calculate_resistance_multiplier(
        resistance=1.0,
        minimum_multiplier=0.25,
    ) == pytest.approx(0.25)


def test_midpoint_resistance_interpolates_linearly() -> None:
    assert calculate_resistance_multiplier(
        resistance=0.50,
        minimum_multiplier=0.20,
    ) == pytest.approx(0.60)


@pytest.mark.parametrize(
    ("resistance", "minimum_multiplier", "expected_name"),
    [
        ("invalid", 0.25, "resistance"),
        (0.50, "invalid", "minimum_multiplier"),
    ],
)
def test_resistance_multiplier_values_must_be_numeric(
    resistance: object,
    minimum_multiplier: object,
    expected_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{expected_name} must be numeric",
    ):
        calculate_resistance_multiplier(
            resistance,
            minimum_multiplier,
        )


@pytest.mark.parametrize(
    ("resistance", "minimum_multiplier", "expected_name"),
    [
        (float("nan"), 0.25, "resistance"),
        (0.50, float("inf"), "minimum_multiplier"),
    ],
)
def test_resistance_multiplier_values_must_be_finite(
    resistance: float,
    minimum_multiplier: float,
    expected_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{expected_name} must be finite",
    ):
        calculate_resistance_multiplier(
            resistance,
            minimum_multiplier,
        )


@pytest.mark.parametrize(
    ("resistance", "minimum_multiplier", "expected_name"),
    [
        (-0.01, 0.25, "resistance"),
        (1.01, 0.25, "resistance"),
        (0.50, -0.01, "minimum_multiplier"),
        (0.50, 1.01, "minimum_multiplier"),
    ],
)
def test_resistance_multiplier_values_must_be_in_range(
    resistance: float,
    minimum_multiplier: float,
    expected_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{expected_name} must be between 0 and 1",
    ):
        calculate_resistance_multiplier(
            resistance,
            minimum_multiplier,
        )


def test_zero_exposure_preserves_opening_state() -> None:
    result = update_fighter_dynamic_state(
        FighterDynamicState.opening_state(),
        exposure(),
        parameters(),
        calibration(),
    )

    assert result == FighterDynamicState.opening_state()


def test_fatigue_resistance_reduces_gain() -> None:
    low_resistance = update_fighter_dynamic_state(
        fighter_state(),
        exposure(fatigue=0.40),
        parameters(
            fatigue_accumulation_resistance=0.0,
        ),
        calibration(),
    )
    high_resistance = update_fighter_dynamic_state(
        fighter_state(),
        exposure(fatigue=0.40),
        parameters(
            fatigue_accumulation_resistance=1.0,
        ),
        calibration(),
    )

    assert low_resistance.fatigue == pytest.approx(0.40)
    assert high_resistance.fatigue == pytest.approx(0.10)
    assert high_resistance.fatigue < low_resistance.fatigue


def test_damage_resistance_reduces_gain() -> None:
    low_resistance = update_fighter_dynamic_state(
        fighter_state(),
        exposure(damage=0.40),
        parameters(
            damage_resistance=0.0,
        ),
        calibration(),
    )
    high_resistance = update_fighter_dynamic_state(
        fighter_state(),
        exposure(damage=0.40),
        parameters(
            damage_resistance=1.0,
        ),
        calibration(),
    )

    assert low_resistance.damage == pytest.approx(0.40)
    assert high_resistance.damage == pytest.approx(0.08)
    assert high_resistance.damage < low_resistance.damage


def test_acute_stress_resistance_reduces_gain() -> None:
    low_resistance = update_fighter_dynamic_state(
        fighter_state(),
        exposure(stress=0.40),
        parameters(
            acute_stress_resistance=0.0,
        ),
        calibration(),
    )
    high_resistance = update_fighter_dynamic_state(
        fighter_state(),
        exposure(stress=0.40),
        parameters(
            acute_stress_resistance=1.0,
        ),
        calibration(),
    )

    assert low_resistance.acute_stress == pytest.approx(0.40)
    assert high_resistance.acute_stress == pytest.approx(0.06)
    assert (
        high_resistance.acute_stress
        < low_resistance.acute_stress
    )


def test_low_workload_recovers_existing_fatigue() -> None:
    result = update_fighter_dynamic_state(
        fighter_state(fatigue=0.50),
        exposure(fatigue=0.0),
        parameters(recovery_ability=0.50),
        calibration(),
    )

    assert result.fatigue == pytest.approx(
        0.50 - 0.08 * 0.50
    )


def test_high_workload_does_not_recover_fatigue() -> None:
    result = update_fighter_dynamic_state(
        fighter_state(fatigue=0.50),
        exposure(fatigue=0.11),
        parameters(
            fatigue_accumulation_resistance=1.0,
            recovery_ability=1.0,
        ),
        calibration(),
    )

    assert result.fatigue == pytest.approx(
        0.50 + 0.11 * 0.25
    )


def test_low_workload_threshold_is_inclusive() -> None:
    result = update_fighter_dynamic_state(
        fighter_state(fatigue=0.50),
        exposure(fatigue=0.10),
        parameters(
            fatigue_accumulation_resistance=0.50,
            recovery_ability=0.50,
        ),
        calibration(),
    )

    expected_gain = 0.10 * 0.625
    expected_recovery = 0.08 * 0.50

    assert result.fatigue == pytest.approx(
        0.50
        - expected_recovery
        + expected_gain
    )


def test_acute_stress_decays_every_segment() -> None:
    result = update_fighter_dynamic_state(
        fighter_state(acute_stress=0.50),
        exposure(),
        parameters(acute_stress_recovery=0.50),
        calibration(),
    )

    assert result.acute_stress == pytest.approx(
        0.50 - 0.10 * 0.50
    )


def test_current_segment_stress_is_not_immediately_recovered() -> None:
    result = update_fighter_dynamic_state(
        fighter_state(acute_stress=0.0),
        exposure(stress=0.40),
        parameters(
            acute_stress_resistance=0.50,
            acute_stress_recovery=1.0,
        ),
        calibration(),
    )

    expected_multiplier = 0.575

    assert result.acute_stress == pytest.approx(
        0.40 * expected_multiplier
    )


def test_persistent_damage_does_not_recover() -> None:
    result = update_fighter_dynamic_state(
        fighter_state(damage=0.50),
        exposure(),
        parameters(),
        calibration(),
    )

    assert result.damage == pytest.approx(0.50)


def test_state_values_clip_to_one() -> None:
    result = update_fighter_dynamic_state(
        fighter_state(
            fatigue=0.95,
            damage=0.95,
            acute_stress=0.95,
        ),
        exposure(
            fatigue=10.0,
            damage=10.0,
            stress=10.0,
        ),
        parameters(
            fatigue_accumulation_resistance=0.0,
            damage_resistance=0.0,
            acute_stress_resistance=0.0,
        ),
        calibration(),
    )

    assert result == FighterDynamicState(
        fatigue=1.0,
        damage=1.0,
        acute_stress=1.0,
    )


def test_recovery_cannot_reduce_state_below_zero() -> None:
    result = update_fighter_dynamic_state(
        fighter_state(
            fatigue=0.01,
            acute_stress=0.02,
        ),
        exposure(),
        parameters(
            recovery_ability=1.0,
            acute_stress_recovery=1.0,
        ),
        calibration(),
    )

    assert result.fatigue == 0.0
    assert result.acute_stress == 0.0


def test_fatigue_performance_resilience_is_not_used_by_updater() -> None:
    low_resilience = update_fighter_dynamic_state(
        fighter_state(fatigue=0.20),
        exposure(fatigue=0.15),
        parameters(
            fatigue_performance_resilience=0.0,
        ),
        calibration(),
    )
    high_resilience = update_fighter_dynamic_state(
        fighter_state(fatigue=0.20),
        exposure(fatigue=0.15),
        parameters(
            fatigue_performance_resilience=1.0,
        ),
        calibration(),
    )

    assert low_resilience == high_resilience


def test_update_does_not_mutate_inputs() -> None:
    previous = fighter_state(
        fatigue=0.20,
        damage=0.30,
        acute_stress=0.40,
    )
    selected_exposure = exposure(
        fatigue=0.10,
        damage=0.20,
        stress=0.30,
    )

    original_previous = previous
    original_exposure = selected_exposure

    result = update_fighter_dynamic_state(
        previous,
        selected_exposure,
        parameters(),
        calibration(),
    )

    assert previous == original_previous
    assert selected_exposure == original_exposure
    assert result is not previous


@pytest.mark.parametrize(
    (
        "previous_state",
        "selected_exposure",
        "selected_parameters",
        "selected_calibration",
        "expected_message",
    ),
    [
        (
            "invalid",
            exposure(),
            parameters(),
            calibration(),
            "previous_state must be FighterDynamicState",
        ),
        (
            fighter_state(),
            "invalid",
            parameters(),
            calibration(),
            "exposure must be FighterSegmentExposure",
        ),
        (
            fighter_state(),
            exposure(),
            "invalid",
            calibration(),
            "parameters must be FighterDynamicParameters",
        ),
        (
            fighter_state(),
            exposure(),
            parameters(),
            "invalid",
            "calibration must be DynamicStateCalibration",
        ),
    ],
)
def test_update_fighter_requires_correct_types(
    previous_state: object,
    selected_exposure: object,
    selected_parameters: object,
    selected_calibration: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        update_fighter_dynamic_state(
            previous_state,
            selected_exposure,
            selected_parameters,
            selected_calibration,
        )


def test_fight_update_updates_both_fighters() -> None:
    result = update_fight_dynamic_state(
        FightDynamicState.opening_state(),
        segment_exposure(
            red=exposure(fatigue=0.40),
            blue=exposure(damage=0.50),
        ),
        parameters(
            fatigue_accumulation_resistance=0.0,
        ),
        parameters(
            damage_resistance=0.0,
        ),
        calibration(),
    )

    assert result.red.fatigue == pytest.approx(0.40)
    assert result.red.damage == 0.0

    assert result.blue.fatigue == 0.0
    assert result.blue.damage == pytest.approx(0.50)


@pytest.mark.parametrize(
    (
        "previous_state",
        "selected_exposure",
        "expected_message",
    ),
    [
        (
            "invalid",
            segment_exposure(),
            "previous_state must be FightDynamicState",
        ),
        (
            FightDynamicState.opening_state(),
            "invalid",
            "exposure must be SegmentDynamicExposure",
        ),
    ],
)
def test_fight_update_requires_correct_top_level_types(
    previous_state: object,
    selected_exposure: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        update_fight_dynamic_state(
            previous_state,
            selected_exposure,
            parameters(),
            parameters(),
            calibration(),
        )


def test_round_break_recovers_fatigue_and_stress_not_damage() -> None:
    result = apply_fighter_round_break_recovery(
        fighter_state(
            fatigue=0.60,
            damage=0.40,
            acute_stress=0.50,
        ),
        parameters(
            recovery_ability=0.50,
            acute_stress_recovery=0.25,
        ),
        calibration(),
    )

    assert result.fatigue == pytest.approx(
        0.60 - 0.20 * 0.50
    )
    assert result.damage == pytest.approx(0.40)
    assert result.acute_stress == pytest.approx(
        0.50 - 0.40 * 0.25
    )


def test_better_recovery_ability_increases_fatigue_recovery() -> None:
    poor_recovery = apply_fighter_round_break_recovery(
        fighter_state(fatigue=0.60),
        parameters(recovery_ability=0.0),
        calibration(),
    )
    strong_recovery = apply_fighter_round_break_recovery(
        fighter_state(fatigue=0.60),
        parameters(recovery_ability=1.0),
        calibration(),
    )

    assert poor_recovery.fatigue == pytest.approx(0.60)
    assert strong_recovery.fatigue == pytest.approx(0.40)


def test_better_acute_stress_recovery_increases_recovery() -> None:
    poor_recovery = apply_fighter_round_break_recovery(
        fighter_state(acute_stress=0.50),
        parameters(acute_stress_recovery=0.0),
        calibration(),
    )
    strong_recovery = apply_fighter_round_break_recovery(
        fighter_state(acute_stress=0.50),
        parameters(acute_stress_recovery=1.0),
        calibration(),
    )

    assert poor_recovery.acute_stress == pytest.approx(0.50)
    assert strong_recovery.acute_stress == pytest.approx(0.10)


def test_round_break_recovery_clamps_at_zero() -> None:
    result = apply_fighter_round_break_recovery(
        fighter_state(
            fatigue=0.05,
            damage=0.30,
            acute_stress=0.02,
        ),
        parameters(
            recovery_ability=1.0,
            acute_stress_recovery=1.0,
        ),
        calibration(),
    )

    assert result.fatigue == 0.0
    assert result.damage == pytest.approx(0.30)
    assert result.acute_stress == 0.0


@pytest.mark.parametrize(
    (
        "previous_state",
        "selected_parameters",
        "selected_calibration",
        "expected_message",
    ),
    [
        (
            "invalid",
            parameters(),
            calibration(),
            "previous_state must be FighterDynamicState",
        ),
        (
            fighter_state(),
            "invalid",
            calibration(),
            "parameters must be FighterDynamicParameters",
        ),
        (
            fighter_state(),
            parameters(),
            "invalid",
            "calibration must be DynamicStateCalibration",
        ),
    ],
)
def test_fighter_round_break_requires_correct_types(
    previous_state: object,
    selected_parameters: object,
    selected_calibration: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        apply_fighter_round_break_recovery(
            previous_state,
            selected_parameters,
            selected_calibration,
        )


def test_fight_round_break_updates_both_fighters() -> None:
    previous = FightDynamicState(
        red=fighter_state(
            fatigue=0.60,
            damage=0.20,
            acute_stress=0.50,
        ),
        blue=fighter_state(
            fatigue=0.60,
            damage=0.30,
            acute_stress=0.50,
        ),
    )

    result = apply_round_break_recovery(
        previous,
        parameters(
            recovery_ability=1.0,
            acute_stress_recovery=1.0,
        ),
        parameters(
            recovery_ability=0.0,
            acute_stress_recovery=0.0,
        ),
        calibration(),
    )

    assert result.red.fatigue == pytest.approx(0.40)
    assert result.red.damage == pytest.approx(0.20)
    assert result.red.acute_stress == pytest.approx(0.10)

    assert result.blue == previous.blue


def test_fight_round_break_requires_fight_state() -> None:
    with pytest.raises(
        TypeError,
        match="previous_state must be FightDynamicState",
    ):
        apply_round_break_recovery(
            "invalid",
            parameters(),
            parameters(),
            calibration(),
        )
