"""Tests for V2 dynamic full-path integration."""

from dataclasses import replace

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.activity_path_runner import (
    run_static_activity_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
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
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
    StatePenaltyWeights,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_path_runner import (
    run_dynamic_activity_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FightDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_transition_effect_calibration import (
    DynamicTransitionEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


def neutral_transition_parameters() -> FighterTransitionParameters:
    """Build a neutral transition profile."""

    return FighterTransitionParameters(
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


def distance_only_transition_parameters() -> FighterTransitionParameters:
    """Build a profile that remains at distance."""

    return FighterTransitionParameters(
        distance_retention=1.0,
        clinch_entry_tendency=0.0,
        clinch_entry_resistance=0.0,
        takedown_entry_tendency=0.0,
        takedown_completion_ability=0.0,
        takedown_resistance=0.0,
        takedown_persistence=0.0,
        failed_takedown_persistence=0.0,
        clinch_retention=0.0,
        clinch_escape_ability=0.0,
        ground_retention=0.0,
        ground_escape_ability=0.0,
        reversal_ability=0.0,
        phase_imposition=0.0,
        phase_resistance=0.0,
    )


def phase_parameters(
    *,
    distance_attempt_rate: float = 4.0,
    distance_accuracy: float = 0.50,
) -> FighterPhaseParameters:
    """Build complete baseline phase parameters."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=distance_attempt_rate,
            sig_strike_accuracy=distance_accuracy,
            knockdown_probability_per_landed=0.0,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=1.5,
            clinch_strike_accuracy=0.50,
            control_seconds_mean=8.0,
            damaging_clinch_probability=0.05,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=2.0,
            ground_strike_accuracy=0.50,
            control_seconds_mean=15.0,
            submission_attempt_rate=0.20,
            position_advancement_probability=0.25,
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=0.20,
            reversal_attempt_rate=0.08,
            scramble_attempt_rate=0.15,
            submission_defense=0.70,
        ),
    )


def dynamic_parameters(
    **overrides: float,
) -> FighterDynamicParameters:
    """Build fighter dynamic-response parameters."""

    baseline = FighterDynamicParameters(
        fatigue_accumulation_resistance=0.0,
        fatigue_performance_resilience=0.0,
        recovery_ability=0.0,
        damage_resistance=0.0,
        acute_stress_resistance=0.0,
        acute_stress_recovery=0.0,
    )

    return replace(
        baseline,
        **overrides,
    )


def state_calibration(
    *,
    distance_workload: float = 0.0,
    distance_landed_damage: float = 0.0,
    distance_landed_stress: float = 0.0,
    round_break_fatigue_recovery: float = 0.0,
    round_break_stress_recovery: float = 0.0,
) -> DynamicStateCalibration:
    """Build controlled dynamic-state calibration."""

    return DynamicStateCalibration(
        phase_workload=PhaseWorkloadCalibration(
            distance=distance_workload,
            clinch_owner=distance_workload,
            clinch_defender=distance_workload,
            ground_owner=distance_workload,
            ground_defender=distance_workload,
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
            distance_landed_damage=distance_landed_damage,
            clinch_landed_damage=0.0,
            damaging_clinch_bonus_damage=0.0,
            ground_landed_damage=0.0,
            knockdown_damage=0.0,
            distance_landed_stress=distance_landed_stress,
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
            low_workload_threshold=0.0,
            segment_fatigue_recovery=0.0,
            round_break_fatigue_recovery=(
                round_break_fatigue_recovery
            ),
            segment_acute_stress_recovery=0.0,
            round_break_acute_stress_recovery=(
                round_break_stress_recovery
            ),
        ),
    )


def zero_weights() -> StatePenaltyWeights:
    """Return a penalty family with no dynamic effects."""

    return StatePenaltyWeights(
        fatigue=0.0,
        damage=0.0,
        acute_stress=0.0,
    )


def fatigue_weights() -> StatePenaltyWeights:
    """Return a penalty family driven only by fatigue."""

    return StatePenaltyWeights(
        fatigue=1.0,
        damage=0.0,
        acute_stress=0.0,
    )


def zero_phase_effect_calibration() -> DynamicEffectCalibration:
    """Build phase-effect calibration that preserves baselines."""

    return DynamicEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_capability_multiplier=0.10,
        output=zero_weights(),
        accuracy=zero_weights(),
        power=zero_weights(),
        control=zero_weights(),
        grappling=zero_weights(),
        defense=zero_weights(),
    )


def fatigue_phase_effect_calibration() -> DynamicEffectCalibration:
    """Build calibration where fatigue reduces output only."""

    return DynamicEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_capability_multiplier=0.10,
        output=fatigue_weights(),
        accuracy=zero_weights(),
        power=zero_weights(),
        control=zero_weights(),
        grappling=zero_weights(),
        defense=zero_weights(),
    )


def zero_transition_effect_calibration(
) -> DynamicTransitionEffectCalibration:
    """Build transition-effect calibration preserving baselines."""

    return DynamicTransitionEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_transition_multiplier=0.10,
        entry=zero_weights(),
        completion=zero_weights(),
        retention=zero_weights(),
        escape=zero_weights(),
        reversal=zero_weights(),
        persistence=zero_weights(),
        imposition=zero_weights(),
        resistance=zero_weights(),
    )


def fatigue_transition_effect_calibration(
) -> DynamicTransitionEffectCalibration:
    """Build calibration where fatigue reduces retention only."""

    return DynamicTransitionEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_transition_multiplier=0.10,
        entry=zero_weights(),
        completion=zero_weights(),
        retention=fatigue_weights(),
        escape=zero_weights(),
        reversal=zero_weights(),
        persistence=zero_weights(),
        imposition=zero_weights(),
        resistance=zero_weights(),
    )


def run_path(
    *,
    red_transition: FighterTransitionParameters | None = None,
    blue_transition: FighterTransitionParameters | None = None,
    red_phase: FighterPhaseParameters | None = None,
    blue_phase: FighterPhaseParameters | None = None,
    red_dynamic: FighterDynamicParameters | None = None,
    blue_dynamic: FighterDynamicParameters | None = None,
    selected_state_calibration: DynamicStateCalibration | None = None,
    phase_effect: DynamicEffectCalibration | None = None,
    transition_effect: (
        DynamicTransitionEffectCalibration | None
    ) = None,
    scheduled_rounds: int = 3,
    seed: int = 42,
):
    """Run one dynamic path with controlled defaults."""

    return run_dynamic_activity_path(
        red_transition or neutral_transition_parameters(),
        blue_transition or neutral_transition_parameters(),
        red_phase or phase_parameters(),
        blue_phase or phase_parameters(),
        red_dynamic or dynamic_parameters(),
        blue_dynamic or dynamic_parameters(),
        dynamic_state_calibration=(
            selected_state_calibration
            or state_calibration()
        ),
        phase_effect_calibration=(
            phase_effect
            or zero_phase_effect_calibration()
        ),
        transition_effect_calibration=(
            transition_effect
            or zero_transition_effect_calibration()
        ),
        scheduled_rounds=scheduled_rounds,
        seed=seed,
    )


@pytest.mark.parametrize(
    ("scheduled_rounds", "expected_segments"),
    [
        (3, 30),
        (5, 50),
    ],
)
def test_path_contains_all_scheduled_segments(
    scheduled_rounds: int,
    expected_segments: int,
) -> None:
    path = run_path(
        scheduled_rounds=scheduled_rounds,
    )

    assert len(path.segments) == expected_segments


def test_same_seed_produces_identical_dynamic_path() -> None:
    first = run_path(seed=2026)
    second = run_path(seed=2026)

    assert first == second


def test_zero_dynamic_system_matches_static_path() -> None:
    red_transition = neutral_transition_parameters()
    blue_transition = neutral_transition_parameters()
    red_phase = phase_parameters()
    blue_phase = phase_parameters()

    static = run_static_activity_path(
        red_transition,
        blue_transition,
        red_phase,
        blue_phase,
        scheduled_rounds=5,
        seed=5150,
    )

    dynamic = run_path(
        red_transition=red_transition,
        blue_transition=blue_transition,
        red_phase=red_phase,
        blue_phase=blue_phase,
        scheduled_rounds=5,
        seed=5150,
    )

    assert tuple(
        record.state
        for record in dynamic.segments
    ) == tuple(
        record.state
        for record in static.segments
    )

    assert tuple(
        record.activity
        for record in dynamic.segments
    ) == tuple(
        record.activity
        for record in static.segments
    )

    assert tuple(
        record.transition
        for record in dynamic.segments
    ) == tuple(
        record.transition
        for record in static.segments
    )

    assert all(
        record.dynamic_state_before
        == FightDynamicState.opening_state()
        for record in dynamic.segments
    )


def test_dynamic_state_result_feeds_following_segment() -> None:
    path = run_path(
        selected_state_calibration=state_calibration(
            distance_workload=0.02,
        ),
    )

    for current, following in zip(
        path.segments,
        path.segments[1:],
    ):
        assert (
            current.dynamic_state_after_segment
            == following.dynamic_state_before
        )


def test_transition_result_feeds_following_segment() -> None:
    path = run_path()

    for index, record in enumerate(path.segments):
        if record.state.segment_number == SEGMENTS_PER_ROUND:
            assert record.transition is None
            continue

        assert record.transition is not None
        assert (
            record.transition.next_state
            == path.segments[index + 1].state
        )


def test_every_round_begins_at_distance() -> None:
    path = run_path(
        scheduled_rounds=5,
    )

    openings = [
        record
        for record in path.segments
        if record.state.segment_number == 1
    ]

    assert len(openings) == 5

    for record in openings:
        assert record.state.phase is FightPhase.DISTANCE
        assert record.state.phase_owner is None
        assert record.state.position_quality == 0.0
        assert record.state.phase_age_segments == 0


def test_round_break_recovery_occurs_after_nonfinal_rounds() -> None:
    path = run_path(
        red_transition=distance_only_transition_parameters(),
        blue_transition=distance_only_transition_parameters(),
        red_dynamic=dynamic_parameters(
            recovery_ability=1.0,
        ),
        blue_dynamic=dynamic_parameters(
            recovery_ability=1.0,
        ),
        selected_state_calibration=state_calibration(
            distance_workload=0.05,
            round_break_fatigue_recovery=0.20,
        ),
    )

    round_one_end = path.segments[9]
    round_two_opening = path.segments[10]

    assert round_one_end.round_break_recovery_applied is True
    assert round_one_end.dynamic_state_after_activity.red.fatigue == (
        pytest.approx(0.50)
    )
    assert round_one_end.dynamic_state_after_segment.red.fatigue == (
        pytest.approx(0.30)
    )

    assert (
        round_two_opening.dynamic_state_before
        == round_one_end.dynamic_state_after_segment
    )

    assert path.segments[19].round_break_recovery_applied is True


def test_final_round_does_not_apply_round_break_recovery() -> None:
    path = run_path(
        red_transition=distance_only_transition_parameters(),
        blue_transition=distance_only_transition_parameters(),
        selected_state_calibration=state_calibration(
            distance_workload=0.02,
            round_break_fatigue_recovery=0.20,
        ),
    )

    final_segment = path.segments[-1]

    assert final_segment.round_break_recovery_applied is False
    assert (
        final_segment.dynamic_state_after_segment
        == final_segment.dynamic_state_after_activity
    )


def test_activity_and_exposure_use_authoritative_state() -> None:
    path = run_path(
        scheduled_rounds=5,
    )

    for record in path.segments:
        assert record.activity.state == record.state
        assert record.exposure.state == record.state


def test_activity_type_matches_shared_phase() -> None:
    path = run_path(
        scheduled_rounds=5,
        seed=707,
    )

    for record in path.segments:
        if record.state.phase is FightPhase.DISTANCE:
            assert isinstance(
                record.activity,
                DistanceSegmentActivity,
            )

        elif record.state.phase is FightPhase.CLINCH:
            assert isinstance(
                record.activity,
                ClinchSegmentActivity,
            )

        elif record.state.phase is FightPhase.GROUND:
            assert isinstance(
                record.activity,
                GroundSegmentActivity,
            )


def test_effective_phase_parameters_use_pre_activity_state() -> None:
    baseline = phase_parameters(
        distance_attempt_rate=4.0,
    )

    path = run_path(
        red_transition=distance_only_transition_parameters(),
        blue_transition=distance_only_transition_parameters(),
        red_phase=baseline,
        blue_phase=baseline,
        selected_state_calibration=state_calibration(
            distance_workload=0.10,
        ),
        phase_effect=fatigue_phase_effect_calibration(),
    )

    first = path.segments[0]
    second = path.segments[1]

    assert first.dynamic_state_before.red.fatigue == 0.0
    assert (
        first.red_effective_phase
        .distance.sig_strike_attempt_rate
        == pytest.approx(4.0)
    )

    assert first.dynamic_state_after_activity.red.fatigue == (
        pytest.approx(0.10)
    )
    assert second.dynamic_state_before.red.fatigue == (
        pytest.approx(0.10)
    )
    assert (
        second.red_effective_phase
        .distance.sig_strike_attempt_rate
        == pytest.approx(4.0 * 0.90)
    )


def test_effective_transition_uses_post_activity_state() -> None:
    baseline = distance_only_transition_parameters()

    path = run_path(
        red_transition=baseline,
        blue_transition=baseline,
        selected_state_calibration=state_calibration(
            distance_workload=0.10,
        ),
        transition_effect=(
            fatigue_transition_effect_calibration()
        ),
    )

    first = path.segments[0]

    assert first.dynamic_state_before.red.fatigue == 0.0
    assert first.dynamic_state_after_activity.red.fatigue == (
        pytest.approx(0.10)
    )

    assert first.red_effective_transition is not None
    assert first.red_effective_transition.distance_retention == (
        pytest.approx(1.0 * 0.90)
    )


def test_fatigue_resistance_reduces_accumulation_for_equal_path() -> None:
    low_resistance = run_path(
        red_transition=distance_only_transition_parameters(),
        blue_transition=distance_only_transition_parameters(),
        red_dynamic=dynamic_parameters(
            fatigue_accumulation_resistance=0.0,
        ),
        selected_state_calibration=state_calibration(
            distance_workload=0.20,
        ),
    )

    high_resistance = run_path(
        red_transition=distance_only_transition_parameters(),
        blue_transition=distance_only_transition_parameters(),
        red_dynamic=dynamic_parameters(
            fatigue_accumulation_resistance=1.0,
        ),
        selected_state_calibration=state_calibration(
            distance_workload=0.20,
        ),
    )

    assert (
        low_resistance.segments[0]
        .dynamic_state_after_activity.red.fatigue
        == pytest.approx(0.20)
    )
    assert (
        high_resistance.segments[0]
        .dynamic_state_after_activity.red.fatigue
        == pytest.approx(0.05)
    )


def test_performance_resilience_preserves_later_output() -> None:
    baseline = phase_parameters(
        distance_attempt_rate=4.0,
    )

    low_resilience = run_path(
        red_transition=distance_only_transition_parameters(),
        blue_transition=distance_only_transition_parameters(),
        red_phase=baseline,
        red_dynamic=dynamic_parameters(
            fatigue_performance_resilience=0.0,
        ),
        selected_state_calibration=state_calibration(
            distance_workload=0.10,
        ),
        phase_effect=fatigue_phase_effect_calibration(),
    )

    high_resilience = run_path(
        red_transition=distance_only_transition_parameters(),
        blue_transition=distance_only_transition_parameters(),
        red_phase=baseline,
        red_dynamic=dynamic_parameters(
            fatigue_performance_resilience=1.0,
        ),
        selected_state_calibration=state_calibration(
            distance_workload=0.10,
        ),
        phase_effect=fatigue_phase_effect_calibration(),
    )

    low_output = (
        low_resilience.segments[1]
        .red_effective_phase.distance.sig_strike_attempt_rate
    )
    high_output = (
        high_resilience.segments[1]
        .red_effective_phase.distance.sig_strike_attempt_rate
    )

    assert low_output == pytest.approx(4.0 * 0.90)
    assert high_output == pytest.approx(4.0 * 0.98)
    assert high_output > low_output


def test_zero_effect_weights_isolate_random_timeline() -> None:
    zero_state = run_path(
        selected_state_calibration=state_calibration(),
        seed=808,
    )
    accumulating_state = run_path(
        selected_state_calibration=state_calibration(
            distance_workload=0.05,
        ),
        seed=808,
    )

    assert tuple(
        record.state
        for record in zero_state.segments
    ) == tuple(
        record.state
        for record in accumulating_state.segments
    )

    assert tuple(
        record.activity
        for record in zero_state.segments
    ) == tuple(
        record.activity
        for record in accumulating_state.segments
    )

    assert tuple(
        record.transition
        for record in zero_state.segments
    ) == tuple(
        record.transition
        for record in accumulating_state.segments
    )

    assert (
        zero_state.segments[-1].dynamic_state_after_segment
        != accumulating_state.segments[-1]
        .dynamic_state_after_segment
    )


def test_round_break_preserves_persistent_damage() -> None:
    path = run_path(
        red_transition=distance_only_transition_parameters(),
        blue_transition=distance_only_transition_parameters(),
        red_phase=phase_parameters(
            distance_attempt_rate=0.0,
            distance_accuracy=1.0,
        ),
        blue_phase=phase_parameters(
            distance_attempt_rate=20.0,
            distance_accuracy=1.0,
        ),
        selected_state_calibration=state_calibration(
            distance_landed_damage=0.01,
            round_break_fatigue_recovery=0.20,
        ),
        seed=909,
    )

    round_one_end = path.segments[9]

    assert (
        round_one_end
        .dynamic_state_after_activity.red.damage
        > 0.0
    )
    assert (
        round_one_end
        .dynamic_state_after_segment.red.damage
        == round_one_end
        .dynamic_state_after_activity.red.damage
    )


def test_baseline_parameters_remain_unchanged() -> None:
    red_transition = neutral_transition_parameters()
    blue_transition = neutral_transition_parameters()
    red_phase = phase_parameters()
    blue_phase = phase_parameters()

    original_red_transition = neutral_transition_parameters()
    original_blue_transition = neutral_transition_parameters()
    original_red_phase = phase_parameters()
    original_blue_phase = phase_parameters()

    run_path(
        red_transition=red_transition,
        blue_transition=blue_transition,
        red_phase=red_phase,
        blue_phase=blue_phase,
        selected_state_calibration=state_calibration(
            distance_workload=0.05,
        ),
        phase_effect=fatigue_phase_effect_calibration(),
        transition_effect=(
            fatigue_transition_effect_calibration()
        ),
    )

    assert red_transition == original_red_transition
    assert blue_transition == original_blue_transition
    assert red_phase == original_red_phase
    assert blue_phase == original_blue_phase


def test_nonfinal_segments_have_effective_transition_parameters() -> None:
    path = run_path()

    for record in path.segments:
        if record.state.segment_number < SEGMENTS_PER_ROUND:
            assert record.red_effective_transition is not None
            assert record.blue_effective_transition is not None
            assert record.transition is not None
        else:
            assert record.red_effective_transition is None
            assert record.blue_effective_transition is None
            assert record.transition is None


@pytest.mark.parametrize(
    "scheduled_rounds",
    [
        2,
        4,
    ],
)
def test_runner_rejects_unsupported_round_count(
    scheduled_rounds: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="scheduled_rounds",
    ):
        run_path(
            scheduled_rounds=scheduled_rounds,
        )


def test_runner_rejects_negative_seed() -> None:
    with pytest.raises(
        ValueError,
        match="seed",
    ):
        run_path(
            seed=-1,
        )
