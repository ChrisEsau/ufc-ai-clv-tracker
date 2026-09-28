"""Tests for the V2 finish-enabled dynamic path runner."""

from __future__ import annotations

import pytest

import pipeline.simulation.rfs_mc_v2_shared_state.finish_path_runner as runner_module
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    FighterSide,
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
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_transition_effect_calibration import (
    DynamicTransitionEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_calibration import (
    FinishProbabilityCalibration,
    KnockoutFinishCalibration,
    SubmissionFinishCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
    FinishResult,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_probability import (
    SegmentFinishProbabilities,
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


def transition_parameters() -> FighterTransitionParameters:
    """Build a neutral shared-state transition profile."""

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


def phase_parameters() -> FighterPhaseParameters:
    """Build a complete valid phase-activity profile."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=4.0,
            sig_strike_accuracy=0.50,
            knockdown_probability_per_landed=0.02,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=1.50,
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


def dynamic_parameters() -> FighterDynamicParameters:
    """Build a neutral fighter dynamic-response profile."""

    return FighterDynamicParameters(
        fatigue_accumulation_resistance=0.0,
        fatigue_performance_resilience=0.0,
        recovery_ability=0.0,
        damage_resistance=0.0,
        acute_stress_resistance=0.0,
        acute_stress_recovery=0.0,
    )


def state_calibration(
    *,
    phase_workload: float = 0.0,
    round_break_fatigue_recovery: float = 0.0,
) -> DynamicStateCalibration:
    """Build controlled dynamic-state calibration."""

    return DynamicStateCalibration(
        phase_workload=PhaseWorkloadCalibration(
            distance=phase_workload,
            clinch_owner=phase_workload,
            clinch_defender=phase_workload,
            ground_owner=phase_workload,
            ground_defender=phase_workload,
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
            low_workload_threshold=0.0,
            segment_fatigue_recovery=0.0,
            round_break_fatigue_recovery=(
                round_break_fatigue_recovery
            ),
            segment_acute_stress_recovery=0.0,
            round_break_acute_stress_recovery=0.0,
        ),
    )


def zero_weights() -> StatePenaltyWeights:
    """Return a capability family with no dynamic penalty."""

    return StatePenaltyWeights(
        fatigue=0.0,
        damage=0.0,
        acute_stress=0.0,
    )


def phase_effect_calibration() -> DynamicEffectCalibration:
    """Build phase effects that preserve baseline capabilities."""

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


def transition_effect_calibration(
) -> DynamicTransitionEffectCalibration:
    """Build transition effects that preserve baselines."""

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


def no_finish_calibration() -> FinishProbabilityCalibration:
    """Build calibration producing zero finish probability."""

    return FinishProbabilityCalibration(
        knockout=KnockoutFinishCalibration(
            distance_landed_probability=0.0,
            distance_knockdown_probability=0.0,
            clinch_landed_probability=0.0,
            damaging_clinch_probability=0.0,
            ground_landed_probability=0.0,
            defender_fatigue_amplifier=1.0,
            defender_damage_amplifier=1.0,
            defender_acute_stress_amplifier=1.0,
            maximum_segment_probability=1.0,
        ),
        submission=SubmissionFinishCalibration(
            base_probability_per_attempt=0.0,
            position_quality_amplifier=1.0,
            minimum_submission_defense_effect_multiplier=0.10,
            defender_fatigue_amplifier=1.0,
            defender_damage_amplifier=1.0,
            defender_acute_stress_amplifier=1.0,
            maximum_probability_per_attempt=1.0,
            maximum_segment_probability=1.0,
        ),
    )


def run_finish_path(
    *,
    scheduled_rounds: int = 3,
    seed: int = 2026,
    selected_state_calibration: DynamicStateCalibration | None = None,
    selected_finish_calibration: FinishProbabilityCalibration | None = None,
):
    """Run one finish-enabled path using controlled defaults."""

    return runner_module.run_finish_enabled_dynamic_path(
        transition_parameters(),
        transition_parameters(),
        phase_parameters(),
        phase_parameters(),
        dynamic_parameters(),
        dynamic_parameters(),
        dynamic_state_calibration=(
            selected_state_calibration
            or state_calibration()
        ),
        phase_effect_calibration=phase_effect_calibration(),
        transition_effect_calibration=(
            transition_effect_calibration()
        ),
        finish_probability_calibration=(
            selected_finish_calibration
            or no_finish_calibration()
        ),
        scheduled_rounds=scheduled_rounds,
        seed=seed,
    )


def forced_knockout(
    probabilities: SegmentFinishProbabilities,
    *,
    winner: FighterSide = FighterSide.RED,
) -> FinishResult:
    """Build a legal forced KO/TKO in the current segment."""

    return FinishResult(
        state=probabilities.state,
        winner=winner,
        method=FinishMethod.KO_TKO,
        elapsed_seconds_in_segment=15,
    )


@pytest.mark.parametrize(
    ("scheduled_rounds", "expected_segments"),
    [
        (3, 30),
        (5, 50),
    ],
)
def test_zero_finish_calibration_reaches_scheduled_distance(
    scheduled_rounds: int,
    expected_segments: int,
) -> None:
    path = run_finish_path(
        scheduled_rounds=scheduled_rounds,
    )

    assert len(path.segments) == expected_segments
    assert path.finish is None
    assert path.reached_scheduled_distance is True


def test_same_seed_replays_identical_finish_enabled_path() -> None:
    first = run_finish_path(seed=707)
    second = run_finish_path(seed=707)

    assert first == second


def test_zero_finish_path_matches_existing_dynamic_runner() -> None:
    red_transition = transition_parameters()
    blue_transition = transition_parameters()
    red_phase = phase_parameters()
    blue_phase = phase_parameters()
    red_dynamic = dynamic_parameters()
    blue_dynamic = dynamic_parameters()
    selected_state_calibration = state_calibration(
        phase_workload=0.01,
        round_break_fatigue_recovery=0.05,
    )
    selected_phase_effect = phase_effect_calibration()
    selected_transition_effect = transition_effect_calibration()

    baseline = run_dynamic_activity_path(
        red_transition,
        blue_transition,
        red_phase,
        blue_phase,
        red_dynamic,
        blue_dynamic,
        dynamic_state_calibration=selected_state_calibration,
        phase_effect_calibration=selected_phase_effect,
        transition_effect_calibration=(
            selected_transition_effect
        ),
        scheduled_rounds=5,
        seed=919,
    )

    finish_enabled = (
        runner_module.run_finish_enabled_dynamic_path(
            red_transition,
            blue_transition,
            red_phase,
            blue_phase,
            red_dynamic,
            blue_dynamic,
            dynamic_state_calibration=(
                selected_state_calibration
            ),
            phase_effect_calibration=(
                selected_phase_effect
            ),
            transition_effect_calibration=(
                selected_transition_effect
            ),
            finish_probability_calibration=(
                no_finish_calibration()
            ),
            scheduled_rounds=5,
            seed=919,
        )
    )

    assert len(finish_enabled.segments) == len(
        baseline.segments
    )

    for expected, actual in zip(
        baseline.segments,
        finish_enabled.segments,
        strict=True,
    ):
        assert actual.state == expected.state
        assert (
            actual.dynamic_state_before
            == expected.dynamic_state_before
        )
        assert (
            actual.red_effective_phase
            == expected.red_effective_phase
        )
        assert (
            actual.blue_effective_phase
            == expected.blue_effective_phase
        )
        assert actual.activity == expected.activity
        assert actual.exposure == expected.exposure
        assert (
            actual.dynamic_state_after_activity
            == expected.dynamic_state_after_activity
        )
        assert (
            actual.red_effective_transition
            == expected.red_effective_transition
        )
        assert (
            actual.blue_effective_transition
            == expected.blue_effective_transition
        )
        assert actual.transition == expected.transition
        assert (
            actual.round_break_recovery_applied
            == expected.round_break_recovery_applied
        )
        assert (
            actual.dynamic_state_after_segment
            == expected.dynamic_state_after_segment
        )


def test_zero_finish_calibration_records_zero_probabilities() -> None:
    path = run_finish_path()

    for record in path.segments:
        assert record.finish is None
        assert record.finish_probabilities.red.ko_tko_probability == 0.0
        assert (
            record.finish_probabilities.red.submission_probability
            == 0.0
        )
        assert record.finish_probabilities.blue.ko_tko_probability == 0.0
        assert (
            record.finish_probabilities.blue.submission_probability
            == 0.0
        )


def test_unfinished_nonfinal_segments_retain_transitions() -> None:
    path = run_finish_path()

    for record in path.segments:
        if record.state.segment_number < SEGMENTS_PER_ROUND:
            assert record.transition is not None
            assert record.red_effective_transition is not None
            assert record.blue_effective_transition is not None


def test_unfinished_round_ends_have_no_transition() -> None:
    path = run_finish_path()

    for record in path.segments:
        if record.state.segment_number == SEGMENTS_PER_ROUND:
            assert record.transition is None
            assert record.red_effective_transition is None
            assert record.blue_effective_transition is None


def test_forced_first_segment_finish_stops_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "sample_segment_finish",
        lambda probabilities, rng: forced_knockout(
            probabilities
        ),
    )

    path = run_finish_path()

    assert len(path.segments) == 1
    assert path.finish is not None
    assert path.segments[0].finish == path.finish
    assert path.segments[0].transition is None
    assert (
        path.segments[0].round_break_recovery_applied
        is False
    )


def test_forced_later_finish_omits_all_following_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def sample_on_seventh_segment(
        probabilities: SegmentFinishProbabilities,
        rng: object,
    ) -> FinishResult | None:
        nonlocal call_count
        call_count += 1

        if call_count == 7:
            return forced_knockout(
                probabilities,
                winner=FighterSide.BLUE,
            )

        return None

    monkeypatch.setattr(
        runner_module,
        "sample_segment_finish",
        sample_on_seventh_segment,
    )

    path = run_finish_path()

    assert call_count == 7
    assert len(path.segments) == 7
    assert path.finish is not None
    assert path.finish.winner is FighterSide.BLUE
    assert path.finish.segment_number == 7
    assert path.segments[-1].transition is None


def test_finish_on_segment_ten_skips_round_break_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def sample_on_segment_ten(
        probabilities: SegmentFinishProbabilities,
        rng: object,
    ) -> FinishResult | None:
        nonlocal call_count
        call_count += 1

        if call_count == 10:
            return forced_knockout(probabilities)

        return None

    monkeypatch.setattr(
        runner_module,
        "sample_segment_finish",
        sample_on_segment_ten,
    )

    path = run_finish_path(
        selected_state_calibration=state_calibration(
            phase_workload=0.02,
            round_break_fatigue_recovery=0.20,
        ),
    )

    final_record = path.segments[-1]

    assert len(path.segments) == 10
    assert final_record.finish is not None
    assert final_record.round_break_recovery_applied is False
    assert (
        final_record.dynamic_state_after_segment
        == final_record.dynamic_state_after_activity
    )


def test_finish_in_round_two_preserves_prior_round_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def sample_on_segment_eleven(
        probabilities: SegmentFinishProbabilities,
        rng: object,
    ) -> FinishResult | None:
        nonlocal call_count
        call_count += 1

        if call_count == 11:
            return forced_knockout(probabilities)

        return None

    monkeypatch.setattr(
        runner_module,
        "sample_segment_finish",
        sample_on_segment_eleven,
    )

    path = run_finish_path(
        selected_state_calibration=state_calibration(
            phase_workload=0.02,
            round_break_fatigue_recovery=0.10,
        ),
    )

    assert len(path.segments) == 11
    assert (
        path.segments[9].round_break_recovery_applied
        is True
    )
    assert path.segments[10].state.round_number == 2
    assert path.segments[10].state.segment_number == 1
    assert path.segments[10].finish is not None


def test_transition_is_not_sampled_after_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "sample_segment_finish",
        lambda probabilities, rng: forced_knockout(
            probabilities
        ),
    )

    def unexpected_transition(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "transition sampling occurred after a finish"
        )

    monkeypatch.setattr(
        runner_module,
        "sample_and_apply_transition",
        unexpected_transition,
    )

    path = run_finish_path()

    assert len(path.segments) == 1
    assert path.finish is not None


def test_round_break_recovery_is_not_called_after_segment_ten_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def sample_on_segment_ten(
        probabilities: SegmentFinishProbabilities,
        rng: object,
    ) -> FinishResult | None:
        nonlocal call_count
        call_count += 1

        if call_count == 10:
            return forced_knockout(probabilities)

        return None

    monkeypatch.setattr(
        runner_module,
        "sample_segment_finish",
        sample_on_segment_ten,
    )

    def unexpected_recovery(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "round-break recovery occurred after a finish"
        )

    monkeypatch.setattr(
        runner_module,
        "apply_round_break_recovery",
        unexpected_recovery,
    )

    path = run_finish_path()

    assert len(path.segments) == 10
    assert path.finish is not None


def test_finish_probability_uses_post_activity_dynamic_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_states = []
    original_calculator = (
        runner_module.calculate_segment_finish_probabilities
    )

    def capture_state(
        activity: object,
        dynamic_state: object,
        red_effective_phase: object,
        blue_effective_phase: object,
        calibration: object,
    ) -> SegmentFinishProbabilities:
        captured_states.append(dynamic_state)

        return original_calculator(
            activity,
            dynamic_state,
            red_effective_phase,
            blue_effective_phase,
            calibration,
        )

    monkeypatch.setattr(
        runner_module,
        "calculate_segment_finish_probabilities",
        capture_state,
    )
    monkeypatch.setattr(
        runner_module,
        "sample_segment_finish",
        lambda probabilities, rng: forced_knockout(
            probabilities
        ),
    )

    path = run_finish_path(
        selected_state_calibration=state_calibration(
            phase_workload=0.10,
        ),
    )

    record = path.segments[0]

    assert captured_states == [
        record.dynamic_state_after_activity
    ]
    assert (
        record.dynamic_state_after_activity
        != record.dynamic_state_before
    )


def test_runner_does_not_mutate_baseline_parameters() -> None:
    red_transition = transition_parameters()
    blue_transition = transition_parameters()
    red_phase = phase_parameters()
    blue_phase = phase_parameters()

    original_red_transition = transition_parameters()
    original_blue_transition = transition_parameters()
    original_red_phase = phase_parameters()
    original_blue_phase = phase_parameters()

    runner_module.run_finish_enabled_dynamic_path(
        red_transition,
        blue_transition,
        red_phase,
        blue_phase,
        dynamic_parameters(),
        dynamic_parameters(),
        dynamic_state_calibration=state_calibration(
            phase_workload=0.02,
        ),
        phase_effect_calibration=phase_effect_calibration(),
        transition_effect_calibration=(
            transition_effect_calibration()
        ),
        finish_probability_calibration=(
            no_finish_calibration()
        ),
        scheduled_rounds=3,
        seed=5150,
    )

    assert red_transition == original_red_transition
    assert blue_transition == original_blue_transition
    assert red_phase == original_red_phase
    assert blue_phase == original_blue_phase


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
        match="scheduled_rounds must be 3 or 5",
    ):
        run_finish_path(
            scheduled_rounds=scheduled_rounds,
        )


def test_runner_seed_must_be_integer() -> None:
    with pytest.raises(
        TypeError,
        match="seed must be an integer",
    ):
        run_finish_path(
            seed=1.0,
        )


def test_runner_seed_cannot_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match="seed cannot be negative",
    ):
        run_finish_path(
            seed=-1,
        )


def test_runner_requires_finish_probability_calibration() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "finish_probability_calibration must be "
            "FinishProbabilityCalibration"
        ),
    ):
        runner_module.run_finish_enabled_dynamic_path(
            transition_parameters(),
            transition_parameters(),
            phase_parameters(),
            phase_parameters(),
            dynamic_parameters(),
            dynamic_parameters(),
            dynamic_state_calibration=state_calibration(),
            phase_effect_calibration=phase_effect_calibration(),
            transition_effect_calibration=(
                transition_effect_calibration()
            ),
            finish_probability_calibration="invalid",
            scheduled_rounds=3,
            seed=1,
        )
