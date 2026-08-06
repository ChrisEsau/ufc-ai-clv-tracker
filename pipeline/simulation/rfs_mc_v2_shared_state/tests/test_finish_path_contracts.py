"""Tests for V2 finish-enabled dynamic path contracts."""

from dataclasses import FrozenInstanceError, replace

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceFighterActivity,
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_exposure import (
    FighterSegmentExposure,
    SegmentDynamicExposure,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FightDynamicState,
    FighterDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
    FinishResult,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_contracts import (
    FinishEnabledDynamicPath,
    FinishEvaluatedPathSegment,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_probability import (
    FighterSegmentFinishProbabilities,
    SegmentFinishProbabilities,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    SharedTransition,
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


def distance_state(
    round_number: int,
    segment_number: int,
) -> SharedFightState:
    """Build one valid distance state in a continuous round timeline."""

    return SharedFightState(
        phase=FightPhase.DISTANCE,
        phase_owner=None,
        phase_age_segments=segment_number - 1,
        position_quality=0.0,
        round_number=round_number,
        segment_number=segment_number,
    )


def clinch_state(
    round_number: int,
    segment_number: int,
    *,
    owner: FighterSide = FighterSide.RED,
) -> SharedFightState:
    """Build one valid newly entered clinch state."""

    return SharedFightState(
        phase=FightPhase.CLINCH,
        phase_owner=owner,
        phase_age_segments=0,
        position_quality=0.50,
        round_number=round_number,
        segment_number=segment_number,
    )


def phase_parameters() -> FighterPhaseParameters:
    """Build complete valid phase parameters."""

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


def transition_parameters() -> FighterTransitionParameters:
    """Build complete valid transition parameters."""

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


def zero_dynamic_state() -> FightDynamicState:
    """Return fresh two-fighter dynamic state."""

    return FightDynamicState.opening_state()


def altered_dynamic_state() -> FightDynamicState:
    """Return a valid non-opening dynamic state."""

    return FightDynamicState(
        red=FighterDynamicState(
            fatigue=0.20,
            damage=0.10,
            acute_stress=0.05,
        ),
        blue=FighterDynamicState.opening_state(),
    )


def distance_activity(
    state: SharedFightState,
) -> DistanceSegmentActivity:
    """Build zero activity for one distance state."""

    return DistanceSegmentActivity(
        state=state,
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


def zero_exposure(
    state: SharedFightState,
) -> SegmentDynamicExposure:
    """Build zero dynamic exposure for both fighters."""

    fighter = FighterSegmentExposure(
        fatigue_workload=0.0,
        persistent_damage_exposure=0.0,
        acute_stress_exposure=0.0,
    )

    return SegmentDynamicExposure(
        state=state,
        red=fighter,
        blue=fighter,
    )


def finish_probabilities(
    state: SharedFightState,
    *,
    finish: FinishResult | None = None,
) -> SegmentFinishProbabilities:
    """Build legal probabilities corresponding to an optional finish."""

    red_ko = 0.0
    blue_ko = 0.0

    if finish is not None:
        if finish.winner is FighterSide.RED:
            red_ko = 1.0
        else:
            blue_ko = 1.0

    return SegmentFinishProbabilities(
        state=state,
        red=FighterSegmentFinishProbabilities(
            ko_tko_probability=red_ko,
            submission_probability=0.0,
        ),
        blue=FighterSegmentFinishProbabilities(
            ko_tko_probability=blue_ko,
            submission_probability=0.0,
        ),
    )


def stay_transition(
    state: SharedFightState,
) -> SharedTransition:
    """Build a legal distance stay transition."""

    next_state = distance_state(
        state.round_number,
        state.segment_number + 1,
    )

    return SharedTransition(
        previous_state=state,
        next_state=next_state,
        event=TransitionEvent.STAY,
        actor=None,
    )


def clinch_entry_transition(
    state: SharedFightState,
) -> SharedTransition:
    """Build a legal distance-to-clinch transition."""

    return SharedTransition(
        previous_state=state,
        next_state=clinch_state(
            state.round_number,
            state.segment_number + 1,
        ),
        event=TransitionEvent.CLINCH_ENTRY,
        actor=FighterSide.RED,
    )


def knockout_finish(
    state: SharedFightState,
    *,
    winner: FighterSide = FighterSide.RED,
) -> FinishResult:
    """Build one legal KO/TKO result."""

    return FinishResult(
        state=state,
        winner=winner,
        method=FinishMethod.KO_TKO,
        elapsed_seconds_in_segment=15,
    )


def path_segment(
    round_number: int,
    segment_number: int,
    *,
    finish: FinishResult | None = None,
    round_break_recovery_applied: bool = False,
    dynamic_state_before: FightDynamicState | None = None,
    dynamic_state_after_activity: FightDynamicState | None = None,
    dynamic_state_after_segment: FightDynamicState | None = None,
) -> FinishEvaluatedPathSegment:
    """Build one valid distance-only finish-evaluated segment."""

    state = distance_state(
        round_number,
        segment_number,
    )

    before = (
        dynamic_state_before
        or zero_dynamic_state()
    )
    after_activity = (
        dynamic_state_after_activity
        or zero_dynamic_state()
    )
    after_segment = (
        dynamic_state_after_segment
        or after_activity
    )

    is_round_end = segment_number == 10
    has_transition = (
        not is_round_end
        and finish is None
    )

    effective_transition = (
        transition_parameters()
        if has_transition
        else None
    )
    transition = (
        stay_transition(state)
        if has_transition
        else None
    )

    return FinishEvaluatedPathSegment(
        state=state,
        dynamic_state_before=before,
        red_effective_phase=phase_parameters(),
        blue_effective_phase=phase_parameters(),
        activity=distance_activity(state),
        exposure=zero_exposure(state),
        dynamic_state_after_activity=after_activity,
        finish_probabilities=finish_probabilities(
            state,
            finish=finish,
        ),
        finish=finish,
        red_effective_transition=effective_transition,
        blue_effective_transition=effective_transition,
        transition=transition,
        round_break_recovery_applied=(
            round_break_recovery_applied
        ),
        dynamic_state_after_segment=after_segment,
    )


def unfinished_segments(
    scheduled_rounds: int = 3,
) -> tuple[FinishEvaluatedPathSegment, ...]:
    """Build a complete distance-only scheduled path."""

    records: list[FinishEvaluatedPathSegment] = []

    for round_number in range(
        1,
        scheduled_rounds + 1,
    ):
        for segment_number in range(1, 11):
            records.append(
                path_segment(
                    round_number,
                    segment_number,
                    round_break_recovery_applied=(
                        segment_number == 10
                        and round_number < scheduled_rounds
                    ),
                )
            )

    return tuple(records)


def finished_segments(
    finishing_segment: int = 3,
) -> tuple[
    tuple[FinishEvaluatedPathSegment, ...],
    FinishResult,
]:
    """Build an early round-one finish path."""

    records: list[FinishEvaluatedPathSegment] = []

    for segment_number in range(
        1,
        finishing_segment,
    ):
        records.append(
            path_segment(
                1,
                segment_number,
            )
        )

    state = distance_state(
        1,
        finishing_segment,
    )
    finish = knockout_finish(state)

    records.append(
        path_segment(
            1,
            finishing_segment,
            finish=finish,
        )
    )

    return tuple(records), finish


def test_valid_unfinished_nonfinal_segment() -> None:
    selected = path_segment(
        1,
        1,
    )

    assert selected.finish is None
    assert selected.transition is not None
    assert selected.red_effective_transition is not None
    assert selected.blue_effective_transition is not None
    assert selected.round_break_recovery_applied is False


def test_valid_finishing_segment() -> None:
    state = distance_state(
        1,
        3,
    )
    finish = knockout_finish(state)

    selected = path_segment(
        1,
        3,
        finish=finish,
    )

    assert selected.finish is finish
    assert selected.transition is None
    assert selected.red_effective_transition is None
    assert selected.blue_effective_transition is None
    assert selected.round_break_recovery_applied is False


def test_valid_round_end_with_recovery() -> None:
    after_activity = altered_dynamic_state()
    after_recovery = zero_dynamic_state()

    selected = path_segment(
        1,
        10,
        round_break_recovery_applied=True,
        dynamic_state_after_activity=after_activity,
        dynamic_state_after_segment=after_recovery,
    )

    assert selected.transition is None
    assert selected.round_break_recovery_applied is True
    assert selected.dynamic_state_after_segment is after_recovery


def test_valid_final_round_end_without_recovery() -> None:
    after_activity = altered_dynamic_state()

    selected = path_segment(
        3,
        10,
        dynamic_state_after_activity=after_activity,
        dynamic_state_after_segment=after_activity,
    )

    assert selected.transition is None
    assert selected.round_break_recovery_applied is False
    assert (
        selected.dynamic_state_after_segment
        == selected.dynamic_state_after_activity
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_message"),
    [
        (
            "state",
            "invalid",
            "state must be SharedFightState",
        ),
        (
            "dynamic_state_before",
            "invalid",
            "dynamic_state_before must be FightDynamicState",
        ),
        (
            "red_effective_phase",
            "invalid",
            "red_effective_phase must be FighterPhaseParameters",
        ),
        (
            "blue_effective_phase",
            "invalid",
            "blue_effective_phase must be FighterPhaseParameters",
        ),
        (
            "activity",
            "invalid",
            "activity must be a supported phase segment activity",
        ),
        (
            "exposure",
            "invalid",
            "exposure must be SegmentDynamicExposure",
        ),
        (
            "dynamic_state_after_activity",
            "invalid",
            "dynamic_state_after_activity must be FightDynamicState",
        ),
        (
            "finish_probabilities",
            "invalid",
            (
                "finish_probabilities must be "
                "SegmentFinishProbabilities"
            ),
        ),
        (
            "finish",
            "invalid",
            "finish must be FinishResult or None",
        ),
        (
            "red_effective_transition",
            "invalid",
            (
                "red_effective_transition must be "
                "FighterTransitionParameters or None"
            ),
        ),
        (
            "blue_effective_transition",
            "invalid",
            (
                "blue_effective_transition must be "
                "FighterTransitionParameters or None"
            ),
        ),
        (
            "transition",
            "invalid",
            "transition must be SharedTransition or None",
        ),
        (
            "round_break_recovery_applied",
            1,
            (
                "round_break_recovery_applied "
                "must be boolean"
            ),
        ),
        (
            "dynamic_state_after_segment",
            "invalid",
            (
                "dynamic_state_after_segment "
                "must be FightDynamicState"
            ),
        ),
    ],
)
def test_segment_fields_require_correct_types(
    field_name: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    selected = path_segment(
        1,
        1,
    )

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        replace(
            selected,
            **{
                field_name: invalid_value,
            },
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_message"),
    [
        (
            "activity",
            distance_activity(
                distance_state(
                    1,
                    2,
                )
            ),
            "activity state must equal the segment state",
        ),
        (
            "exposure",
            zero_exposure(
                distance_state(
                    1,
                    2,
                )
            ),
            "exposure state must equal the segment state",
        ),
        (
            "finish_probabilities",
            finish_probabilities(
                distance_state(
                    1,
                    2,
                )
            ),
            (
                "finish-probability state "
                "must equal segment state"
            ),
        ),
    ],
)
def test_segment_payloads_must_share_authoritative_state(
    field_name: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    selected = path_segment(
        1,
        1,
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        replace(
            selected,
            **{
                field_name: invalid_value,
            },
        )


def test_finish_state_must_equal_segment_state() -> None:
    state = distance_state(
        1,
        3,
    )
    selected = path_segment(
        1,
        3,
        finish=knockout_finish(state),
    )

    with pytest.raises(
        ValueError,
        match="finish state must equal segment state",
    ):
        replace(
            selected,
            finish=knockout_finish(
                distance_state(
                    1,
                    4,
                )
            ),
        )


def test_finishing_segment_cannot_have_transition() -> None:
    state = distance_state(
        1,
        3,
    )
    selected = path_segment(
        1,
        3,
        finish=knockout_finish(state),
    )
    effective = transition_parameters()

    with pytest.raises(
        ValueError,
        match="finishing segment cannot have a transition",
    ):
        replace(
            selected,
            red_effective_transition=effective,
            blue_effective_transition=effective,
            transition=stay_transition(state),
        )


def test_finishing_segment_cannot_apply_round_break_recovery() -> None:
    state = distance_state(
        1,
        3,
    )
    selected = path_segment(
        1,
        3,
        finish=knockout_finish(state),
    )

    with pytest.raises(
        ValueError,
        match=(
            "finishing segment cannot apply "
            "round-break recovery"
        ),
    ):
        replace(
            selected,
            round_break_recovery_applied=True,
        )


def test_finishing_segment_cannot_change_state_after_finish() -> None:
    state = distance_state(
        1,
        3,
    )
    selected = path_segment(
        1,
        3,
        finish=knockout_finish(state),
    )

    with pytest.raises(
        ValueError,
        match=(
            "finishing segment cannot alter dynamic state "
            "after finish evaluation"
        ),
    ):
        replace(
            selected,
            dynamic_state_after_segment=altered_dynamic_state(),
        )


def test_unfinished_nonfinal_segment_requires_transition() -> None:
    selected = path_segment(
        1,
        1,
    )

    with pytest.raises(
        ValueError,
        match=(
            "unfinished non-final segment "
            "requires a transition"
        ),
    ):
        replace(
            selected,
            red_effective_transition=None,
            blue_effective_transition=None,
            transition=None,
        )


def test_unfinished_nonfinal_segment_cannot_apply_recovery() -> None:
    selected = path_segment(
        1,
        1,
    )

    with pytest.raises(
        ValueError,
        match=(
            "round-break recovery can only "
            "follow segment ten"
        ),
    ):
        replace(
            selected,
            round_break_recovery_applied=True,
        )


def test_unfinished_nonfinal_segment_preserves_post_activity_state() -> None:
    selected = path_segment(
        1,
        1,
    )

    with pytest.raises(
        ValueError,
        match=(
            "non-final segment cannot alter "
            "dynamic state after activity"
        ),
    ):
        replace(
            selected,
            dynamic_state_after_segment=altered_dynamic_state(),
        )


def test_round_ending_segment_cannot_have_effective_transition() -> None:
    selected = path_segment(
        1,
        10,
        round_break_recovery_applied=True,
    )

    with pytest.raises(
        ValueError,
        match="round-ending segment cannot have a transition",
    ):
        replace(
            selected,
            red_effective_transition=transition_parameters(),
        )


def test_round_end_without_recovery_preserves_post_activity_state() -> None:
    selected = path_segment(
        3,
        10,
    )

    with pytest.raises(
        ValueError,
        match=(
            "round-ending segment without recovery cannot "
            "alter dynamic state after activity"
        ),
    ):
        replace(
            selected,
            dynamic_state_after_segment=altered_dynamic_state(),
        )


def test_transition_previous_state_must_equal_segment_state() -> None:
    selected = path_segment(
        1,
        1,
    )

    with pytest.raises(
        ValueError,
        match=(
            "transition previous state "
            "must equal segment state"
        ),
    ):
        replace(
            selected,
            transition=stay_transition(
                distance_state(
                    1,
                    2,
                )
            ),
        )


def test_valid_unfinished_path_reaches_scheduled_distance() -> None:
    selected = FinishEnabledDynamicPath(
        scheduled_rounds=3,
        seed=2026,
        segments=unfinished_segments(3),
        finish=None,
    )

    assert len(selected.segments) == 30
    assert selected.finish is None
    assert selected.reached_scheduled_distance is True


def test_valid_finished_path_stops_on_finishing_segment() -> None:
    segments, finish = finished_segments(
        finishing_segment=3,
    )

    selected = FinishEnabledDynamicPath(
        scheduled_rounds=3,
        seed=2026,
        segments=segments,
        finish=finish,
    )

    assert len(selected.segments) == 3
    assert selected.segments[-1].finish is finish
    assert selected.finish is finish
    assert selected.reached_scheduled_distance is False


@pytest.mark.parametrize(
    "scheduled_rounds",
    [
        2,
        4,
    ],
)
def test_path_rejects_unsupported_round_count(
    scheduled_rounds: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="scheduled_rounds must be 3 or 5",
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=scheduled_rounds,
            seed=1,
            segments=unfinished_segments(3),
            finish=None,
        )


def test_path_seed_must_be_integer() -> None:
    with pytest.raises(
        TypeError,
        match="seed must be an integer",
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1.0,
            segments=unfinished_segments(3),
            finish=None,
        )


def test_path_seed_cannot_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match="seed cannot be negative",
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=-1,
            segments=unfinished_segments(3),
            finish=None,
        )


def test_path_segments_must_be_tuple() -> None:
    with pytest.raises(
        TypeError,
        match="segments must be a tuple",
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=list(
                unfinished_segments(3)
            ),
            finish=None,
        )


def test_path_must_contain_segment() -> None:
    with pytest.raises(
        ValueError,
        match="path must contain at least one segment",
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=(),
            finish=None,
        )


def test_path_segment_container_requires_contract_values() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "segments must contain "
            "FinishEvaluatedPathSegment values"
        ),
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=(
                "invalid",
            ),
            finish=None,
        )


def test_path_cannot_exceed_scheduled_segment_count() -> None:
    records = (
        unfinished_segments(3)
        + (
            path_segment(
                4,
                1,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="path contains too many segments",
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=records,
            finish=None,
        )


def test_path_must_begin_with_fresh_dynamic_state() -> None:
    records = list(
        unfinished_segments(3)
    )
    records[0] = replace(
        records[0],
        dynamic_state_before=altered_dynamic_state(),
    )

    with pytest.raises(
        ValueError,
        match="fight must begin with fresh dynamic state",
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=tuple(records),
            finish=None,
        )


def test_path_level_finish_is_required_when_segment_finished() -> None:
    segments, _ = finished_segments()

    with pytest.raises(
        ValueError,
        match="path-level finish is missing",
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=segments,
            finish=None,
        )


def test_unfinished_path_requires_all_scheduled_segments() -> None:
    records = unfinished_segments(3)[:2]

    with pytest.raises(
        ValueError,
        match=(
            "unfinished path must contain "
            "all scheduled segments"
        ),
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=records,
            finish=None,
        )


def test_path_finish_must_be_finish_result_or_none() -> None:
    segments, _ = finished_segments()

    with pytest.raises(
        TypeError,
        match="finish must be FinishResult or None",
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=segments,
            finish="invalid",
        )


def test_finish_must_exist_only_on_final_stored_segment() -> None:
    finished, finish = finished_segments(
        finishing_segment=3,
    )
    records = (
        finished
        + (
            path_segment(
                1,
                4,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "finish must occur only on "
            "the final stored segment"
        ),
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=records,
            finish=finish,
        )


def test_path_finish_must_equal_final_segment_finish() -> None:
    segments, finish = finished_segments()

    different_finish = knockout_finish(
        finish.state,
        winner=FighterSide.BLUE,
    )

    with pytest.raises(
        ValueError,
        match=(
            "path finish must equal "
            "final segment finish"
        ),
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=segments,
            finish=different_finish,
        )


@pytest.mark.parametrize(
    ("replacement", "expected_message"),
    [
        (
            path_segment(
                2,
                2,
            ),
            "path round sequence is inconsistent",
        ),
        (
            path_segment(
                1,
                3,
            ),
            "path segment sequence is inconsistent",
        ),
    ],
)
def test_path_sequence_must_match_segment_index(
    replacement: FinishEvaluatedPathSegment,
    expected_message: str,
) -> None:
    records = list(
        unfinished_segments(3)
    )
    records[1] = replacement

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=tuple(records),
            finish=None,
        )


def test_path_requires_round_break_recovery_between_rounds() -> None:
    records = list(
        unfinished_segments(3)
    )
    records[9] = replace(
        records[9],
        round_break_recovery_applied=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "round-break recovery timing "
            "is inconsistent"
        ),
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=tuple(records),
            finish=None,
        )


def test_dynamic_state_result_must_feed_following_segment() -> None:
    records = list(
        unfinished_segments(3)
    )
    records[1] = replace(
        records[1],
        dynamic_state_before=altered_dynamic_state(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "dynamic-state result must "
            "feed the next segment"
        ),
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=tuple(records),
            finish=None,
        )


def test_transition_result_must_equal_following_state() -> None:
    records = list(
        unfinished_segments(3)
    )
    records[0] = replace(
        records[0],
        transition=clinch_entry_transition(
            records[0].state
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "transition result must equal "
            "the following segment state"
        ),
    ):
        FinishEnabledDynamicPath(
            scheduled_rounds=3,
            seed=1,
            segments=tuple(records),
            finish=None,
        )


def test_finish_path_contracts_are_immutable() -> None:
    segments, finish = finished_segments()

    selected = FinishEnabledDynamicPath(
        scheduled_rounds=3,
        seed=1,
        segments=segments,
        finish=finish,
    )

    with pytest.raises(FrozenInstanceError):
        selected.finish = None
