"""Deterministic Stage 1 tests for authoritative V2 phase exposure."""

import pytest

from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side
from pipeline.simulation.event_clock_mc_v2.causal.timeline import ActivePhase, PhaseSegment, PhaseTimeline
from pipeline.simulation.event_clock_mc_v2.causal.transitions import (
    clinch_takedown,
    direct_takedown,
    enter_clinch,
    escape_ground,
    reverse_ground,
    separate_clinch,
    start_next_round,
)


def test_initial_state_starts_standing_without_controllers() -> None:
    state = FightState()

    assert state.phase is Phase.STANDING
    assert state.clinch_controller is None
    assert state.ground_controller is None


@pytest.mark.parametrize(
    ("clinch_controller", "ground_controller"),
    [(Side.RED, None), (None, Side.BLUE), (Side.RED, Side.BLUE)],
)
def test_standing_rejects_controllers(clinch_controller, ground_controller) -> None:
    with pytest.raises(ValueError, match="standing cannot carry"):
        FightState(clinch_controller=clinch_controller, ground_controller=ground_controller)


def test_clinch_and_ground_controller_invariants() -> None:
    with pytest.raises(ValueError, match="clinch requires"):
        FightState(phase=Phase.CLINCH)
    with pytest.raises(ValueError, match="clinch cannot carry a ground"):
        FightState(
            phase=Phase.CLINCH,
            clinch_controller=Side.BLUE,
            ground_controller=Side.RED,
        )
    with pytest.raises(ValueError, match="ground requires"):
        FightState(phase=Phase.GROUND)
    with pytest.raises(ValueError, match="ground cannot carry a clinch"):
        FightState(
            phase=Phase.GROUND,
            clinch_controller=Side.BLUE,
            ground_controller=Side.RED,
        )


@pytest.mark.parametrize("record_type", [PhaseSegment, ActivePhase])
def test_standing_timeline_records_reject_controller(record_type) -> None:
    if record_type is PhaseSegment:
        args = (0.0, 5.0, Phase.STANDING, Side.RED, "start", "end")
    else:
        args = (0.0, Phase.STANDING, Side.RED, "start")

    with pytest.raises(ValueError, match="standing phase cannot carry"):
        record_type(*args)


@pytest.mark.parametrize("record_type", [PhaseSegment, ActivePhase])
def test_ground_timeline_records_require_controller(record_type) -> None:
    if record_type is PhaseSegment:
        args = (0.0, 5.0, Phase.GROUND, None, "start", "end")
    else:
        args = (0.0, Phase.GROUND, None, "start")

    with pytest.raises(ValueError, match="ground phase requires"):
        record_type(*args)


@pytest.mark.parametrize("record_type", [PhaseSegment, ActivePhase])
def test_clinch_timeline_records_require_explicit_controller(record_type) -> None:
    if record_type is PhaseSegment:
        args = (0.0, 5.0, Phase.CLINCH, None, "start", "end")
    else:
        args = (0.0, Phase.CLINCH, None, "start")

    with pytest.raises(ValueError, match="clinch phase requires"):
        record_type(*args)


@pytest.mark.parametrize("record_type", [PhaseSegment, ActivePhase])
def test_timeline_records_reject_free_form_controller_values(record_type) -> None:
    if record_type is PhaseSegment:
        args = (0.0, 5.0, Phase.CLINCH, "red", "start", "end")
    else:
        args = (0.0, Phase.CLINCH, "red", "start")

    with pytest.raises(ValueError, match="must be a Side"):
        record_type(*args)


def test_clinch_transition_sets_only_clinch_controller_and_separation_clears_it() -> None:
    state, timeline = _initial()
    state = enter_clinch(state, timeline, 20.0, Side.RED)

    assert state.phase is Phase.CLINCH
    assert state.clinch_controller is Side.RED
    assert state.ground_controller is None

    state = separate_clinch(state, timeline, 32.0)
    assert state.phase is Phase.STANDING
    assert state.clinch_controller is state.ground_controller is None


def test_direct_takedown_sets_only_ground_controller_and_escape_clears_it() -> None:
    state, timeline = _initial()
    state = direct_takedown(state, timeline, 12.0, Side.BLUE)

    assert state.phase is Phase.GROUND
    assert state.ground_controller is Side.BLUE
    assert state.clinch_controller is None

    state = escape_ground(state, timeline, 19.0)
    assert state.phase is Phase.STANDING
    assert state.clinch_controller is state.ground_controller is None


def test_clinch_takedown_changes_controller_representation() -> None:
    state, timeline = _initial()
    state = enter_clinch(state, timeline, 5.0, Side.BLUE)
    state = clinch_takedown(state, timeline, 9.0, Side.RED)

    assert state.phase is Phase.GROUND
    assert state.clinch_controller is None
    assert state.ground_controller is Side.RED


def test_ground_reversal_opens_new_ground_segment_at_exact_timestamp() -> None:
    state, timeline = _initial()
    state = direct_takedown(state, timeline, 10.0, Side.RED)
    state = reverse_ground(state, timeline, 25.0, Side.BLUE)

    assert state.phase is Phase.GROUND
    assert state.ground_controller is Side.BLUE
    assert timeline.segments[-1].end_time == 25.0
    assert timeline.active.start_time == 25.0
    assert timeline.active.phase is Phase.GROUND
    assert timeline.active.controller is Side.BLUE
    with pytest.raises(ValueError, match="must change"):
        reverse_ground(state, timeline, 26.0, Side.BLUE)


def test_segment_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        PhaseSegment(2.0, 1.0, Phase.STANDING, None, "start", "end")


@pytest.mark.parametrize("next_start", [4.0, 6.0])
def test_timeline_validation_rejects_overlap_and_gap(next_start: float) -> None:
    timeline = PhaseTimeline(ActivePhase(0.0, Phase.STANDING, None, "round_start"))
    timeline._segments = [  # Deliberate corruption proves validation fails loudly.
        PhaseSegment(0.0, 5.0, Phase.STANDING, None, "start", "transition"),
        PhaseSegment(next_start, 8.0, Phase.CLINCH, Side.RED, "entry", "exit"),
    ]
    timeline._active = None

    with pytest.raises(ValueError, match="chronological and contiguous"):
        timeline.validate()


def test_time_cannot_move_backward() -> None:
    state, timeline = _initial()
    state = enter_clinch(state, timeline, 20.0, Side.RED)

    with pytest.raises(ValueError, match="backward"):
        clinch_takedown(state, timeline, 19.0, Side.RED)


def test_same_timestamp_transition_has_no_artificial_gap() -> None:
    state, timeline = _initial()
    state = enter_clinch(state, timeline, 20.0, Side.RED)
    state = clinch_takedown(state, timeline, 20.0, Side.RED)

    assert timeline.segments[-1].start_time == timeline.segments[-1].end_time == 20.0
    assert timeline.segments[-1].end_time == timeline.active.start_time
    timeline.validate()


def test_round_boundary_closes_phase_and_resets_to_standing() -> None:
    state, timeline = _initial()
    state = direct_takedown(state, timeline, 120.0, Side.RED)
    state = start_next_round(state, timeline, 300.0)

    assert timeline.segments[-1].end_time == 300.0
    assert timeline.active.start_time == 300.0
    assert timeline.active.phase is Phase.STANDING
    assert state.round_number == 2
    assert state.phase is Phase.STANDING
    assert state.clinch_controller is state.ground_controller is None


def test_scripted_path_exactly_conserves_phase_exposure() -> None:
    state, timeline = _scripted_path()

    assert state.phase is Phase.STANDING
    assert state.fight_time_seconds == 71.0
    assert [
        (segment.start_time, segment.end_time, segment.phase, segment.controller)
        for segment in timeline.segments_through(90.0)
    ] == [
        (0.0, 20.0, Phase.STANDING, None),
        (20.0, 32.0, Phase.CLINCH, Side.RED),
        (32.0, 50.0, Phase.GROUND, Side.RED),
        (50.0, 71.0, Phase.GROUND, Side.BLUE),
        (71.0, 90.0, Phase.STANDING, None),
    ]
    assert timeline.exposure_seconds_through(90.0) == {
        Phase.STANDING: 39.0,
        Phase.CLINCH: 12.0,
        Phase.GROUND: 39.0,
    }
    assert sum(timeline.exposure_seconds_through(90.0).values()) == 90.0
    timeline.validate()


def test_reporting_horizon_is_non_destructive_and_allows_later_transition() -> None:
    state, timeline = _scripted_path()
    completed_before = timeline.segments
    active_before = timeline.active

    snapshot = timeline.segments_through(90.0)

    assert snapshot[-1] == PhaseSegment(
        71.0,
        90.0,
        Phase.STANDING,
        None,
        "ground_escape",
        "reporting_horizon",
    )
    assert timeline.segments == completed_before
    assert timeline.active is active_before
    assert timeline.active.start_time == state.phase_started_at == 71.0
    assert timeline.active.phase is state.phase is Phase.STANDING

    state = enter_clinch(state, timeline, 100.0, Side.RED)
    assert state.phase is Phase.CLINCH
    assert timeline.segments[-1].end_time == 100.0
    assert timeline.active == ActivePhase(100.0, Phase.CLINCH, Side.RED, "clinch_entry")


def test_repeated_deterministic_construction_is_identical() -> None:
    first_state, first_timeline = _scripted_path()
    second_state, second_timeline = _scripted_path()

    assert first_state == second_state
    assert first_timeline.segments == second_timeline.segments


def _initial() -> tuple[FightState, PhaseTimeline]:
    state = FightState()
    return state, PhaseTimeline.from_state(state)


def _scripted_path() -> tuple[FightState, PhaseTimeline]:
    state, timeline = _initial()
    state = enter_clinch(state, timeline, 20.0, Side.RED)
    state = clinch_takedown(state, timeline, 32.0, Side.RED)
    state = reverse_ground(state, timeline, 50.0, Side.BLUE)
    state = escape_ground(state, timeline, 71.0)
    return state, timeline
