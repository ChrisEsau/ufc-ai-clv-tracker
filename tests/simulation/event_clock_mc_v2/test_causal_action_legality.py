"""Stage 2 tests for typed action attempts and authoritative legality."""

from dataclasses import fields

import pytest

from pipeline.simulation.event_clock_mc_v2.causal.events import ActionEvent, ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.legality import legal_actions, validate_action_event
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side


STANDING_MENU = (
    ActionFamily.STAND_ATTACK,
    ActionFamily.STAND_COUNTER,
    ActionFamily.PRESSURE,
    ActionFamily.RESET_RANGE,
    ActionFamily.CLINCH_ENTRY,
    ActionFamily.TAKEDOWN_ENTRY,
)
CLINCH_MENU = (
    ActionFamily.CLINCH_STRIKE,
    ActionFamily.CLINCH_CONTROL,
    ActionFamily.CLINCH_TAKEDOWN,
    ActionFamily.BREAK_CLINCH,
)
GROUND_TOP_MENU = (
    ActionFamily.GROUND_STRIKE,
    ActionFamily.ADVANCE_POSITION,
    ActionFamily.SUBMISSION_ATTACK,
    ActionFamily.CONTROL,
    ActionFamily.DISENGAGE,
)
GROUND_BOTTOM_MENU = (
    ActionFamily.ESCAPE_STAND,
    ActionFamily.IMPROVE_POSITION,
    ActionFamily.REVERSAL,
    ActionFamily.SUBMISSION_ATTACK,
    ActionFamily.BOTTOM_STRIKE,
)


@pytest.mark.parametrize("actor", list(Side))
def test_standing_menu_is_exact_for_both_fighters(actor: Side) -> None:
    assert legal_actions(FightState(), actor) == STANDING_MENU


@pytest.mark.parametrize("actor", list(Side))
def test_clinch_menu_is_exact_for_controller_and_non_controller(actor: Side) -> None:
    state = _state(Phase.CLINCH, clinch_controller=Side.RED)

    assert legal_actions(state, actor) == CLINCH_MENU


@pytest.mark.parametrize(
    ("controller", "top", "bottom"),
    [(Side.RED, Side.RED, Side.BLUE), (Side.BLUE, Side.BLUE, Side.RED)],
)
def test_ground_roles_derive_from_authoritative_controller(
    controller: Side, top: Side, bottom: Side
) -> None:
    state = _state(Phase.GROUND, ground_controller=controller)

    assert legal_actions(state, top) == GROUND_TOP_MENU
    assert legal_actions(state, bottom) == GROUND_BOTTOM_MENU


@pytest.mark.parametrize(
    ("phase", "controller", "action"),
    [
        (Phase.CLINCH, Side.RED, ActionFamily.STAND_ATTACK),
        (Phase.GROUND, Side.RED, ActionFamily.STAND_ATTACK),
        (Phase.STANDING, None, ActionFamily.CLINCH_STRIKE),
        (Phase.GROUND, Side.RED, ActionFamily.CLINCH_STRIKE),
    ],
)
def test_cross_phase_actions_are_rejected(
    phase: Phase, controller: Side | None, action: ActionFamily
) -> None:
    state = _state(
        phase,
        clinch_controller=controller if phase is Phase.CLINCH else None,
        ground_controller=controller if phase is Phase.GROUND else None,
    )
    event = ActionEvent(state.fight_time_seconds, Side.RED, action, state.phase)

    with pytest.raises(ValueError, match="is not legal"):
        validate_action_event(event, state)


def test_ground_top_action_is_rejected_for_bottom_fighter() -> None:
    state = _state(Phase.GROUND, ground_controller=Side.RED)
    event = ActionEvent(15.0, Side.BLUE, ActionFamily.GROUND_STRIKE, Phase.GROUND)

    with pytest.raises(ValueError, match="is not legal"):
        validate_action_event(event, state)


def test_ground_bottom_action_is_rejected_for_top_fighter() -> None:
    state = _state(Phase.GROUND, ground_controller=Side.BLUE)
    event = ActionEvent(15.0, Side.BLUE, ActionFamily.REVERSAL, Phase.GROUND)

    with pytest.raises(ValueError, match="is not legal"):
        validate_action_event(event, state)


def test_source_phase_mismatch_is_rejected() -> None:
    event = ActionEvent(0.0, Side.RED, ActionFamily.STAND_ATTACK, Phase.CLINCH)

    with pytest.raises(ValueError, match="does not match"):
        validate_action_event(event, FightState())


def test_event_timestamp_before_current_fight_time_is_rejected() -> None:
    state = _state(Phase.STANDING, fight_time_seconds=20.0, phase_started_at=10.0)
    event = ActionEvent(19.99, Side.RED, ActionFamily.STAND_ATTACK, Phase.STANDING)

    with pytest.raises(ValueError, match="cannot precede"):
        validate_action_event(event, state)


@pytest.mark.parametrize("timestamp", [-1.0, float("inf"), float("-inf"), float("nan"), "1.0", True])
def test_malformed_event_timestamps_are_rejected(timestamp) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        ActionEvent(timestamp, Side.RED, ActionFamily.STAND_ATTACK, Phase.STANDING)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"actor": "red"}, "actor must be a Side"),
        ({"action_family": "stand_attack"}, "action_family must be an ActionFamily"),
        ({"source_phase": "standing"}, "source_phase must be a Phase"),
    ],
)
def test_event_rejects_free_form_enum_values(overrides, message: str) -> None:
    values = {
        "timestamp_seconds": 0.0,
        "actor": Side.RED,
        "action_family": ActionFamily.STAND_ATTACK,
        "source_phase": Phase.STANDING,
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        ActionEvent(**values)


def test_action_event_contains_attempt_fields_only() -> None:
    assert tuple(field.name for field in fields(ActionEvent)) == (
        "timestamp_seconds",
        "actor",
        "action_family",
        "source_phase",
    )


def test_valid_event_validation_is_deterministic_and_side_effect_free() -> None:
    state = _state(Phase.GROUND, ground_controller=Side.RED)
    event = ActionEvent(15.0, Side.BLUE, ActionFamily.ESCAPE_STAND, Phase.GROUND)
    state_before = state

    assert validate_action_event(event, state) is None
    assert validate_action_event(event, state) is None
    assert state == state_before


def test_legal_actions_rejects_free_form_actor() -> None:
    with pytest.raises(ValueError, match="actor must be a Side"):
        legal_actions(FightState(), "red")


def _state(
    phase: Phase,
    *,
    fight_time_seconds: float = 10.0,
    phase_started_at: float = 10.0,
    clinch_controller: Side | None = None,
    ground_controller: Side | None = None,
) -> FightState:
    return FightState(
        fight_time_seconds=fight_time_seconds,
        phase=phase,
        phase_started_at=phase_started_at,
        clinch_controller=clinch_controller,
        ground_controller=ground_controller,
    )
