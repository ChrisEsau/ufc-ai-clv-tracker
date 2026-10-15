"""Tests for V2 shared phase-transition contracts."""

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    SharedTransition,
    TransitionEvent,
)


def state(
    phase: FightPhase,
    *,
    owner: FighterSide | None,
    age: int,
    quality: float = 0.0,
    segment: int = 1,
) -> SharedFightState:
    """Build a shared state in segment one."""

    return SharedFightState(
        phase=phase,
        phase_owner=owner,
        phase_age_segments=age,
        position_quality=quality,
        round_number=1,
        segment_number=segment,
    )


def test_distance_stay_is_legal() -> None:
    transition = SharedTransition(
        previous_state=state(
            FightPhase.DISTANCE,
            owner=None,
            age=0,
        ),
        next_state=state(
            FightPhase.DISTANCE,
            owner=None,
            age=1,
            segment=2,
        ),
        event=TransitionEvent.STAY,
        actor=None,
    )

    assert transition.event is TransitionEvent.STAY


def test_red_can_enter_clinch() -> None:
    transition = SharedTransition(
        previous_state=state(
            FightPhase.DISTANCE,
            owner=None,
            age=2,
        ),
        next_state=state(
            FightPhase.CLINCH,
            owner=FighterSide.RED,
            age=0,
            quality=0.35,
            segment=2,
        ),
        event=TransitionEvent.CLINCH_ENTRY,
        actor=FighterSide.RED,
    )

    assert transition.next_state.phase_owner is FighterSide.RED


def test_blue_can_complete_takedown() -> None:
    transition = SharedTransition(
        previous_state=state(
            FightPhase.DISTANCE,
            owner=None,
            age=2,
        ),
        next_state=state(
            FightPhase.GROUND,
            owner=FighterSide.BLUE,
            age=0,
            quality=0.55,
            segment=2,
        ),
        event=TransitionEvent.TAKEDOWN,
        actor=FighterSide.BLUE,
    )

    assert transition.actor is FighterSide.BLUE


def test_clinch_can_break_to_distance() -> None:
    SharedTransition(
        previous_state=state(
            FightPhase.CLINCH,
            owner=FighterSide.RED,
            age=3,
            quality=0.40,
        ),
        next_state=state(
            FightPhase.DISTANCE,
            owner=None,
            age=0,
            segment=2,
        ),
        event=TransitionEvent.CLINCH_BREAK,
        actor=None,
    )


def test_clinch_ownership_can_change() -> None:
    SharedTransition(
        previous_state=state(
            FightPhase.CLINCH,
            owner=FighterSide.RED,
            age=2,
            quality=0.45,
        ),
        next_state=state(
            FightPhase.CLINCH,
            owner=FighterSide.BLUE,
            age=0,
            quality=0.30,
            segment=2,
        ),
        event=TransitionEvent.OWNERSHIP_CHANGE,
        actor=FighterSide.BLUE,
    )


def test_ground_defender_can_escape() -> None:
    SharedTransition(
        previous_state=state(
            FightPhase.GROUND,
            owner=FighterSide.RED,
            age=4,
            quality=0.70,
        ),
        next_state=state(
            FightPhase.DISTANCE,
            owner=None,
            age=0,
            segment=2,
        ),
        event=TransitionEvent.GROUND_ESCAPE,
        actor=FighterSide.BLUE,
    )


def test_ground_defender_can_reverse() -> None:
    SharedTransition(
        previous_state=state(
            FightPhase.GROUND,
            owner=FighterSide.RED,
            age=3,
            quality=0.65,
        ),
        next_state=state(
            FightPhase.GROUND,
            owner=FighterSide.BLUE,
            age=0,
            quality=0.35,
            segment=2,
        ),
        event=TransitionEvent.REVERSAL,
        actor=FighterSide.BLUE,
    )


def test_illegal_event_phase_pair_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="illegal phase pair",
    ):
        SharedTransition(
            previous_state=state(
                FightPhase.DISTANCE,
                owner=None,
                age=1,
            ),
            next_state=state(
                FightPhase.GROUND,
                owner=FighterSide.RED,
                age=0,
                quality=0.50,
            segment=2,
            ),
            event=TransitionEvent.CLINCH_ENTRY,
            actor=FighterSide.RED,
        )


def test_takedown_actor_must_own_ground() -> None:
    with pytest.raises(
        ValueError,
        match="actor must own",
    ):
        SharedTransition(
            previous_state=state(
                FightPhase.DISTANCE,
                owner=None,
                age=1,
            ),
            next_state=state(
                FightPhase.GROUND,
                owner=FighterSide.BLUE,
                age=0,
                quality=0.50,
            segment=2,
            ),
            event=TransitionEvent.TAKEDOWN,
            actor=FighterSide.RED,
        )


def test_stay_cannot_change_owner() -> None:
    with pytest.raises(
        ValueError,
        match="cannot change phase owner",
    ):
        SharedTransition(
            previous_state=state(
                FightPhase.GROUND,
                owner=FighterSide.RED,
                age=2,
                quality=0.60,
            ),
            next_state=state(
                FightPhase.GROUND,
                owner=FighterSide.BLUE,
                age=3,
                quality=0.40,
            segment=2,
            ),
            event=TransitionEvent.STAY,
            actor=None,
        )


def test_transition_must_advance_to_next_segment() -> None:
    """A transition result belongs to the following segment."""

    with pytest.raises(
        ValueError,
        match="advance to the next segment",
    ):
        SharedTransition(
            previous_state=state(
                FightPhase.DISTANCE,
                owner=None,
                age=1,
                segment=4,
            ),
            next_state=state(
                FightPhase.DISTANCE,
                owner=None,
                age=2,
                segment=4,
            ),
            event=TransitionEvent.STAY,
            actor=None,
        )


def test_end_of_round_uses_round_reset() -> None:
    """Segment ten cannot transition into another same-round state."""

    with pytest.raises(
        ValueError,
        match="end-of-round state",
    ):
        SharedTransition(
            previous_state=state(
                FightPhase.DISTANCE,
                owner=None,
                age=3,
                segment=10,
            ),
            next_state=state(
                FightPhase.DISTANCE,
                owner=None,
                age=4,
                segment=10,
            ),
            event=TransitionEvent.STAY,
            actor=None,
        )
