"""Tests for V2 shared fight-state contracts."""

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)


def test_fight_opens_at_distance() -> None:
    """Every simulated fight must begin at distance."""

    state = SharedFightState.opening_state()

    assert state.phase is FightPhase.DISTANCE
    assert state.phase_owner is None
    assert state.position_quality == 0.0
    assert state.phase_age_segments == 0
    assert state.round_number == 1
    assert state.segment_number == 1


def test_distance_cannot_have_owner() -> None:
    """No fighter owns the distance phase."""

    with pytest.raises(
        ValueError,
        match="distance phase cannot have",
    ):
        SharedFightState(
            phase=FightPhase.DISTANCE,
            phase_owner=FighterSide.RED,
            phase_age_segments=1,
            position_quality=0.0,
            round_number=1,
            segment_number=1,
        )


def test_distance_cannot_have_position_quality() -> None:
    """Position quality is meaningful only in owned phases."""

    with pytest.raises(
        ValueError,
        match="distance phase must have zero",
    ):
        SharedFightState(
            phase=FightPhase.DISTANCE,
            phase_owner=None,
            phase_age_segments=1,
            position_quality=0.40,
            round_number=1,
            segment_number=1,
        )


@pytest.mark.parametrize(
    "phase",
    [
        FightPhase.CLINCH,
        FightPhase.GROUND,
    ],
)
def test_owned_phase_requires_owner(
    phase: FightPhase,
) -> None:
    """Clinch and ground phases require a controlling side."""

    with pytest.raises(
        ValueError,
        match="require a phase owner",
    ):
        SharedFightState(
            phase=phase,
            phase_owner=None,
            phase_age_segments=1,
            position_quality=0.50,
            round_number=1,
            segment_number=1,
        )


@pytest.mark.parametrize(
    "phase",
    [
        FightPhase.CLINCH,
        FightPhase.GROUND,
    ],
)
def test_owned_phase_accepts_valid_owner(
    phase: FightPhase,
) -> None:
    """A valid owned phase carries one controlling fighter."""

    state = SharedFightState(
        phase=phase,
        phase_owner=FighterSide.BLUE,
        phase_age_segments=1,
        position_quality=0.60,
        round_number=1,
        segment_number=2,
    )

    assert state.phase_owner is FighterSide.BLUE
    assert state.phase_owner.opponent is FighterSide.RED


def test_new_round_resets_to_distance() -> None:
    """Every new round must restart from distance."""

    ground_state = SharedFightState(
        phase=FightPhase.GROUND,
        phase_owner=FighterSide.RED,
        phase_age_segments=4,
        position_quality=0.80,
        round_number=1,
        segment_number=10,
    )

    round_two = ground_state.reset_for_round(
        round_number=2,
    )

    assert round_two.phase is FightPhase.DISTANCE
    assert round_two.phase_owner is None
    assert round_two.phase_age_segments == 0
    assert round_two.position_quality == 0.0
    assert round_two.round_number == 2
    assert round_two.segment_number == 1
