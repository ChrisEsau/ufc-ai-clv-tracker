"""Tests for RFS Monte Carlo V2 finish-result contracts."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
    FinishResult,
)


def shared_state(
    *,
    phase: FightPhase = FightPhase.DISTANCE,
    owner: FighterSide | None = None,
    round_number: int = 1,
    segment_number: int = 1,
) -> SharedFightState:
    """Build one valid authoritative shared fight state."""

    if phase is FightPhase.DISTANCE:
        selected_owner = None
        phase_age_segments = 0
        position_quality = 0.0
    else:
        selected_owner = owner or FighterSide.RED
        phase_age_segments = 1
        position_quality = 0.50

    return SharedFightState(
        phase=phase,
        phase_owner=selected_owner,
        phase_age_segments=phase_age_segments,
        position_quality=position_quality,
        round_number=round_number,
        segment_number=segment_number,
    )


def finish_result(
    *,
    state: SharedFightState | None = None,
    winner: FighterSide = FighterSide.RED,
    method: FinishMethod = FinishMethod.KO_TKO,
    elapsed_seconds_in_segment: int = 15,
) -> FinishResult:
    """Build one valid finish result."""

    return FinishResult(
        state=state or shared_state(),
        winner=winner,
        method=method,
        elapsed_seconds_in_segment=(
            elapsed_seconds_in_segment
        ),
    )


def test_finish_method_values_are_stable() -> None:
    assert FinishMethod.KO_TKO.value == "ko_tko"
    assert FinishMethod.SUBMISSION.value == "submission"


@pytest.mark.parametrize(
    ("state", "winner"),
    [
        (
            shared_state(
                phase=FightPhase.DISTANCE,
            ),
            FighterSide.RED,
        ),
        (
            shared_state(
                phase=FightPhase.CLINCH,
                owner=FighterSide.BLUE,
            ),
            FighterSide.RED,
        ),
        (
            shared_state(
                phase=FightPhase.GROUND,
                owner=FighterSide.RED,
            ),
            FighterSide.BLUE,
        ),
        (
            shared_state(
                phase=FightPhase.GROUND,
                owner=FighterSide.BLUE,
            ),
            FighterSide.BLUE,
        ),
    ],
)
def test_ko_tko_finish_is_supported_across_phases(
    state: SharedFightState,
    winner: FighterSide,
) -> None:
    result = finish_result(
        state=state,
        winner=winner,
        method=FinishMethod.KO_TKO,
    )

    assert result.state == state
    assert result.winner is winner
    assert result.method is FinishMethod.KO_TKO


@pytest.mark.parametrize(
    "winner",
    [
        FighterSide.RED,
        FighterSide.BLUE,
    ],
)
def test_valid_submission_requires_ground_owner(
    winner: FighterSide,
) -> None:
    state = shared_state(
        phase=FightPhase.GROUND,
        owner=winner,
    )

    result = finish_result(
        state=state,
        winner=winner,
        method=FinishMethod.SUBMISSION,
    )

    assert result.state.phase is FightPhase.GROUND
    assert result.state.phase_owner is winner
    assert result.winner is winner


@pytest.mark.parametrize(
    "winner",
    [
        FighterSide.RED,
        FighterSide.BLUE,
    ],
)
def test_finish_result_exposes_result_properties(
    winner: FighterSide,
) -> None:
    result = finish_result(
        state=shared_state(
            round_number=3,
            segment_number=7,
        ),
        winner=winner,
        elapsed_seconds_in_segment=12,
    )

    assert result.winner is winner
    assert result.loser is winner.opponent
    assert result.round_number == 3
    assert result.segment_number == 7
    assert result.elapsed_seconds_in_segment == 12


@pytest.mark.parametrize(
    (
        "segment_number",
        "elapsed_seconds_in_segment",
        "expected_elapsed_seconds_in_round",
    ),
    [
        (1, 1, 1),
        (1, 30, 30),
        (2, 1, 31),
        (5, 15, 135),
        (10, 30, 300),
    ],
)
def test_elapsed_seconds_in_round_are_calculated_correctly(
    segment_number: int,
    elapsed_seconds_in_segment: int,
    expected_elapsed_seconds_in_round: int,
) -> None:
    result = finish_result(
        state=shared_state(
            segment_number=segment_number,
        ),
        elapsed_seconds_in_segment=(
            elapsed_seconds_in_segment
        ),
    )

    assert (
        result.elapsed_seconds_in_round
        == expected_elapsed_seconds_in_round
    )


def test_finish_result_is_immutable() -> None:
    result = finish_result()

    with pytest.raises(FrozenInstanceError):
        result.winner = FighterSide.BLUE


def test_finish_state_must_be_shared_fight_state() -> None:
    with pytest.raises(
        TypeError,
        match="state must be SharedFightState",
    ):
        FinishResult(
            state="invalid",
            winner=FighterSide.RED,
            method=FinishMethod.KO_TKO,
            elapsed_seconds_in_segment=15,
        )


def test_finish_winner_must_be_fighter_side() -> None:
    with pytest.raises(
        TypeError,
        match="winner must be FighterSide",
    ):
        FinishResult(
            state=shared_state(),
            winner="red",
            method=FinishMethod.KO_TKO,
            elapsed_seconds_in_segment=15,
        )


def test_finish_method_must_be_finish_method() -> None:
    with pytest.raises(
        TypeError,
        match="method must be FinishMethod",
    ):
        FinishResult(
            state=shared_state(),
            winner=FighterSide.RED,
            method="ko_tko",
            elapsed_seconds_in_segment=15,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        "15",
    ],
)
def test_elapsed_seconds_in_segment_must_be_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            "elapsed_seconds_in_segment must be an integer"
        ),
    ):
        FinishResult(
            state=shared_state(),
            winner=FighterSide.RED,
            method=FinishMethod.KO_TKO,
            elapsed_seconds_in_segment=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1,
        0,
        31,
        100,
    ],
)
def test_elapsed_seconds_in_segment_must_be_in_bounds(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "elapsed_seconds_in_segment must be between "
            "1 and 30"
        ),
    ):
        finish_result(
            elapsed_seconds_in_segment=invalid_value,
        )


@pytest.mark.parametrize(
    "phase",
    [
        FightPhase.DISTANCE,
        FightPhase.CLINCH,
    ],
)
def test_submission_finish_requires_ground_phase(
    phase: FightPhase,
) -> None:
    with pytest.raises(
        ValueError,
        match="submission finish requires a ground state",
    ):
        finish_result(
            state=shared_state(
                phase=phase,
                owner=FighterSide.RED,
            ),
            winner=FighterSide.RED,
            method=FinishMethod.SUBMISSION,
        )


@pytest.mark.parametrize(
    ("owner", "winner"),
    [
        (
            FighterSide.RED,
            FighterSide.BLUE,
        ),
        (
            FighterSide.BLUE,
            FighterSide.RED,
        ),
    ],
)
def test_submission_winner_must_own_ground_phase(
    owner: FighterSide,
    winner: FighterSide,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "submission winner must own the ground phase"
        ),
    ):
        finish_result(
            state=shared_state(
                phase=FightPhase.GROUND,
                owner=owner,
            ),
            winner=winner,
            method=FinishMethod.SUBMISSION,
        )
