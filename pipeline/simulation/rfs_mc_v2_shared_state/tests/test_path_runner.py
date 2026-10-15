"""Tests for the V2 shared-state path runner."""

from dataclasses import replace

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    run_shared_state_path,
    select_transition_distribution,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


def parameters(
    **overrides: float,
) -> FighterTransitionParameters:
    """Build a neutral transition profile."""

    neutral = FighterTransitionParameters(
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

    return replace(
        neutral,
        **overrides,
    )


def shared_state(
    phase: FightPhase,
    *,
    owner: FighterSide | None,
    quality: float,
) -> SharedFightState:
    """Build a valid shared state for distribution tests."""

    return SharedFightState(
        phase=phase,
        phase_owner=owner,
        phase_age_segments=1,
        position_quality=quality,
        round_number=1,
        segment_number=2,
    )


def test_distance_state_selects_distance_distribution() -> None:
    distribution = select_transition_distribution(
        shared_state(
            FightPhase.DISTANCE,
            owner=None,
            quality=0.0,
        ),
        parameters(),
        parameters(),
    )

    assert distribution.source_phase is FightPhase.DISTANCE

    assert {
        option.event
        for option in distribution.options
    } == {
        TransitionEvent.STAY,
        TransitionEvent.CLINCH_ENTRY,
        TransitionEvent.TAKEDOWN,
    }


def test_clinch_state_selects_owner_aware_distribution() -> None:
    distribution = select_transition_distribution(
        shared_state(
            FightPhase.CLINCH,
            owner=FighterSide.BLUE,
            quality=0.40,
        ),
        parameters(),
        parameters(),
    )

    assert distribution.source_phase is FightPhase.CLINCH

    assert distribution.probability(
        TransitionEvent.OWNERSHIP_CHANGE,
        FighterSide.RED,
    ) > 0.0

    assert distribution.probability(
        TransitionEvent.TAKEDOWN,
        FighterSide.BLUE,
    ) > 0.0


def test_ground_state_assigns_defensive_actions_to_non_owner() -> None:
    distribution = select_transition_distribution(
        shared_state(
            FightPhase.GROUND,
            owner=FighterSide.BLUE,
            quality=0.65,
        ),
        parameters(),
        parameters(),
    )

    assert distribution.source_phase is FightPhase.GROUND

    for event in (
        TransitionEvent.GROUND_ESCAPE,
        TransitionEvent.SCRAMBLE_TO_CLINCH,
        TransitionEvent.REVERSAL,
    ):
        assert distribution.probability(
            event,
            FighterSide.RED,
        ) > 0.0


@pytest.mark.parametrize(
    ("scheduled_rounds", "expected_segments"),
    [
        (3, 30),
        (5, 50),
    ],
)
def test_path_contains_every_scheduled_segment(
    scheduled_rounds: int,
    expected_segments: int,
) -> None:
    path = run_shared_state_path(
        parameters(),
        parameters(),
        scheduled_rounds=scheduled_rounds,
        seed=42,
    )

    assert len(path.segments) == expected_segments


def test_same_seed_produces_identical_path() -> None:
    first = run_shared_state_path(
        parameters(),
        parameters(),
        scheduled_rounds=3,
        seed=2026,
    )
    second = run_shared_state_path(
        parameters(),
        parameters(),
        scheduled_rounds=3,
        seed=2026,
    )

    assert first == second


def test_every_round_begins_at_distance() -> None:
    path = run_shared_state_path(
        parameters(),
        parameters(),
        scheduled_rounds=5,
        seed=17,
    )

    opening_segments = [
        record
        for record in path.segments
        if record.state.segment_number == 1
    ]

    assert len(opening_segments) == 5

    for record in opening_segments:
        assert record.state.phase is FightPhase.DISTANCE
        assert record.state.phase_owner is None
        assert record.state.position_quality == 0.0
        assert record.state.phase_age_segments == 0


def test_transition_result_feeds_following_segment() -> None:
    path = run_shared_state_path(
        parameters(),
        parameters(),
        scheduled_rounds=3,
        seed=91,
    )

    for index, record in enumerate(path.segments):
        if record.state.segment_number == SEGMENTS_PER_ROUND:
            continue

        assert record.transition is not None
        assert (
            record.transition.previous_state
            == record.state
        )
        assert (
            record.transition.next_state
            == path.segments[index + 1].state
        )


def test_round_end_has_no_transition_and_next_round_resets() -> None:
    path = run_shared_state_path(
        parameters(),
        parameters(),
        scheduled_rounds=3,
        seed=100,
    )

    for round_number in (1, 2, 3):
        round_end_index = (
            round_number * SEGMENTS_PER_ROUND
        ) - 1

        round_end = path.segments[round_end_index]

        assert round_end.state.segment_number == 10
        assert round_end.transition is None

        if round_number < 3:
            next_round = path.segments[
                round_end_index + 1
            ].state

            assert next_round.round_number == round_number + 1
            assert next_round.segment_number == 1
            assert next_round.phase is FightPhase.DISTANCE
            assert next_round.phase_owner is None


def test_every_owned_state_has_exactly_one_owner() -> None:
    path = run_shared_state_path(
        parameters(),
        parameters(),
        scheduled_rounds=5,
        seed=5150,
    )

    for record in path.segments:
        state = record.state

        if state.phase is FightPhase.DISTANCE:
            assert state.phase_owner is None
            assert state.position_quality == 0.0
        else:
            assert state.phase_owner in {
                FighterSide.RED,
                FighterSide.BLUE,
            }


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
        run_shared_state_path(
            parameters(),
            parameters(),
            scheduled_rounds=scheduled_rounds,
            seed=1,
        )


def test_runner_rejects_negative_seed() -> None:
    with pytest.raises(
        ValueError,
        match="seed",
    ):
        run_shared_state_path(
            parameters(),
            parameters(),
            scheduled_rounds=3,
            seed=-1,
        )
