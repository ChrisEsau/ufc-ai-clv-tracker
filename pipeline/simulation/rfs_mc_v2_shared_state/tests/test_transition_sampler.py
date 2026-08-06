"""Tests for seeded transition sampling and state application."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_engine import (
    TransitionDistribution,
    TransitionProbability,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_sampler import (
    TransitionStateCalibration,
    apply_transition_option,
    sample_and_apply_transition,
    sample_transition_option,
)


def state(
    phase: FightPhase,
    *,
    owner: FighterSide | None,
    age: int,
    quality: float,
    segment: int = 1,
) -> SharedFightState:
    """Create a shared state for sampler tests."""

    return SharedFightState(
        phase=phase,
        phase_owner=owner,
        phase_age_segments=age,
        position_quality=quality,
        round_number=1,
        segment_number=segment,
    )


def option(
    event: TransitionEvent,
    actor: FighterSide | None,
) -> TransitionProbability:
    """Create a certain transition option."""

    return TransitionProbability(
        event=event,
        actor=actor,
        probability=1.0,
    )


def test_sampling_is_deterministic_for_same_seed() -> None:
    distribution = TransitionDistribution(
        source_phase=FightPhase.DISTANCE,
        options=(
            TransitionProbability(
                event=TransitionEvent.STAY,
                actor=None,
                probability=0.60,
            ),
            TransitionProbability(
                event=TransitionEvent.CLINCH_ENTRY,
                actor=FighterSide.RED,
                probability=0.25,
            ),
            TransitionProbability(
                event=TransitionEvent.TAKEDOWN,
                actor=FighterSide.BLUE,
                probability=0.15,
            ),
        ),
    )

    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first_sequence = [
        sample_transition_option(
            distribution,
            first_rng,
        )
        for _ in range(100)
    ]
    second_sequence = [
        sample_transition_option(
            distribution,
            second_rng,
        )
        for _ in range(100)
    ]

    assert first_sequence == second_sequence


def test_sampling_tracks_probability_distribution() -> None:
    distribution = TransitionDistribution(
        source_phase=FightPhase.DISTANCE,
        options=(
            TransitionProbability(
                event=TransitionEvent.STAY,
                actor=None,
                probability=0.70,
            ),
            TransitionProbability(
                event=TransitionEvent.CLINCH_ENTRY,
                actor=FighterSide.RED,
                probability=0.20,
            ),
            TransitionProbability(
                event=TransitionEvent.TAKEDOWN,
                actor=FighterSide.BLUE,
                probability=0.10,
            ),
        ),
    )

    rng = np.random.default_rng(2026)
    draw_count = 30_000

    counts = Counter(
        (
            selected.event,
            selected.actor,
        )
        for selected in (
            sample_transition_option(
                distribution,
                rng,
            )
            for _ in range(draw_count)
        )
    )

    assert (
        counts[(TransitionEvent.STAY, None)]
        / draw_count
    ) == pytest.approx(0.70, abs=0.01)

    assert (
        counts[
            (
                TransitionEvent.CLINCH_ENTRY,
                FighterSide.RED,
            )
        ]
        / draw_count
    ) == pytest.approx(0.20, abs=0.01)

    assert (
        counts[
            (
                TransitionEvent.TAKEDOWN,
                FighterSide.BLUE,
            )
        ]
        / draw_count
    ) == pytest.approx(0.10, abs=0.01)


def test_sample_and_apply_rejects_source_phase_mismatch() -> None:
    distribution = TransitionDistribution(
        source_phase=FightPhase.CLINCH,
        options=(
            TransitionProbability(
                event=TransitionEvent.STAY,
                actor=None,
                probability=1.0,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="source phase",
    ):
        sample_and_apply_transition(
            state(
                FightPhase.DISTANCE,
                owner=None,
                age=0,
                quality=0.0,
            ),
            distribution,
            np.random.default_rng(1),
        )


def test_stay_advances_segment_and_phase_age() -> None:
    transition = apply_transition_option(
        state(
            FightPhase.GROUND,
            owner=FighterSide.RED,
            age=2,
            quality=0.72,
            segment=4,
        ),
        option(
            TransitionEvent.STAY,
            None,
        ),
    )

    next_state = transition.next_state

    assert next_state.phase is FightPhase.GROUND
    assert next_state.phase_owner is FighterSide.RED
    assert next_state.position_quality == 0.72
    assert next_state.phase_age_segments == 3
    assert next_state.segment_number == 5


def test_clinch_entry_sets_actor_as_owner() -> None:
    transition = apply_transition_option(
        state(
            FightPhase.DISTANCE,
            owner=None,
            age=3,
            quality=0.0,
        ),
        option(
            TransitionEvent.CLINCH_ENTRY,
            FighterSide.RED,
        ),
    )

    next_state = transition.next_state

    assert next_state.phase is FightPhase.CLINCH
    assert next_state.phase_owner is FighterSide.RED
    assert next_state.position_quality == 0.30
    assert next_state.phase_age_segments == 0


def test_takedown_sets_actor_as_ground_owner() -> None:
    transition = apply_transition_option(
        state(
            FightPhase.CLINCH,
            owner=FighterSide.RED,
            age=1,
            quality=0.40,
        ),
        option(
            TransitionEvent.TAKEDOWN,
            FighterSide.BLUE,
        ),
    )

    next_state = transition.next_state

    assert next_state.phase is FightPhase.GROUND
    assert next_state.phase_owner is FighterSide.BLUE
    assert next_state.position_quality == 0.55
    assert next_state.phase_age_segments == 0


def test_clinch_break_returns_to_distance() -> None:
    transition = apply_transition_option(
        state(
            FightPhase.CLINCH,
            owner=FighterSide.RED,
            age=2,
            quality=0.45,
        ),
        option(
            TransitionEvent.CLINCH_BREAK,
            None,
        ),
    )

    next_state = transition.next_state

    assert next_state.phase is FightPhase.DISTANCE
    assert next_state.phase_owner is None
    assert next_state.position_quality == 0.0
    assert next_state.phase_age_segments == 0


def test_clinch_ownership_change_sets_new_owner() -> None:
    transition = apply_transition_option(
        state(
            FightPhase.CLINCH,
            owner=FighterSide.RED,
            age=2,
            quality=0.60,
        ),
        option(
            TransitionEvent.OWNERSHIP_CHANGE,
            FighterSide.BLUE,
        ),
    )

    next_state = transition.next_state

    assert next_state.phase is FightPhase.CLINCH
    assert next_state.phase_owner is FighterSide.BLUE
    assert next_state.position_quality == 0.30


def test_ground_escape_returns_to_distance() -> None:
    transition = apply_transition_option(
        state(
            FightPhase.GROUND,
            owner=FighterSide.RED,
            age=3,
            quality=0.70,
        ),
        option(
            TransitionEvent.GROUND_ESCAPE,
            FighterSide.BLUE,
        ),
    )

    assert transition.next_state.phase is FightPhase.DISTANCE
    assert transition.next_state.phase_owner is None
    assert transition.next_state.position_quality == 0.0


def test_scramble_moves_defender_to_clinch_owner() -> None:
    transition = apply_transition_option(
        state(
            FightPhase.GROUND,
            owner=FighterSide.RED,
            age=2,
            quality=0.65,
        ),
        option(
            TransitionEvent.SCRAMBLE_TO_CLINCH,
            FighterSide.BLUE,
        ),
    )

    next_state = transition.next_state

    assert next_state.phase is FightPhase.CLINCH
    assert next_state.phase_owner is FighterSide.BLUE
    assert next_state.position_quality == 0.25


def test_reversal_changes_ground_owner() -> None:
    transition = apply_transition_option(
        state(
            FightPhase.GROUND,
            owner=FighterSide.RED,
            age=4,
            quality=0.80,
        ),
        option(
            TransitionEvent.REVERSAL,
            FighterSide.BLUE,
        ),
    )

    next_state = transition.next_state

    assert next_state.phase is FightPhase.GROUND
    assert next_state.phase_owner is FighterSide.BLUE
    assert next_state.position_quality == 0.40


def test_end_of_round_requires_reset() -> None:
    with pytest.raises(
        ValueError,
        match="end-of-round state",
    ):
        apply_transition_option(
            state(
                FightPhase.DISTANCE,
                owner=None,
                age=5,
                quality=0.0,
                segment=10,
            ),
            option(
                TransitionEvent.STAY,
                None,
            ),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("clinch_entry_position_quality", -0.01),
        ("takedown_position_quality", 1.01),
        (
            "ownership_change_position_quality",
            float("nan"),
        ),
        ("scramble_position_quality", float("inf")),
        ("reversal_position_quality", -0.01),
    ],
)
def test_position_quality_calibration_is_bounded(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        replace(
            TransitionStateCalibration(),
            **{field_name: value},
        )
