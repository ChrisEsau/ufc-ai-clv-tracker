"""Tests for V2 static shared-state activity paths."""

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
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.path_runner import (
    run_shared_state_path,
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


def transition_parameters(
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


def phase_parameters(
    *,
    distance_attempt_rate: float = 3.0,
    clinch_attempt_rate: float = 1.5,
    ground_attempt_rate: float = 2.0,
    submission_attempt_rate: float = 0.20,
) -> FighterPhaseParameters:
    """Build a complete static phase-specific profile."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=distance_attempt_rate,
            sig_strike_accuracy=0.45,
            knockdown_probability_per_landed=0.02,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=clinch_attempt_rate,
            clinch_strike_accuracy=0.50,
            control_seconds_mean=8.0,
            damaging_clinch_probability=0.08,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=ground_attempt_rate,
            ground_strike_accuracy=0.52,
            control_seconds_mean=15.0,
            submission_attempt_rate=submission_attempt_rate,
            position_advancement_probability=0.25,
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=0.20,
            reversal_attempt_rate=0.08,
            scramble_attempt_rate=0.15,
            submission_defense=0.70,
        ),
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
    """The path must contain every 30-second scheduled segment."""

    path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        phase_parameters(),
        phase_parameters(),
        scheduled_rounds=scheduled_rounds,
        seed=42,
    )

    assert len(path.segments) == expected_segments


def test_same_seed_produces_identical_activity_path() -> None:
    """All shared states and activity draws must be reproducible."""

    first = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        phase_parameters(),
        phase_parameters(),
        scheduled_rounds=3,
        seed=2026,
    )

    second = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        phase_parameters(),
        phase_parameters(),
        scheduled_rounds=3,
        seed=2026,
    )

    assert first == second


def test_activity_path_matches_state_only_timeline() -> None:
    """Activity generation must not alter the transition stream."""

    red_transition = transition_parameters(
        takedown_entry_tendency=0.80,
        phase_imposition=0.75,
    )
    blue_transition = transition_parameters(
        takedown_resistance=0.70,
        phase_resistance=0.65,
    )

    state_only = run_shared_state_path(
        red_transition,
        blue_transition,
        scheduled_rounds=5,
        seed=5150,
    )

    activity_path = run_static_activity_path(
        red_transition,
        blue_transition,
        phase_parameters(),
        phase_parameters(),
        scheduled_rounds=5,
        seed=5150,
    )

    assert tuple(
        record.state
        for record in activity_path.segments
    ) == tuple(
        record.state
        for record in state_only.segments
    )

    assert tuple(
        record.transition
        for record in activity_path.segments
    ) == tuple(
        record.transition
        for record in state_only.segments
    )


def test_activity_parameters_do_not_change_phase_timeline() -> None:
    """Transition randomness must remain independent of activity rates."""

    zero_activity = phase_parameters(
        distance_attempt_rate=0.0,
        clinch_attempt_rate=0.0,
        ground_attempt_rate=0.0,
        submission_attempt_rate=0.0,
    )

    high_activity = phase_parameters(
        distance_attempt_rate=10.0,
        clinch_attempt_rate=8.0,
        ground_attempt_rate=8.0,
        submission_attempt_rate=1.5,
    )

    low_path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        zero_activity,
        zero_activity,
        scheduled_rounds=3,
        seed=101,
    )

    high_path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        high_activity,
        high_activity,
        scheduled_rounds=3,
        seed=101,
    )

    assert tuple(
        record.state
        for record in low_path.segments
    ) == tuple(
        record.state
        for record in high_path.segments
    )

    assert tuple(
        record.transition
        for record in low_path.segments
    ) == tuple(
        record.transition
        for record in high_path.segments
    )


def test_activity_parameters_change_generated_activity() -> None:
    """Different phase rates must still change segment activity."""

    zero_activity = phase_parameters(
        distance_attempt_rate=0.0,
        clinch_attempt_rate=0.0,
        ground_attempt_rate=0.0,
        submission_attempt_rate=0.0,
    )

    high_activity = phase_parameters(
        distance_attempt_rate=10.0,
        clinch_attempt_rate=8.0,
        ground_attempt_rate=8.0,
        submission_attempt_rate=1.5,
    )

    zero_path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        zero_activity,
        zero_activity,
        scheduled_rounds=3,
        seed=202,
    )

    high_path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        high_activity,
        high_activity,
        scheduled_rounds=3,
        seed=202,
    )

    zero_distance_attempts = sum(
        record.activity.red.sig_str_attempted
        + record.activity.blue.sig_str_attempted
        for record in zero_path.segments
        if isinstance(
            record.activity,
            DistanceSegmentActivity,
        )
    )

    high_distance_attempts = sum(
        record.activity.red.sig_str_attempted
        + record.activity.blue.sig_str_attempted
        for record in high_path.segments
        if isinstance(
            record.activity,
            DistanceSegmentActivity,
        )
    )

    assert zero_distance_attempts == 0
    assert high_distance_attempts > 0


def test_activity_type_matches_shared_phase() -> None:
    """Every segment must use exactly one matching activity engine."""

    path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        phase_parameters(),
        phase_parameters(),
        scheduled_rounds=5,
        seed=303,
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


def test_every_activity_record_uses_segment_state() -> None:
    """Activity cannot describe a different physical segment state."""

    path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        phase_parameters(),
        phase_parameters(),
        scheduled_rounds=3,
        seed=404,
    )

    for record in path.segments:
        assert record.activity.state == record.state


def test_transition_result_feeds_following_segment() -> None:
    """Each sampled transition must produce the next segment state."""

    path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        phase_parameters(),
        phase_parameters(),
        scheduled_rounds=3,
        seed=505,
    )

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
    """Activity paths preserve the required round-opening reset."""

    path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        phase_parameters(),
        phase_parameters(),
        scheduled_rounds=5,
        seed=606,
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
        assert isinstance(
            record.activity,
            DistanceSegmentActivity,
        )


def test_phase_specific_role_legality() -> None:
    """Owner and defender activity must remain physically consistent."""

    path = run_static_activity_path(
        transition_parameters(),
        transition_parameters(),
        phase_parameters(),
        phase_parameters(),
        scheduled_rounds=5,
        seed=707,
    )

    for record in path.segments:
        activity = record.activity

        if isinstance(activity, ClinchSegmentActivity):
            if record.state.phase_owner is FighterSide.RED:
                assert activity.blue.control_seconds == 0
            else:
                assert activity.red.control_seconds == 0

        elif isinstance(activity, GroundSegmentActivity):
            if record.state.phase_owner is FighterSide.RED:
                owner = activity.red
                defender = activity.blue
            else:
                owner = activity.blue
                defender = activity.red

            assert owner.escape_attempts == 0
            assert owner.reversal_attempts == 0
            assert owner.scramble_attempts == 0

            assert defender.ground_str_attempted == 0
            assert defender.ground_str_landed == 0
            assert defender.control_seconds == 0
            assert defender.submission_attempts == 0
            assert defender.position_advancements == 0


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
    """Only standard three- and five-round fights are supported."""

    with pytest.raises(
        ValueError,
        match="scheduled_rounds",
    ):
        run_static_activity_path(
            transition_parameters(),
            transition_parameters(),
            phase_parameters(),
            phase_parameters(),
            scheduled_rounds=scheduled_rounds,
            seed=1,
        )


def test_runner_rejects_negative_seed() -> None:
    """Simulation seeds must remain nonnegative."""

    with pytest.raises(
        ValueError,
        match="seed",
    ):
        run_static_activity_path(
            transition_parameters(),
            transition_parameters(),
            phase_parameters(),
            phase_parameters(),
            scheduled_rounds=3,
            seed=-1,
        )
