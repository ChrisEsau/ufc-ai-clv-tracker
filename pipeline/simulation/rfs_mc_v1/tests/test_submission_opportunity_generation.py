"""Locked regression tests for submission opportunity generation."""

import numpy as np
import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v1.segment_engine import (
    DEFAULT_ACTIVITY_PARAMETERS,
    SegmentActivity,
    _sample_submission_attempts,
    _thin_striking_activity,
)


def make_activity(
    *,
    phase: FightPhase,
    control_seconds: int,
    submission_attempts: int,
) -> SegmentActivity:
    """Build minimal segment activity for eligibility tests."""

    return SegmentActivity(
        phase=phase,
        sig_str_attempted=0,
        sig_str_landed=0,
        td_attempted=0,
        td_landed=0,
        control_seconds=control_seconds,
        ground_str_attempted=0,
        ground_str_landed=0,
        submission_attempts=submission_attempts,
        knockdowns=0,
    )


def test_distance_attempt_without_control_is_rejected() -> None:
    """A submission attempt cannot occur at distance without control."""

    with pytest.raises(
        ValueError,
        match="submission attempts require",
    ):
        make_activity(
            phase=FightPhase.DISTANCE,
            control_seconds=0,
            submission_attempts=1,
        )


def test_ground_attempt_is_allowed() -> None:
    """Ground position is a valid submission opportunity."""

    activity = make_activity(
        phase=FightPhase.GROUND,
        control_seconds=0,
        submission_attempts=1,
    )

    assert activity.submission_attempts == 1


def test_control_attempt_is_allowed_outside_ground_phase() -> None:
    """Positive control time is also a valid opportunity."""

    activity = make_activity(
        phase=FightPhase.CLINCH,
        control_seconds=10,
        submission_attempts=1,
    )

    assert activity.submission_attempts == 1


def test_sampler_never_generates_attempt_without_position() -> None:
    """The attempt sampler must return zero outside eligible positions."""

    rng = np.random.default_rng(42)

    attempts = [
        _sample_submission_attempts(
            parameters=DEFAULT_ACTIVITY_PARAMETERS,
            phase=FightPhase.DISTANCE,
            control_seconds=0,
            td_landed=0,
            rng=rng,
        )
        for _ in range(10_000)
    ]

    assert sum(attempts) == 0



def test_forced_clinch_clears_ground_submission_activity() -> None:
    """A phase adjustment cannot preserve invalid ground activity."""

    original = SegmentActivity(
        phase=FightPhase.GROUND,
        sig_str_attempted=4,
        sig_str_landed=2,
        td_attempted=0,
        td_landed=0,
        control_seconds=0,
        ground_str_attempted=3,
        ground_str_landed=2,
        submission_attempts=1,
        knockdowns=0,
    )

    adjusted = _thin_striking_activity(
        activity=original,
        attempt_multiplier=0.60,
        power_multiplier=0.55,
        forced_phase=FightPhase.CLINCH,
        rng=np.random.default_rng(42),
    )

    assert adjusted.phase == FightPhase.CLINCH
    assert adjusted.control_seconds == 0
    assert adjusted.ground_str_attempted == 0
    assert adjusted.ground_str_landed == 0
    assert adjusted.submission_attempts == 0
