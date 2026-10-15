"""Tests for V2 distance-phase activity generation."""

from dataclasses import replace

import numpy as np
import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceFighterActivity,
    generate_distance_segment_activity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    DistanceRateParameters,
)


def distance_state() -> SharedFightState:
    """Return a valid shared distance state."""

    return SharedFightState(
        phase=FightPhase.DISTANCE,
        phase_owner=None,
        phase_age_segments=2,
        position_quality=0.0,
        round_number=1,
        segment_number=3,
    )


def parameters(
    **overrides: float,
) -> DistanceRateParameters:
    """Build valid distance rates with optional overrides."""

    baseline = DistanceRateParameters(
        sig_strike_attempt_rate=3.0,
        sig_strike_accuracy=0.45,
        knockdown_probability_per_landed=0.02,
    )

    return replace(
        baseline,
        **overrides,
    )


def total_activity(
    selected_parameters: DistanceRateParameters,
    *,
    seed: int,
    segment_count: int = 10_000,
) -> tuple[int, int, int]:
    """Generate aggregate activity for one fighter."""

    rng = np.random.default_rng(seed)

    attempts = 0
    landed = 0
    knockdowns = 0

    for _ in range(segment_count):
        activity = generate_distance_segment_activity(
            distance_state(),
            selected_parameters,
            parameters(
                sig_strike_attempt_rate=0.0,
            ),
            rng,
        ).red

        attempts += activity.sig_str_attempted
        landed += activity.sig_str_landed
        knockdowns += activity.knockdowns

    return attempts, landed, knockdowns


def test_valid_distance_activity_is_accepted() -> None:
    activity = DistanceFighterActivity(
        sig_str_attempted=5,
        sig_str_landed=2,
        knockdowns=1,
    )

    assert activity.sig_str_attempted == 5
    assert activity.sig_str_landed == 2
    assert activity.knockdowns == 1


@pytest.mark.parametrize(
    "field_name",
    [
        "sig_str_attempted",
        "sig_str_landed",
        "knockdowns",
    ],
)
def test_distance_activity_cannot_be_negative(
    field_name: str,
) -> None:
    values = {
        "sig_str_attempted": 2,
        "sig_str_landed": 1,
        "knockdowns": 0,
    }
    values[field_name] = -1

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        DistanceFighterActivity(**values)


def test_landed_strikes_cannot_exceed_attempts() -> None:
    with pytest.raises(
        ValueError,
        match="sig_str_landed cannot exceed",
    ):
        DistanceFighterActivity(
            sig_str_attempted=2,
            sig_str_landed=3,
            knockdowns=0,
        )


def test_knockdowns_cannot_exceed_landed_strikes() -> None:
    with pytest.raises(
        ValueError,
        match="knockdowns cannot exceed",
    ):
        DistanceFighterActivity(
            sig_str_attempted=4,
            sig_str_landed=2,
            knockdowns=3,
        )


def test_generation_requires_shared_distance_state() -> None:
    clinch_state = SharedFightState(
        phase=FightPhase.CLINCH,
        phase_owner=FighterSide.RED,
        phase_age_segments=1,
        position_quality=0.40,
        round_number=1,
        segment_number=3,
    )

    with pytest.raises(
        ValueError,
        match="requires a distance",
    ):
        generate_distance_segment_activity(
            clinch_state,
            parameters(),
            parameters(),
            np.random.default_rng(1),
        )


def test_same_seed_produces_same_activity_sequence() -> None:
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first = [
        generate_distance_segment_activity(
            distance_state(),
            parameters(),
            parameters(),
            first_rng,
        )
        for _ in range(100)
    ]

    second = [
        generate_distance_segment_activity(
            distance_state(),
            parameters(),
            parameters(),
            second_rng,
        )
        for _ in range(100)
    ]

    assert first == second


def test_zero_attempt_rates_produce_zero_activity() -> None:
    zero = parameters(
        sig_strike_attempt_rate=0.0,
    )

    result = generate_distance_segment_activity(
        distance_state(),
        zero,
        zero,
        np.random.default_rng(7),
    )

    assert result.red == DistanceFighterActivity(0, 0, 0)
    assert result.blue == DistanceFighterActivity(0, 0, 0)


def test_higher_attempt_rate_increases_attempts() -> None:
    low_attempts, _, _ = total_activity(
        parameters(
            sig_strike_attempt_rate=1.0,
        ),
        seed=10,
    )
    high_attempts, _, _ = total_activity(
        parameters(
            sig_strike_attempt_rate=5.0,
        ),
        seed=10,
    )

    assert high_attempts > low_attempts * 4


def test_higher_accuracy_increases_landings() -> None:
    _, low_landed, _ = total_activity(
        parameters(
            sig_strike_attempt_rate=5.0,
            sig_strike_accuracy=0.20,
        ),
        seed=20,
    )
    _, high_landed, _ = total_activity(
        parameters(
            sig_strike_attempt_rate=5.0,
            sig_strike_accuracy=0.80,
        ),
        seed=20,
    )

    assert high_landed > low_landed * 3


def test_higher_knockdown_rate_increases_knockdowns() -> None:
    _, _, low_knockdowns = total_activity(
        parameters(
            sig_strike_attempt_rate=6.0,
            sig_strike_accuracy=1.0,
            knockdown_probability_per_landed=0.01,
        ),
        seed=30,
    )
    _, _, high_knockdowns = total_activity(
        parameters(
            sig_strike_attempt_rate=6.0,
            sig_strike_accuracy=1.0,
            knockdown_probability_per_landed=0.20,
        ),
        seed=30,
    )

    assert high_knockdowns > low_knockdowns * 10
