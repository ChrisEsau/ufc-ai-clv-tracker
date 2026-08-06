"""Tests for V2 clinch-phase activity generation."""

from dataclasses import replace

import numpy as np
import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchFighterActivity,
    generate_clinch_segment_activity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
)


def clinch_state(
    owner: FighterSide = FighterSide.RED,
) -> SharedFightState:
    """Return a valid shared clinch state."""

    return SharedFightState(
        phase=FightPhase.CLINCH,
        phase_owner=owner,
        phase_age_segments=2,
        position_quality=0.45,
        round_number=1,
        segment_number=3,
    )


def parameters(
    **overrides: float,
) -> ClinchRateParameters:
    """Build valid clinch rates with optional overrides."""

    baseline = ClinchRateParameters(
        clinch_strike_attempt_rate=1.5,
        clinch_strike_accuracy=0.50,
        control_seconds_mean=8.0,
        damaging_clinch_probability=0.08,
    )

    return replace(
        baseline,
        **overrides,
    )


def aggregate_owner_activity(
    selected_parameters: ClinchRateParameters,
    *,
    seed: int,
    segment_count: int = 10_000,
) -> tuple[int, int, int, int]:
    """Generate aggregate red-owner clinch activity."""

    rng = np.random.default_rng(seed)

    attempts = 0
    landed = 0
    damaging = 0
    control_seconds = 0

    passive_blue = parameters(
        clinch_strike_attempt_rate=0.0,
        control_seconds_mean=0.0,
    )

    for _ in range(segment_count):
        activity = generate_clinch_segment_activity(
            clinch_state(FighterSide.RED),
            selected_parameters,
            passive_blue,
            rng,
        ).red

        attempts += activity.clinch_str_attempted
        landed += activity.clinch_str_landed
        damaging += activity.damaging_clinch_strikes
        control_seconds += activity.control_seconds

    return (
        attempts,
        landed,
        damaging,
        control_seconds,
    )


def test_valid_clinch_activity_is_accepted() -> None:
    activity = ClinchFighterActivity(
        clinch_str_attempted=4,
        clinch_str_landed=2,
        damaging_clinch_strikes=1,
        control_seconds=12,
    )

    assert activity.clinch_str_attempted == 4
    assert activity.clinch_str_landed == 2
    assert activity.damaging_clinch_strikes == 1
    assert activity.control_seconds == 12


@pytest.mark.parametrize(
    "field_name",
    [
        "clinch_str_attempted",
        "clinch_str_landed",
        "damaging_clinch_strikes",
        "control_seconds",
    ],
)
def test_clinch_activity_cannot_be_negative(
    field_name: str,
) -> None:
    values = {
        "clinch_str_attempted": 2,
        "clinch_str_landed": 1,
        "damaging_clinch_strikes": 0,
        "control_seconds": 5,
    }
    values[field_name] = -1

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        ClinchFighterActivity(**values)


def test_landed_clinch_strikes_cannot_exceed_attempts() -> None:
    with pytest.raises(
        ValueError,
        match="clinch_str_landed cannot exceed",
    ):
        ClinchFighterActivity(
            clinch_str_attempted=2,
            clinch_str_landed=3,
            damaging_clinch_strikes=0,
            control_seconds=0,
        )


def test_damaging_strikes_cannot_exceed_landings() -> None:
    with pytest.raises(
        ValueError,
        match="damaging_clinch_strikes cannot exceed",
    ):
        ClinchFighterActivity(
            clinch_str_attempted=4,
            clinch_str_landed=2,
            damaging_clinch_strikes=3,
            control_seconds=0,
        )


def test_control_time_cannot_exceed_segment() -> None:
    with pytest.raises(
        ValueError,
        match="control_seconds cannot exceed",
    ):
        ClinchFighterActivity(
            clinch_str_attempted=0,
            clinch_str_landed=0,
            damaging_clinch_strikes=0,
            control_seconds=31,
        )


def test_generation_requires_shared_clinch_state() -> None:
    distance_state = SharedFightState.opening_state()

    with pytest.raises(
        ValueError,
        match="requires a clinch",
    ):
        generate_clinch_segment_activity(
            distance_state,
            parameters(),
            parameters(),
            np.random.default_rng(1),
        )


def test_same_seed_produces_same_activity_sequence() -> None:
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first = [
        generate_clinch_segment_activity(
            clinch_state(FighterSide.RED),
            parameters(),
            parameters(),
            first_rng,
        )
        for _ in range(100)
    ]

    second = [
        generate_clinch_segment_activity(
            clinch_state(FighterSide.RED),
            parameters(),
            parameters(),
            second_rng,
        )
        for _ in range(100)
    ]

    assert first == second


def test_only_red_owner_accumulates_control() -> None:
    rng = np.random.default_rng(100)

    results = [
        generate_clinch_segment_activity(
            clinch_state(FighterSide.RED),
            parameters(control_seconds_mean=20.0),
            parameters(control_seconds_mean=20.0),
            rng,
        )
        for _ in range(100)
    ]

    assert any(
        result.red.control_seconds > 0
        for result in results
    )
    assert all(
        result.blue.control_seconds == 0
        for result in results
    )


def test_only_blue_owner_accumulates_control() -> None:
    rng = np.random.default_rng(101)

    results = [
        generate_clinch_segment_activity(
            clinch_state(FighterSide.BLUE),
            parameters(control_seconds_mean=20.0),
            parameters(control_seconds_mean=20.0),
            rng,
        )
        for _ in range(100)
    ]

    assert all(
        result.red.control_seconds == 0
        for result in results
    )
    assert any(
        result.blue.control_seconds > 0
        for result in results
    )


def test_zero_rates_produce_zero_activity() -> None:
    zero = parameters(
        clinch_strike_attempt_rate=0.0,
        control_seconds_mean=0.0,
    )

    result = generate_clinch_segment_activity(
        clinch_state(FighterSide.RED),
        zero,
        zero,
        np.random.default_rng(7),
    )

    expected = ClinchFighterActivity(
        clinch_str_attempted=0,
        clinch_str_landed=0,
        damaging_clinch_strikes=0,
        control_seconds=0,
    )

    assert result.red == expected
    assert result.blue == expected


def test_higher_attempt_rate_increases_attempts() -> None:
    low_attempts, _, _, _ = aggregate_owner_activity(
        parameters(
            clinch_strike_attempt_rate=0.5,
        ),
        seed=10,
    )
    high_attempts, _, _, _ = aggregate_owner_activity(
        parameters(
            clinch_strike_attempt_rate=3.0,
        ),
        seed=10,
    )

    assert high_attempts > low_attempts * 5


def test_higher_accuracy_increases_landings() -> None:
    _, low_landed, _, _ = aggregate_owner_activity(
        parameters(
            clinch_strike_attempt_rate=4.0,
            clinch_strike_accuracy=0.20,
        ),
        seed=20,
    )
    _, high_landed, _, _ = aggregate_owner_activity(
        parameters(
            clinch_strike_attempt_rate=4.0,
            clinch_strike_accuracy=0.80,
        ),
        seed=20,
    )

    assert high_landed > low_landed * 3


def test_higher_damage_probability_increases_damaging_strikes() -> None:
    _, _, low_damaging, _ = aggregate_owner_activity(
        parameters(
            clinch_strike_attempt_rate=5.0,
            clinch_strike_accuracy=1.0,
            damaging_clinch_probability=0.02,
        ),
        seed=30,
    )
    _, _, high_damaging, _ = aggregate_owner_activity(
        parameters(
            clinch_strike_attempt_rate=5.0,
            clinch_strike_accuracy=1.0,
            damaging_clinch_probability=0.30,
        ),
        seed=30,
    )

    assert high_damaging > low_damaging * 10


def test_higher_control_mean_increases_owner_control() -> None:
    _, _, _, low_control = aggregate_owner_activity(
        parameters(
            control_seconds_mean=4.0,
        ),
        seed=40,
    )
    _, _, _, high_control = aggregate_owner_activity(
        parameters(
            control_seconds_mean=20.0,
        ),
        seed=40,
    )

    assert high_control > low_control * 4
