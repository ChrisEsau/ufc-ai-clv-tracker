"""Tests for V2 ground-phase activity generation."""

from dataclasses import replace

import numpy as np
import pytest

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundFighterActivity,
    generate_ground_segment_activity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)


def ground_state(
    owner: FighterSide = FighterSide.RED,
) -> SharedFightState:
    """Return a valid shared ground state."""

    return SharedFightState(
        phase=FightPhase.GROUND,
        phase_owner=owner,
        phase_age_segments=2,
        position_quality=0.65,
        round_number=1,
        segment_number=3,
    )


def owner_parameters(
    **overrides: float,
) -> GroundOwnerRateParameters:
    """Build valid ground-owner rates."""

    baseline = GroundOwnerRateParameters(
        ground_strike_attempt_rate=2.0,
        ground_strike_accuracy=0.52,
        control_seconds_mean=15.0,
        submission_attempt_rate=0.20,
        position_advancement_probability=0.25,
    )

    return replace(
        baseline,
        **overrides,
    )


def defender_parameters(
    **overrides: float,
) -> GroundDefenderRateParameters:
    """Build valid ground-defender rates."""

    baseline = GroundDefenderRateParameters(
        escape_attempt_rate=0.20,
        reversal_attempt_rate=0.08,
        scramble_attempt_rate=0.15,
        submission_defense=0.70,
    )

    return replace(
        baseline,
        **overrides,
    )


def aggregate_owner_activity(
    selected_parameters: GroundOwnerRateParameters,
    *,
    seed: int,
    segment_count: int = 10_000,
) -> tuple[int, int, int, int, int]:
    """Aggregate generated ground-owner activity."""

    rng = np.random.default_rng(seed)

    attempted = 0
    landed = 0
    control = 0
    submissions = 0
    advancements = 0

    passive_defender = defender_parameters(
        escape_attempt_rate=0.0,
        reversal_attempt_rate=0.0,
        scramble_attempt_rate=0.0,
    )

    for _ in range(segment_count):
        activity = generate_ground_segment_activity(
            ground_state(FighterSide.RED),
            selected_parameters,
            passive_defender,
            rng,
        ).red

        attempted += activity.ground_str_attempted
        landed += activity.ground_str_landed
        control += activity.control_seconds
        submissions += activity.submission_attempts
        advancements += activity.position_advancements

    return (
        attempted,
        landed,
        control,
        submissions,
        advancements,
    )


def aggregate_defender_activity(
    selected_parameters: GroundDefenderRateParameters,
    *,
    seed: int,
    segment_count: int = 10_000,
) -> tuple[int, int, int]:
    """Aggregate generated ground-defender activity."""

    rng = np.random.default_rng(seed)

    escapes = 0
    reversals = 0
    scrambles = 0

    passive_owner = owner_parameters(
        ground_strike_attempt_rate=0.0,
        control_seconds_mean=0.0,
        submission_attempt_rate=0.0,
        position_advancement_probability=0.0,
    )

    for _ in range(segment_count):
        activity = generate_ground_segment_activity(
            ground_state(FighterSide.RED),
            passive_owner,
            selected_parameters,
            rng,
        ).blue

        escapes += activity.escape_attempts
        reversals += activity.reversal_attempts
        scrambles += activity.scramble_attempts

    return escapes, reversals, scrambles


def test_valid_ground_activity_is_accepted() -> None:
    activity = GroundFighterActivity(
        ground_str_attempted=5,
        ground_str_landed=3,
        control_seconds=18,
        submission_attempts=1,
        position_advancements=1,
        escape_attempts=0,
        reversal_attempts=0,
        scramble_attempts=0,
    )

    assert activity.ground_str_attempted == 5
    assert activity.ground_str_landed == 3
    assert activity.control_seconds == 18
    assert activity.submission_attempts == 1


@pytest.mark.parametrize(
    "field_name",
    [
        "ground_str_attempted",
        "ground_str_landed",
        "control_seconds",
        "submission_attempts",
        "position_advancements",
        "escape_attempts",
        "reversal_attempts",
        "scramble_attempts",
    ],
)
def test_ground_activity_cannot_be_negative(
    field_name: str,
) -> None:
    values = {
        "ground_str_attempted": 2,
        "ground_str_landed": 1,
        "control_seconds": 5,
        "submission_attempts": 0,
        "position_advancements": 0,
        "escape_attempts": 0,
        "reversal_attempts": 0,
        "scramble_attempts": 0,
    }
    values[field_name] = -1

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        GroundFighterActivity(**values)


def test_ground_activity_counts_must_be_integers() -> None:
    with pytest.raises(
        TypeError,
        match="submission_attempts",
    ):
        GroundFighterActivity(
            ground_str_attempted=2,
            ground_str_landed=1,
            control_seconds=5,
            submission_attempts=0.5,
            position_advancements=0,
            escape_attempts=0,
            reversal_attempts=0,
            scramble_attempts=0,
        )


def test_landed_ground_strikes_cannot_exceed_attempts() -> None:
    with pytest.raises(
        ValueError,
        match="ground_str_landed cannot exceed",
    ):
        GroundFighterActivity(
            ground_str_attempted=2,
            ground_str_landed=3,
            control_seconds=0,
            submission_attempts=0,
            position_advancements=0,
            escape_attempts=0,
            reversal_attempts=0,
            scramble_attempts=0,
        )


def test_control_time_cannot_exceed_segment() -> None:
    with pytest.raises(
        ValueError,
        match="control_seconds cannot exceed",
    ):
        GroundFighterActivity(
            ground_str_attempted=0,
            ground_str_landed=0,
            control_seconds=31,
            submission_attempts=0,
            position_advancements=0,
            escape_attempts=0,
            reversal_attempts=0,
            scramble_attempts=0,
        )


def test_position_advancement_cannot_exceed_one() -> None:
    with pytest.raises(
        ValueError,
        match="position_advancements cannot exceed",
    ):
        GroundFighterActivity(
            ground_str_attempted=0,
            ground_str_landed=0,
            control_seconds=0,
            submission_attempts=0,
            position_advancements=2,
            escape_attempts=0,
            reversal_attempts=0,
            scramble_attempts=0,
        )


def test_generation_requires_shared_ground_state() -> None:
    with pytest.raises(
        ValueError,
        match="requires a ground",
    ):
        generate_ground_segment_activity(
            SharedFightState.opening_state(),
            owner_parameters(),
            defender_parameters(),
            np.random.default_rng(1),
        )


def test_same_seed_produces_same_activity_sequence() -> None:
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first = [
        generate_ground_segment_activity(
            ground_state(FighterSide.RED),
            owner_parameters(),
            defender_parameters(),
            first_rng,
        )
        for _ in range(100)
    ]

    second = [
        generate_ground_segment_activity(
            ground_state(FighterSide.RED),
            owner_parameters(),
            defender_parameters(),
            second_rng,
        )
        for _ in range(100)
    ]

    assert first == second


def test_red_owner_generates_only_owner_activity() -> None:
    rng = np.random.default_rng(100)

    results = [
        generate_ground_segment_activity(
            ground_state(FighterSide.RED),
            owner_parameters(
                ground_strike_attempt_rate=4.0,
                control_seconds_mean=20.0,
                submission_attempt_rate=0.75,
                position_advancement_probability=0.75,
            ),
            defender_parameters(
                escape_attempt_rate=0.75,
                reversal_attempt_rate=0.50,
                scramble_attempt_rate=0.75,
            ),
            rng,
        )
        for _ in range(200)
    ]

    assert any(
        result.red.ground_str_attempted > 0
        or result.red.control_seconds > 0
        or result.red.submission_attempts > 0
        for result in results
    )

    assert all(
        result.red.escape_attempts == 0
        and result.red.reversal_attempts == 0
        and result.red.scramble_attempts == 0
        for result in results
    )

    assert all(
        result.blue.ground_str_attempted == 0
        and result.blue.ground_str_landed == 0
        and result.blue.control_seconds == 0
        and result.blue.submission_attempts == 0
        and result.blue.position_advancements == 0
        for result in results
    )


def test_blue_owner_generates_only_owner_activity() -> None:
    rng = np.random.default_rng(101)

    results = [
        generate_ground_segment_activity(
            ground_state(FighterSide.BLUE),
            owner_parameters(
                ground_strike_attempt_rate=4.0,
                control_seconds_mean=20.0,
                submission_attempt_rate=0.75,
            ),
            defender_parameters(
                escape_attempt_rate=0.75,
                reversal_attempt_rate=0.50,
                scramble_attempt_rate=0.75,
            ),
            rng,
        )
        for _ in range(200)
    ]

    assert all(
        result.red.ground_str_attempted == 0
        and result.red.control_seconds == 0
        and result.red.submission_attempts == 0
        for result in results
    )

    assert all(
        result.blue.escape_attempts == 0
        and result.blue.reversal_attempts == 0
        and result.blue.scramble_attempts == 0
        for result in results
    )


def test_zero_rates_produce_zero_activity() -> None:
    result = generate_ground_segment_activity(
        ground_state(FighterSide.RED),
        owner_parameters(
            ground_strike_attempt_rate=0.0,
            control_seconds_mean=0.0,
            submission_attempt_rate=0.0,
            position_advancement_probability=0.0,
        ),
        defender_parameters(
            escape_attempt_rate=0.0,
            reversal_attempt_rate=0.0,
            scramble_attempt_rate=0.0,
        ),
        np.random.default_rng(7),
    )

    zero = GroundFighterActivity(
        ground_str_attempted=0,
        ground_str_landed=0,
        control_seconds=0,
        submission_attempts=0,
        position_advancements=0,
        escape_attempts=0,
        reversal_attempts=0,
        scramble_attempts=0,
    )

    assert result.red == zero
    assert result.blue == zero


def test_higher_ground_strike_rate_increases_attempts() -> None:
    low_attempts, _, _, _, _ = aggregate_owner_activity(
        owner_parameters(
            ground_strike_attempt_rate=0.5,
        ),
        seed=10,
    )
    high_attempts, _, _, _, _ = aggregate_owner_activity(
        owner_parameters(
            ground_strike_attempt_rate=3.0,
        ),
        seed=10,
    )

    assert high_attempts > low_attempts * 5


def test_higher_ground_accuracy_increases_landings() -> None:
    _, low_landed, _, _, _ = aggregate_owner_activity(
        owner_parameters(
            ground_strike_attempt_rate=4.0,
            ground_strike_accuracy=0.20,
        ),
        seed=20,
    )
    _, high_landed, _, _, _ = aggregate_owner_activity(
        owner_parameters(
            ground_strike_attempt_rate=4.0,
            ground_strike_accuracy=0.80,
        ),
        seed=20,
    )

    assert high_landed > low_landed * 3


def test_higher_control_mean_increases_control() -> None:
    _, _, low_control, _, _ = aggregate_owner_activity(
        owner_parameters(
            control_seconds_mean=4.0,
        ),
        seed=30,
    )
    _, _, high_control, _, _ = aggregate_owner_activity(
        owner_parameters(
            control_seconds_mean=20.0,
        ),
        seed=30,
    )

    assert high_control > low_control * 4


def test_higher_submission_rate_increases_attempts() -> None:
    _, _, _, low_submissions, _ = aggregate_owner_activity(
        owner_parameters(
            submission_attempt_rate=0.05,
        ),
        seed=40,
    )
    _, _, _, high_submissions, _ = aggregate_owner_activity(
        owner_parameters(
            submission_attempt_rate=0.75,
        ),
        seed=40,
    )

    assert high_submissions > low_submissions * 10


def test_higher_advancement_probability_increases_advancements() -> None:
    _, _, _, _, low_advancements = aggregate_owner_activity(
        owner_parameters(
            position_advancement_probability=0.10,
        ),
        seed=50,
    )
    _, _, _, _, high_advancements = aggregate_owner_activity(
        owner_parameters(
            position_advancement_probability=0.80,
        ),
        seed=50,
    )

    assert high_advancements > low_advancements * 6


def test_higher_escape_rate_increases_escape_attempts() -> None:
    low_escapes, _, _ = aggregate_defender_activity(
        defender_parameters(
            escape_attempt_rate=0.05,
        ),
        seed=60,
    )
    high_escapes, _, _ = aggregate_defender_activity(
        defender_parameters(
            escape_attempt_rate=0.75,
        ),
        seed=60,
    )

    assert high_escapes > low_escapes * 10


def test_higher_reversal_rate_increases_reversal_attempts() -> None:
    _, low_reversals, _ = aggregate_defender_activity(
        defender_parameters(
            reversal_attempt_rate=0.03,
        ),
        seed=70,
    )
    _, high_reversals, _ = aggregate_defender_activity(
        defender_parameters(
            reversal_attempt_rate=0.60,
        ),
        seed=70,
    )

    assert high_reversals > low_reversals * 12


def test_higher_scramble_rate_increases_scramble_attempts() -> None:
    _, _, low_scrambles = aggregate_defender_activity(
        defender_parameters(
            scramble_attempt_rate=0.05,
        ),
        seed=80,
    )
    _, _, high_scrambles = aggregate_defender_activity(
        defender_parameters(
            scramble_attempt_rate=0.75,
        ),
        seed=80,
    )

    assert high_scrambles > low_scrambles * 10


def test_submission_defense_is_not_sampled_as_activity() -> None:
    """Submission defense belongs to later conversion-hazard logic."""

    low_defense_rng = np.random.default_rng(90)
    high_defense_rng = np.random.default_rng(90)

    low_defense = [
        generate_ground_segment_activity(
            ground_state(FighterSide.RED),
            owner_parameters(),
            defender_parameters(
                submission_defense=0.10,
            ),
            low_defense_rng,
        )
        for _ in range(100)
    ]

    high_defense = [
        generate_ground_segment_activity(
            ground_state(FighterSide.RED),
            owner_parameters(),
            defender_parameters(
                submission_defense=0.95,
            ),
            high_defense_rng,
        )
        for _ in range(100)
    ]

    assert low_defense == high_defense
