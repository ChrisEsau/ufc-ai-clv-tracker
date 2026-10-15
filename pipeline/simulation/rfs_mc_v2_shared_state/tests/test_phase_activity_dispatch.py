"""Tests for V2 unified phase-activity dispatch."""

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_activity_dispatch import (
    generate_phase_segment_activity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)


def phase_parameters(
    *,
    distance_attempt_rate: float = 0.0,
    clinch_control_mean: float = 0.0,
    ground_control_mean: float = 0.0,
    ground_advancement_probability: float = 0.0,
    escape_attempt_rate: float = 0.0,
) -> FighterPhaseParameters:
    """Build a controlled fighter phase-parameter bundle."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=distance_attempt_rate,
            sig_strike_accuracy=0.50,
            knockdown_probability_per_landed=0.02,
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=0.0,
            clinch_strike_accuracy=0.50,
            control_seconds_mean=clinch_control_mean,
            damaging_clinch_probability=0.10,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=0.0,
            ground_strike_accuracy=0.50,
            control_seconds_mean=ground_control_mean,
            submission_attempt_rate=0.0,
            position_advancement_probability=(
                ground_advancement_probability
            ),
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=escape_attempt_rate,
            reversal_attempt_rate=0.0,
            scramble_attempt_rate=0.0,
            submission_defense=0.70,
        ),
    )


def state(
    phase: FightPhase,
    *,
    owner: FighterSide | None,
    quality: float,
) -> SharedFightState:
    """Build one valid shared phase state."""

    return SharedFightState(
        phase=phase,
        phase_owner=owner,
        phase_age_segments=1,
        position_quality=quality,
        round_number=1,
        segment_number=2,
    )


def test_distance_state_routes_to_distance_engine() -> None:
    shared_state = state(
        FightPhase.DISTANCE,
        owner=None,
        quality=0.0,
    )

    result = generate_phase_segment_activity(
        shared_state,
        phase_parameters(),
        phase_parameters(),
        np.random.default_rng(1),
    )

    assert isinstance(result, DistanceSegmentActivity)
    assert result.state == shared_state


def test_clinch_state_routes_to_clinch_engine() -> None:
    shared_state = state(
        FightPhase.CLINCH,
        owner=FighterSide.RED,
        quality=0.40,
    )

    result = generate_phase_segment_activity(
        shared_state,
        phase_parameters(
            clinch_control_mean=30.0,
        ),
        phase_parameters(
            clinch_control_mean=30.0,
        ),
        np.random.default_rng(2),
    )

    assert isinstance(result, ClinchSegmentActivity)
    assert result.state == shared_state

    assert result.red.control_seconds == 30
    assert result.blue.control_seconds == 0


def test_blue_clinch_owner_uses_blue_control_parameters() -> None:
    result = generate_phase_segment_activity(
        state(
            FightPhase.CLINCH,
            owner=FighterSide.BLUE,
            quality=0.40,
        ),
        phase_parameters(
            clinch_control_mean=0.0,
        ),
        phase_parameters(
            clinch_control_mean=30.0,
        ),
        np.random.default_rng(3),
    )

    assert isinstance(result, ClinchSegmentActivity)
    assert result.red.control_seconds == 0
    assert result.blue.control_seconds == 30


def test_red_ground_owner_uses_red_owner_parameters() -> None:
    result = generate_phase_segment_activity(
        state(
            FightPhase.GROUND,
            owner=FighterSide.RED,
            quality=0.65,
        ),
        phase_parameters(
            ground_control_mean=30.0,
            ground_advancement_probability=1.0,
        ),
        phase_parameters(
            ground_control_mean=0.0,
            escape_attempt_rate=0.0,
        ),
        np.random.default_rng(4),
    )

    assert isinstance(result, GroundSegmentActivity)

    assert result.red.control_seconds == 30
    assert result.red.position_advancements == 1

    assert result.blue.control_seconds == 0
    assert result.blue.position_advancements == 0


def test_blue_ground_owner_uses_blue_owner_parameters() -> None:
    result = generate_phase_segment_activity(
        state(
            FightPhase.GROUND,
            owner=FighterSide.BLUE,
            quality=0.65,
        ),
        phase_parameters(
            ground_control_mean=0.0,
            escape_attempt_rate=0.0,
        ),
        phase_parameters(
            ground_control_mean=30.0,
            ground_advancement_probability=1.0,
        ),
        np.random.default_rng(5),
    )

    assert isinstance(result, GroundSegmentActivity)

    assert result.red.control_seconds == 0
    assert result.red.position_advancements == 0

    assert result.blue.control_seconds == 30
    assert result.blue.position_advancements == 1


def test_ground_defender_parameters_come_from_non_owner() -> None:
    rng = np.random.default_rng(6)

    results = [
        generate_phase_segment_activity(
            state(
                FightPhase.GROUND,
                owner=FighterSide.RED,
                quality=0.65,
            ),
            phase_parameters(
                ground_control_mean=0.0,
            ),
            phase_parameters(
                escape_attempt_rate=2.0,
            ),
            rng,
        )
        for _ in range(100)
    ]

    assert all(
        result.red.escape_attempts == 0
        for result in results
    )

    assert any(
        result.blue.escape_attempts > 0
        for result in results
    )


def test_dispatch_is_deterministic_for_same_seed() -> None:
    shared_state = state(
        FightPhase.DISTANCE,
        owner=None,
        quality=0.0,
    )

    red = phase_parameters(
        distance_attempt_rate=4.0,
    )
    blue = phase_parameters(
        distance_attempt_rate=3.0,
    )

    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first = [
        generate_phase_segment_activity(
            shared_state,
            red,
            blue,
            first_rng,
        )
        for _ in range(100)
    ]

    second = [
        generate_phase_segment_activity(
            shared_state,
            red,
            blue,
            second_rng,
        )
        for _ in range(100)
    ]

    assert first == second
