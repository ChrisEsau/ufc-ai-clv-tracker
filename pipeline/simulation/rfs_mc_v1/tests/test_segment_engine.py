"""Tests for the RFS Monte Carlo V1 segment activity engine."""

import numpy as np
import pytest

from pipeline.simulation.rfs_mc_v1.contracts import (
    FighterSimulationProfile,
    ParameterEstimate,
    ProfileSource,
)
from pipeline.simulation.rfs_mc_v1.segment_engine import (
    ActivityParameters,
    SEGMENT_SECONDS,
    aggregate_segment_activity,
    build_activity_parameters,
    generate_matchup_segment,
    generate_segment_activity,
)


def make_profile(
    fighter_id: str,
) -> FighterSimulationProfile:
    values = {
        "phase_mix_disruption": 0.2,
        "wrestling_persistence": 0.4,
        "sig_attempt_trajectory": 0.5,
        "sig_accuracy_trajectory": 0.1,
        "control_per_td_attempt": 30.0,
        "control_to_damage": 1.5,
        "submission_pressure": 0.5,
        "knockdowns_absorbed": 0.2,
    }

    return FighterSimulationProfile(
        fighter_id=fighter_id,
        fighter_name=f"Fighter {fighter_id}",
        target_date="2026-08-05",
        weight_class="Lightweight",
        gender="male",
        scheduled_rounds=3,
        prior_fight_count=4,
        valid_round_fight_count=4,
        is_low_experience=False,
        parameters={
            name: ParameterEstimate(
                value=value,
                source=ProfileSource.FIGHTER,
                effective_sample_size=4.0,
                uncertainty=0.5,
            )
            for name, value in values.items()
        },
    )


def test_activity_parameters_require_phase_sum_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ActivityParameters(
            distance_phase_probability=0.5,
            clinch_phase_probability=0.2,
            ground_phase_probability=0.2,
            sig_attempt_rate=4.0,
            sig_accuracy=0.45,
            td_attempt_rate=0.3,
            td_accuracy=0.35,
            control_probability=0.3,
            mean_control_seconds=10.0,
            ground_attempt_rate=1.0,
            ground_accuracy=0.5,
            submission_attempt_rate=0.1,
            knockdown_probability=0.01,
        )


def test_build_activity_parameters_are_valid() -> None:
    parameters = build_activity_parameters(make_profile("red"))

    phase_total = (
        parameters.distance_phase_probability
        + parameters.clinch_phase_probability
        + parameters.ground_phase_probability
    )

    assert phase_total == pytest.approx(1.0)
    assert parameters.sig_attempt_rate > 0
    assert 0 <= parameters.sig_accuracy <= 1


def test_segment_generation_is_deterministic_for_seed() -> None:
    parameters = build_activity_parameters(make_profile("red"))

    first = generate_segment_activity(
        parameters,
        np.random.default_rng(1234),
    )
    second = generate_segment_activity(
        parameters,
        np.random.default_rng(1234),
    )

    assert first == second


def test_segment_counts_obey_contracts() -> None:
    parameters = build_activity_parameters(make_profile("red"))
    rng = np.random.default_rng(55)

    for _ in range(500):
        segment = generate_segment_activity(parameters, rng)

        assert segment.sig_str_landed <= segment.sig_str_attempted
        assert segment.td_landed <= segment.td_attempted
        assert (
            segment.ground_str_landed
            <= segment.ground_str_attempted
        )
        assert 0 <= segment.control_seconds <= SEGMENT_SECONDS


def test_matchup_segment_is_deterministic() -> None:
    red = make_profile("red")
    blue = make_profile("blue")

    first = generate_matchup_segment(
        red_profile=red,
        blue_profile=blue,
        round_number=1,
        segment_number=1,
        rng=np.random.default_rng(77),
    )
    second = generate_matchup_segment(
        red_profile=red,
        blue_profile=blue,
        round_number=1,
        segment_number=1,
        rng=np.random.default_rng(77),
    )

    assert first == second


def test_aggregate_segment_activity() -> None:
    parameters = build_activity_parameters(make_profile("red"))
    rng = np.random.default_rng(101)

    segments = [
        generate_segment_activity(parameters, rng)
        for _ in range(10)
    ]

    totals = aggregate_segment_activity(segments)

    assert totals["sig_str_attempted"] == sum(
        segment.sig_str_attempted for segment in segments
    )
    assert totals["control_seconds"] == sum(
        segment.control_seconds for segment in segments
    )
    assert totals["knockdowns"] == sum(
        segment.knockdowns for segment in segments
    )
