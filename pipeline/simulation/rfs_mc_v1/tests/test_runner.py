"""Tests for the activity-only Monte Carlo runner."""

from pipeline.simulation.rfs_mc_v1.contracts import (
    FighterSimulationProfile,
    MatchupSimulationRequest,
    ParameterEstimate,
    ProfileSource,
)
from pipeline.simulation.rfs_mc_v1.runner import (
    simulate_activity_path,
    simulate_activity_paths,
    summarize_activity_simulation,
)
from pipeline.simulation.rfs_mc_v1.segment_engine import (
    SEGMENTS_PER_ROUND,
)


def make_profile(
    fighter_id: str,
    *,
    scheduled_rounds: int = 3,
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
        scheduled_rounds=scheduled_rounds,
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


def make_request(
    *,
    path_count: int = 5,
    seed: int = 123,
    scheduled_rounds: int = 3,
) -> MatchupSimulationRequest:
    return MatchupSimulationRequest(
        red_profile=make_profile(
            "red",
            scheduled_rounds=scheduled_rounds,
        ),
        blue_profile=make_profile(
            "blue",
            scheduled_rounds=scheduled_rounds,
        ),
        path_count=path_count,
        seed=seed,
        calibration_version="activity_v0",
    )


def test_activity_path_has_expected_segment_count() -> None:
    request = make_request(scheduled_rounds=3)

    path = simulate_activity_path(
        request,
        path_index=0,
        seed=99,
    )

    assert len(path.segments) == 3 * SEGMENTS_PER_ROUND


def test_five_round_path_has_expected_segment_count() -> None:
    request = make_request(scheduled_rounds=5)

    path = simulate_activity_path(
        request,
        path_index=0,
        seed=99,
    )

    assert len(path.segments) == 5 * SEGMENTS_PER_ROUND


def test_activity_paths_are_reproducible() -> None:
    request = make_request(path_count=4, seed=456)

    first = simulate_activity_paths(request)
    second = simulate_activity_paths(request)

    assert first == second


def test_different_root_seeds_change_results() -> None:
    first = simulate_activity_paths(
        make_request(path_count=4, seed=1)
    )
    second = simulate_activity_paths(
        make_request(path_count=4, seed=2)
    )

    assert first.paths != second.paths


def test_each_path_has_unique_child_seed() -> None:
    result = simulate_activity_paths(
        make_request(path_count=20, seed=77)
    )

    seeds = [path.seed for path in result.paths]

    assert len(seeds) == len(set(seeds))


def test_summary_contains_expected_metrics() -> None:
    result = simulate_activity_paths(
        make_request(path_count=10, seed=91)
    )

    summary = summarize_activity_simulation(result)

    assert summary["path_count"] == 10
    assert "sig_str_attempted" in summary["red"]
    assert "control_seconds" in summary["blue"]
    assert summary["red"]["sig_str_attempted"]["mean"] >= 0
