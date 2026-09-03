"""Tests for simulation aggregation and CLI parsing."""

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import (
    FighterSimulationProfile,
    MatchupSimulationRequest,
    ParameterEstimate,
    ProfileSource,
)
from pipeline.simulation.rfs_mc_v1.run_simulation import (
    build_parser,
)
from pipeline.simulation.rfs_mc_v1.runner import (
    simulate_scored_paths,
    summarize_scored_paths,
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
        "defensive_deterioration": 0.4,
        "late_sig_output_ratio": 1.0,
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


def make_request(
    path_count: int = 50,
) -> MatchupSimulationRequest:
    return MatchupSimulationRequest(
        red_profile=make_profile("red"),
        blue_profile=make_profile("blue"),
        path_count=path_count,
        seed=123,
        calibration_version="summary_test",
    )


def test_summary_probabilities_sum_to_one() -> None:
    paths = simulate_scored_paths(
        make_request(path_count=60)
    )
    summary = summarize_scored_paths(paths)

    total = (
        summary["red_win_probability"]
        + summary["blue_win_probability"]
        + summary["draw_probability"]
    )

    assert total == pytest.approx(1.0)


def test_finish_and_distance_probabilities_sum_to_one() -> None:
    paths = simulate_scored_paths(
        make_request(path_count=60)
    )
    summary = summarize_scored_paths(paths)

    total = (
        summary["finish_probability"]
        + summary["distance_probability"]
    )

    assert total == pytest.approx(1.0)


def test_method_probabilities_sum_to_one() -> None:
    paths = simulate_scored_paths(
        make_request(path_count=60)
    )
    summary = summarize_scored_paths(paths)

    total = sum(
        summary["method_probabilities"].values()
    )

    assert total == pytest.approx(1.0)


def test_cli_parser_accepts_required_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--red-fighter-id",
            "red",
            "--blue-fighter-id",
            "blue",
            "--target-date",
            "2026-08-10",
            "--weight-class",
            "Lightweight",
            "--gender",
            "male",
        ]
    )

    assert args.red_fighter_id == "red"
    assert args.blue_fighter_id == "blue"
    assert args.paths == 1000
    assert args.scheduled_rounds == 3
