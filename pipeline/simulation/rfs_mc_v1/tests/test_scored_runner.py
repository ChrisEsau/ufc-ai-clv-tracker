"""Tests for scorecard integration with simulated paths."""

from pipeline.simulation.rfs_mc_v1.contracts import (
    FighterSimulationProfile,
    MatchupSimulationRequest,
    ParameterEstimate,
    ProfileSource,
)
from pipeline.simulation.rfs_mc_v1.runner import (
    score_finish_aware_path,
    simulate_finish_aware_path,
    simulate_scored_paths,
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
    *,
    path_count: int = 5,
) -> MatchupSimulationRequest:
    return MatchupSimulationRequest(
        red_profile=make_profile("red"),
        blue_profile=make_profile("blue"),
        path_count=path_count,
        seed=123,
        calibration_version="scoring_v0",
    )


def test_finished_path_has_no_scorecard() -> None:
    request = make_request()

    finished_path = None

    for seed in range(2000):
        path = simulate_finish_aware_path(
            request,
            path_index=0,
            seed=seed,
        )
        if path.outcome.method != "decision":
            finished_path = path
            break

    assert finished_path is not None

    scored = score_finish_aware_path(finished_path)

    assert scored.decision is None
    assert scored.path == finished_path


def test_decision_path_receives_scorecard() -> None:
    request = make_request()

    decision_path = None

    for seed in range(500):
        path = simulate_finish_aware_path(
            request,
            path_index=0,
            seed=seed,
        )
        if path.outcome.method == "decision":
            decision_path = path
            break

    assert decision_path is not None

    scored = score_finish_aware_path(decision_path)

    assert scored.decision is not None
    assert len(scored.decision.round_scores) == 3
    assert scored.path.outcome.winner in {
        "red",
        "blue",
        None,
    }


def test_simulate_scored_paths_returns_requested_count() -> None:
    request = make_request(path_count=12)

    results = simulate_scored_paths(request)

    assert len(results) == 12
    assert all(
        result.path.outcome.method
        in {"ko_tko", "submission", "decision"}
        for result in results
    )
