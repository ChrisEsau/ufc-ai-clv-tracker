"""Tests for finish-aware Monte Carlo fight paths."""

from pipeline.simulation.rfs_mc_v1.contracts import (
    FighterSimulationProfile,
    MatchupSimulationRequest,
    ParameterEstimate,
    ProfileSource,
)
from pipeline.simulation.rfs_mc_v1.runner import (
    simulate_finish_aware_path,
    simulate_finish_aware_paths,
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
        "defensive_deterioration": 0.4,
        "late_sig_output_ratio": 1.0,
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
    seed: int = 44,
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
        calibration_version="finish_aware_v0",
    )


def test_finish_aware_path_is_reproducible() -> None:
    request = make_request()

    first = simulate_finish_aware_path(
        request,
        path_index=0,
        seed=123,
    )
    second = simulate_finish_aware_path(
        request,
        path_index=0,
        seed=123,
    )

    assert first == second


def test_path_never_exceeds_scheduled_segments() -> None:
    request = make_request(scheduled_rounds=3)

    result = simulate_finish_aware_path(
        request,
        path_index=0,
        seed=567,
    )

    assert 1 <= len(result.traces) <= (
        3 * SEGMENTS_PER_ROUND
    )


def test_decision_completes_all_segments() -> None:
    request = make_request()

    decision_path = None

    for seed in range(500):
        result = simulate_finish_aware_path(
            request,
            path_index=0,
            seed=seed,
        )
        if result.outcome.method == "decision":
            decision_path = result
            break

    assert decision_path is not None
    assert len(decision_path.traces) == (
        3 * SEGMENTS_PER_ROUND
    )
    assert decision_path.outcome.elapsed_seconds == 900


def test_finish_metadata_matches_terminal_trace() -> None:
    request = make_request()

    finished_path = None

    for seed in range(2000):
        result = simulate_finish_aware_path(
            request,
            path_index=0,
            seed=seed,
        )
        if result.outcome.method != "decision":
            finished_path = result
            break

    assert finished_path is not None

    final_trace = finished_path.traces[-1]
    outcome = finished_path.outcome

    assert final_trace.finish_result.finished is True
    assert outcome.winner == final_trace.finish_result.winner
    assert outcome.loser == final_trace.finish_result.loser
    assert outcome.method == final_trace.finish_result.method.value
    assert outcome.finish_round == final_trace.activity.round_number
    assert outcome.finish_segment == final_trace.activity.segment_number

    expected_elapsed = (
        (outcome.finish_round - 1)
        * SEGMENTS_PER_ROUND
        * 30
        + outcome.finish_segment * 30
    )

    assert outcome.elapsed_seconds == expected_elapsed


def test_no_segments_occur_after_finish() -> None:
    request = make_request()

    for seed in range(100):
        result = simulate_finish_aware_path(
            request,
            path_index=0,
            seed=seed,
        )

        finished_traces = [
            trace
            for trace in result.traces
            if trace.finish_result.finished
        ]

        assert len(finished_traces) <= 1

        if finished_traces:
            assert result.traces[-1] == finished_traces[0]


def test_multiple_paths_have_unique_seeds() -> None:
    request = make_request(path_count=20)

    paths = simulate_finish_aware_paths(request)

    seeds = [path.seed for path in paths]

    assert len(paths) == 20
    assert len(seeds) == len(set(seeds))


def test_path_totals_only_include_generated_segments() -> None:
    request = make_request()

    result = simulate_finish_aware_path(
        request,
        path_index=0,
        seed=998,
    )

    red_attempts = sum(
        trace.activity.red.sig_str_attempted
        for trace in result.traces
    )
    blue_attempts = sum(
        trace.activity.blue.sig_str_attempted
        for trace in result.traces
    )

    assert result.red_totals["sig_str_attempted"] == red_attempts
    assert result.blue_totals["sig_str_attempted"] == blue_attempts
