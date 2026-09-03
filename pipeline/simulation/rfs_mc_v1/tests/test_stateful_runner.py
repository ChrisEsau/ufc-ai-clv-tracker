"""Tests for stateful Monte Carlo activity paths."""

from pipeline.simulation.rfs_mc_v1.contracts import (
    FighterSimulationProfile,
    MatchupSimulationRequest,
    ParameterEstimate,
    ProfileSource,
)
from pipeline.simulation.rfs_mc_v1.runner import (
    simulate_stateful_activity_path,
    simulate_stateful_activity_paths,
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
    path_count: int = 3,
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
        calibration_version="dynamic_state_v0",
    )


def test_stateful_path_has_one_trace_per_segment() -> None:
    request = make_request()

    result = simulate_stateful_activity_path(
        request,
        path_index=0,
        seed=123,
    )

    assert len(result.traces) == 3 * SEGMENTS_PER_ROUND


def test_stateful_path_is_reproducible() -> None:
    request = make_request()

    first = simulate_stateful_activity_path(
        request,
        path_index=0,
        seed=123,
    )
    second = simulate_stateful_activity_path(
        request,
        path_index=0,
        seed=123,
    )

    assert first == second


def test_states_evolve_independently() -> None:
    request = make_request()

    result = simulate_stateful_activity_path(
        request,
        path_index=0,
        seed=333,
    )

    assert result.final_red_state is not result.final_blue_state

    red_signature = (
        result.final_red_state.energy,
        result.final_red_state.head_damage,
        result.final_red_state.cumulative_strike_activity,
    )
    blue_signature = (
        result.final_blue_state.energy,
        result.final_blue_state.head_damage,
        result.final_blue_state.cumulative_strike_activity,
    )

    assert red_signature != blue_signature


def test_trace_snapshots_do_not_alias_final_state() -> None:
    request = make_request()

    result = simulate_stateful_activity_path(
        request,
        path_index=0,
        seed=555,
    )

    assert result.traces[0].red_state is not result.final_red_state
    assert result.traces[0].blue_state is not result.final_blue_state


def test_final_states_are_valid() -> None:
    request = make_request()

    result = simulate_stateful_activity_path(
        request,
        path_index=0,
        seed=789,
    )

    result.final_red_state.validate()
    result.final_blue_state.validate()


def test_multiple_stateful_paths_have_unique_seeds() -> None:
    request = make_request(path_count=12)

    paths = simulate_stateful_activity_paths(request)

    seeds = [path.seed for path in paths]

    assert len(paths) == 12
    assert len(seeds) == len(set(seeds))
