"""Tests for path-specific dynamic state updates."""

from copy import deepcopy

from pipeline.simulation.rfs_mc_v1.contracts import (
    FightPhase,
    FighterSimulationProfile,
    ParameterEstimate,
    ProfileSource,
)
from pipeline.simulation.rfs_mc_v1.dynamic_state import (
    apply_between_round_recovery,
    calculate_energy_cost,
    initialize_dynamic_state,
    update_dynamic_state,
)
from pipeline.simulation.rfs_mc_v1.segment_engine import SegmentActivity


def make_profile() -> FighterSimulationProfile:
    values = {
        "defensive_deterioration": 0.5,
        "knockdowns_absorbed": 0.4,
        "late_sig_output_ratio": 1.0,
    }

    return FighterSimulationProfile(
        fighter_id="fighter-a",
        fighter_name="Fighter A",
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


def make_activity(
    *,
    phase: FightPhase = FightPhase.DISTANCE,
    sig_attempted: int = 8,
    sig_landed: int = 4,
    td_attempted: int = 1,
    td_landed: int = 0,
    control_seconds: int = 0,
    ground_attempted: int = 0,
    ground_landed: int = 0,
    submission_attempts: int = 0,
    knockdowns: int = 0,
) -> SegmentActivity:
    return SegmentActivity(
        phase=phase,
        sig_str_attempted=sig_attempted,
        sig_str_landed=sig_landed,
        td_attempted=td_attempted,
        td_landed=td_landed,
        control_seconds=control_seconds,
        ground_str_attempted=ground_attempted,
        ground_str_landed=ground_landed,
        submission_attempts=submission_attempts,
        knockdowns=knockdowns,
    )


def test_initial_state_is_valid() -> None:
    state = initialize_dynamic_state(make_profile())
    state.validate()

    assert 0 < state.energy <= 1
    assert 0 < state.chin_integrity <= 1
    assert 0 < state.defensive_stability <= 1


def test_energy_cost_increases_with_workload() -> None:
    low = calculate_energy_cost(
        make_activity(sig_attempted=2, sig_landed=1)
    )
    high = calculate_energy_cost(
        make_activity(
            sig_attempted=15,
            sig_landed=8,
            td_attempted=3,
            td_landed=1,
        )
    )

    assert high > low


def test_received_activity_accumulates_damage() -> None:
    profile = make_profile()
    state = initialize_dynamic_state(profile)

    updated = update_dynamic_state(
        state=state,
        own_activity=make_activity(),
        opponent_activity=make_activity(
            sig_attempted=10,
            sig_landed=7,
            ground_attempted=4,
            ground_landed=3,
            knockdowns=1,
        ),
        profile=profile,
    )

    assert updated.head_damage > 0
    assert updated.body_damage > 0
    assert updated.leg_damage > 0
    assert updated.chin_integrity < 1
    assert updated.defensive_stability < 1


def test_update_is_deterministic() -> None:
    profile = make_profile()
    initial = initialize_dynamic_state(profile)

    first = update_dynamic_state(
        state=deepcopy(initial),
        own_activity=make_activity(),
        opponent_activity=make_activity(sig_landed=6),
        profile=profile,
    )
    second = update_dynamic_state(
        state=deepcopy(initial),
        own_activity=make_activity(),
        opponent_activity=make_activity(sig_landed=6),
        profile=profile,
    )

    assert first == second


def test_between_round_recovery_improves_bounded_state() -> None:
    profile = make_profile()
    state = initialize_dynamic_state(profile)

    state.energy = 0.50
    state.defensive_stability = 0.60
    state.chin_integrity = 0.70
    state.submission_danger = 0.80

    recovered = apply_between_round_recovery(state)

    assert recovered.energy > 0.50
    assert recovered.defensive_stability > 0.60
    assert recovered.chin_integrity > 0.70
    assert recovered.submission_danger < 0.80

    recovered.validate()


def test_cumulative_activity_updates() -> None:
    profile = make_profile()
    state = initialize_dynamic_state(profile)

    activity = make_activity(
        phase=FightPhase.GROUND,
        sig_attempted=9,
        ground_attempted=3,
        td_attempted=2,
        submission_attempts=1,
        knockdowns=1,
    )

    updated = update_dynamic_state(
        state=state,
        own_activity=activity,
        opponent_activity=make_activity(),
        profile=profile,
    )

    assert updated.cumulative_strike_activity == 12
    assert updated.cumulative_wrestling_activity == 3
    assert updated.knockdowns == 1
