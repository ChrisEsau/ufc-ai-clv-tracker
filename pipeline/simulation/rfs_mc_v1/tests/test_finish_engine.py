"""Tests for competing probabilistic finish hazards."""

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import (
    DynamicFighterState,
    FightPhase,
    FighterSimulationProfile,
    ParameterEstimate,
    ProfileSource,
)
from pipeline.simulation.rfs_mc_v1.finish_engine import (
    FinishMethod,
    calculate_finish_hazards,
    sample_competing_finish,
)
from pipeline.simulation.rfs_mc_v1.segment_engine import (
    SegmentActivity,
)


def make_profile(fighter_id: str) -> FighterSimulationProfile:
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
            "finish_test_placeholder": ParameterEstimate(
                value=0.0,
                source=ProfileSource.FIGHTER,
                effective_sample_size=4.0,
                uncertainty=0.5,
            )
        },
    )


def make_activity(
    *,
    landed: int = 0,
    ground_landed: int = 0,
    control_seconds: int = 0,
    submissions: int = 0,
    knockdowns: int = 0,
) -> SegmentActivity:
    return SegmentActivity(
        phase=FightPhase.GROUND,
        sig_str_attempted=max(landed, 0),
        sig_str_landed=landed,
        td_attempted=0,
        td_landed=0,
        control_seconds=control_seconds,
        ground_str_attempted=max(ground_landed, 0),
        ground_str_landed=ground_landed,
        submission_attempts=submissions,
        knockdowns=knockdowns,
    )


def test_damage_and_knockdown_raise_ko_hazard() -> None:
    profile = make_profile("blue")

    healthy = DynamicFighterState()
    damaged = DynamicFighterState(
        energy=0.35,
        head_damage=0.80,
        chin_integrity=0.35,
        defensive_stability=0.45,
    )

    low = calculate_finish_hazards(
        defender_state=healthy,
        attacker_activity=make_activity(landed=1),
        defender_profile=profile,
    )
    high = calculate_finish_hazards(
        defender_state=damaged,
        attacker_activity=make_activity(
            landed=7,
            ground_landed=3,
            knockdowns=1,
        ),
        defender_profile=profile,
    )

    assert high.ko_tko > low.ko_tko


def test_control_and_submission_attempt_raise_submission_hazard() -> None:
    profile = make_profile("blue")

    safe = DynamicFighterState()
    endangered = DynamicFighterState(
        energy=0.40,
        defensive_stability=0.50,
        submission_danger=0.85,
    )

    low = calculate_finish_hazards(
        defender_state=safe,
        attacker_activity=make_activity(),
        defender_profile=profile,
    )
    high = calculate_finish_hazards(
        defender_state=endangered,
        attacker_activity=make_activity(
            control_seconds=25,
            submissions=2,
        ),
        defender_profile=profile,
    )

    assert high.submission > low.submission


def test_finish_sampling_is_reproducible() -> None:
    red_profile = make_profile("red")
    blue_profile = make_profile("blue")

    red_state = DynamicFighterState()
    blue_state = DynamicFighterState(
        energy=0.20,
        head_damage=1.20,
        chin_integrity=0.20,
        defensive_stability=0.25,
    )

    red_activity = make_activity(
        landed=9,
        ground_landed=4,
        knockdowns=1,
    )
    blue_activity = make_activity()

    first = sample_competing_finish(
        red_state=red_state,
        blue_state=blue_state,
        red_activity=red_activity,
        blue_activity=blue_activity,
        red_profile=red_profile,
        blue_profile=blue_profile,
        rng=np.random.default_rng(42),
    )
    second = sample_competing_finish(
        red_state=red_state,
        blue_state=blue_state,
        red_activity=red_activity,
        blue_activity=blue_activity,
        red_profile=red_profile,
        blue_profile=blue_profile,
        rng=np.random.default_rng(42),
    )

    assert first == second


def test_finish_results_are_contract_valid() -> None:
    red_profile = make_profile("red")
    blue_profile = make_profile("blue")

    result = sample_competing_finish(
        red_state=DynamicFighterState(),
        blue_state=DynamicFighterState(
            energy=0.10,
            head_damage=2.0,
            chin_integrity=0.10,
            defensive_stability=0.10,
            submission_danger=0.90,
        ),
        red_activity=make_activity(
            landed=10,
            ground_landed=5,
            control_seconds=25,
            submissions=2,
            knockdowns=1,
        ),
        blue_activity=make_activity(),
        red_profile=red_profile,
        blue_profile=blue_profile,
        rng=np.random.default_rng(1),
    )

    if result.finished:
        assert result.winner in {"red", "blue"}
        assert result.loser in {"red", "blue"}
        assert result.winner != result.loser
        assert result.method in {
            FinishMethod.KO_TKO,
            FinishMethod.SUBMISSION,
        }


def test_no_hard_threshold_guarantees_finish() -> None:
    red_profile = make_profile("red")
    blue_profile = make_profile("blue")

    heavily_damaged = DynamicFighterState(
        energy=0.0,
        head_damage=5.0,
        chin_integrity=0.0,
        defensive_stability=0.0,
        submission_danger=1.0,
    )

    activity = make_activity(
        landed=15,
        ground_landed=10,
        control_seconds=30,
        submissions=3,
        knockdowns=1,
    )

    unfinished_count = 0

    for seed in range(200):
        result = sample_competing_finish(
            red_state=DynamicFighterState(),
            blue_state=heavily_damaged,
            red_activity=activity,
            blue_activity=make_activity(),
            red_profile=red_profile,
            blue_profile=blue_profile,
            rng=np.random.default_rng(seed),
        )

        unfinished_count += int(not result.finished)

    assert unfinished_count > 0
