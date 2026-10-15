"""Tests for RFS Monte Carlo V1 contracts."""

import pytest

from pipeline.simulation.rfs_mc_v1.contracts import (
    DynamicFighterState,
    FighterSimulationProfile,
    MatchupSimulationRequest,
    ParameterEstimate,
    ProfileSource,
)


def make_profile(
    fighter_id: str,
    *,
    prior_fights: int = 3,
    scheduled_rounds: int = 3,
) -> FighterSimulationProfile:
    return FighterSimulationProfile(
        fighter_id=fighter_id,
        fighter_name=f"Fighter {fighter_id}",
        target_date="2026-08-05",
        weight_class="Lightweight",
        gender="male",
        scheduled_rounds=scheduled_rounds,
        prior_fight_count=prior_fights,
        valid_round_fight_count=prior_fights,
        is_low_experience=prior_fights < 3,
        parameters={
            "strike_pace": ParameterEstimate(
                value=10.0,
                source=ProfileSource.FIGHTER,
                effective_sample_size=float(prior_fights),
                uncertainty=0.25,
            )
        },
    )


def test_profile_marks_low_experience_consistently() -> None:
    profile = make_profile("red", prior_fights=2)
    assert profile.is_low_experience is True


def test_profile_rejects_inconsistent_low_experience_flag() -> None:
    with pytest.raises(ValueError, match="is_low_experience"):
        FighterSimulationProfile(
            fighter_id="red",
            fighter_name="Red Fighter",
            target_date="2026-08-05",
            weight_class="Lightweight",
            gender="male",
            scheduled_rounds=3,
            prior_fight_count=2,
            valid_round_fight_count=2,
            is_low_experience=False,
            parameters={
                "strike_pace": ParameterEstimate(
                    value=10.0,
                    source=ProfileSource.FIGHTER,
                    effective_sample_size=2.0,
                    uncertainty=0.5,
                )
            },
        )


def test_dynamic_state_default_is_valid() -> None:
    state = DynamicFighterState()
    state.validate()


def test_dynamic_state_rejects_invalid_bounds() -> None:
    state = DynamicFighterState(energy=1.1)

    with pytest.raises(ValueError, match="energy"):
        state.validate()


def test_matchup_request_requires_distinct_fighters() -> None:
    profile = make_profile("same")

    with pytest.raises(ValueError, match="must be different"):
        MatchupSimulationRequest(
            red_profile=profile,
            blue_profile=profile,
            path_count=100,
            seed=7,
        )


def test_matchup_request_requires_matching_rounds() -> None:
    with pytest.raises(ValueError, match="scheduled rounds"):
        MatchupSimulationRequest(
            red_profile=make_profile("red", scheduled_rounds=3),
            blue_profile=make_profile("blue", scheduled_rounds=5),
            path_count=100,
            seed=7,
        )
