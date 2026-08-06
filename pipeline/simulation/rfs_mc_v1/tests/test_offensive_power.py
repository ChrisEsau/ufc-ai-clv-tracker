"""Tests for leakage-safe offensive power profiles."""

import pandas as pd
import pytest

from pipeline.simulation.rfs_mc_v1.contracts import (
    FighterSimulationProfile,
    ParameterEstimate,
    ProfileSource,
)
from pipeline.simulation.rfs_mc_v1.offensive_power import (
    augment_profile_with_offensive_power,
    build_offensive_power_estimates,
)


def make_round_stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fight_id": "old-1",
                "event_date": "2025-01-01",
                "fighter_id": "power",
                "round": 1,
                "kd": 2,
                "sig_str_landed": 10,
                "head_landed": 8,
                "head_attempted": 12,
                "ground_landed": 2,
                "ground_attempted": 3,
            },
            {
                "fight_id": "old-2",
                "event_date": "2025-06-01",
                "fighter_id": "power",
                "round": 1,
                "kd": 1,
                "sig_str_landed": 8,
                "head_landed": 6,
                "head_attempted": 10,
                "ground_landed": 0,
                "ground_attempted": 0,
            },
            {
                "fight_id": "old-3",
                "event_date": "2025-07-01",
                "fighter_id": "volume",
                "round": 1,
                "kd": 0,
                "sig_str_landed": 20,
                "head_landed": 8,
                "head_attempted": 20,
                "ground_landed": 1,
                "ground_attempted": 4,
            },
            {
                "fight_id": "target",
                "event_date": "2026-01-01",
                "fighter_id": "power",
                "round": 1,
                "kd": 100,
                "sig_str_landed": 100,
                "head_landed": 100,
                "head_attempted": 100,
                "ground_landed": 100,
                "ground_attempted": 100,
            },
        ]
    )


def make_profile() -> FighterSimulationProfile:
    return FighterSimulationProfile(
        fighter_id="power",
        fighter_name="Power Fighter",
        target_date="2026-01-01",
        weight_class="Welterweight",
        gender="male",
        scheduled_rounds=3,
        prior_fight_count=2,
        valid_round_fight_count=2,
        is_low_experience=True,
        parameters={
            "placeholder": ParameterEstimate(
                value=0.0,
                source=ProfileSource.GLOBAL,
                effective_sample_size=2.0,
                uncertainty=0.7,
            )
        },
    )


def test_power_estimates_exclude_target_date() -> None:
    estimates = build_offensive_power_estimates(
        make_round_stats(),
        fighter_id="power",
        target_date="2026-01-01",
    )

    assert estimates["offensive_kd_per_fight"].value < 10
    assert estimates["round1_kd_per_fight"].value < 10


def test_knockdown_efficiency_is_positive() -> None:
    estimates = build_offensive_power_estimates(
        make_round_stats(),
        fighter_id="power",
        target_date="2026-01-01",
    )

    assert (
        estimates["offensive_kd_per_sig_landed"].value
        > 0
    )
    assert estimates["round1_kd_per_fight"].value > 0


def test_sparse_profile_is_shrunk_and_marked_global() -> None:
    estimates = build_offensive_power_estimates(
        make_round_stats(),
        fighter_id="power",
        target_date="2026-01-01",
    )

    assert all(
        estimate.source is ProfileSource.GLOBAL
        for estimate in estimates.values()
    )


def test_profile_augmentation_preserves_existing_parameters() -> None:
    profile = augment_profile_with_offensive_power(
        make_profile(),
        make_round_stats(),
    )

    assert "placeholder" in profile.parameters
    assert "offensive_kd_per_sig_landed" in profile.parameters
    assert "round1_kd_per_fight" in profile.parameters


def test_missing_prior_history_raises() -> None:
    with pytest.raises(ValueError, match="No prior round stats"):
        build_offensive_power_estimates(
            make_round_stats(),
            fighter_id="unknown",
            target_date="2026-01-01",
        )
