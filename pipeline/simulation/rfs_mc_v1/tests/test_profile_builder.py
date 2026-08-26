"""Tests for the RFS Monte Carlo V1 profile builder."""

import pandas as pd
import pytest

from pipeline.simulation.rfs_mc_v1.contracts import ProfileSource
from pipeline.simulation.rfs_mc_v1.profile_builder import (
    ProfileBuilderError,
    build_profile_from_history,
    select_latest_prior_row,
)


def make_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fight_id": "fight-1",
                "fighter_id": "fighter-a",
                "fighter_name": "Fighter A",
                "date": "2025-01-01",
                "rfs_traj_prior_fight_count": 1,
                "rfs_traj_prior_valid_trajectory_count": 1,
                "rfs_traj_ewm_sig_attempt_slope": -1.0,
            },
            {
                "fight_id": "fight-2",
                "fighter_id": "fighter-a",
                "fighter_name": "Fighter A",
                "date": "2025-06-01",
                "rfs_traj_prior_fight_count": 3,
                "rfs_traj_prior_valid_trajectory_count": 3,
                "rfs_traj_ewm_sig_attempt_slope": 0.5,
            },
            {
                "fight_id": "target-fight",
                "fighter_id": "fighter-a",
                "fighter_name": "Fighter A",
                "date": "2026-01-01",
                "rfs_traj_prior_fight_count": 4,
                "rfs_traj_prior_valid_trajectory_count": 4,
                "rfs_traj_ewm_sig_attempt_slope": 99.0,
            },
        ]
    )


def test_select_latest_prior_row_excludes_target_date() -> None:
    row = select_latest_prior_row(
        make_history(),
        fighter_id="fighter-a",
        target_date="2026-01-01",
    )

    assert row["fight_id"] == "fight-2"
    assert row["rfs_traj_ewm_sig_attempt_slope"] == 0.5


def test_build_profile_from_history_maps_parameter() -> None:
    profile = build_profile_from_history(
        make_history(),
        fighter_id="fighter-a",
        target_date="2026-01-01",
        scheduled_rounds=3,
        weight_class="Lightweight",
        gender="male",
        parameter_map={
            "sig_attempt_slope": "rfs_traj_ewm_sig_attempt_slope",
        },
        prior_fight_count_column="rfs_traj_prior_fight_count",
        prior_valid_count_column=(
            "rfs_traj_prior_valid_trajectory_count"
        ),
    )

    estimate = profile.parameters["sig_attempt_slope"]

    assert profile.prior_fight_count == 3
    assert profile.is_low_experience is False
    assert estimate.value == 0.5
    assert estimate.source is ProfileSource.FIGHTER
    assert estimate.effective_sample_size == 3.0


def test_sparse_profile_uses_global_fallback_provenance() -> None:
    history = make_history().iloc[[0]].copy()

    profile = build_profile_from_history(
        history,
        fighter_id="fighter-a",
        target_date="2025-02-01",
        scheduled_rounds=3,
        weight_class="Lightweight",
        gender="male",
        parameter_map={
            "sig_attempt_slope": "rfs_traj_ewm_sig_attempt_slope",
        },
        prior_fight_count_column="rfs_traj_prior_fight_count",
        prior_valid_count_column=(
            "rfs_traj_prior_valid_trajectory_count"
        ),
    )

    assert profile.is_low_experience is True
    assert (
        profile.parameters["sig_attempt_slope"].source
        is ProfileSource.GLOBAL
    )


def test_no_prior_state_raises() -> None:
    with pytest.raises(ProfileBuilderError, match="No prior state"):
        select_latest_prior_row(
            make_history(),
            fighter_id="fighter-a",
            target_date="2024-01-01",
        )


def test_unknown_fighter_raises() -> None:
    with pytest.raises(ProfileBuilderError, match="No history rows"):
        select_latest_prior_row(
            make_history(),
            fighter_id="missing",
            target_date="2026-01-01",
        )
