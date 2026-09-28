"""Tests for leakage-safe historical matchup loading."""

import pandas as pd
import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.historical_matchup_loader import (
    HistoricalMatchupLoadError,
    load_historical_matchup,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fight_id": ["f1", "f1"],
            "fighter_id": ["red_id", "blue_id"],
            "fighter_name": ["Red Fighter", "Blue Fighter"],
            "corner": ["red", "blue"],
            "date": ["2025-01-01", "2025-01-01"],
            "event_name": ["UFC Test", "UFC Test"],
            "division": ["Lightweight", "Lightweight"],
            "total_rounds": [3, 3],
            "rfs_traj_prior_fight_count": [4, 6],
            "rfs_traj_ewm_sig_attempt_slope": [0.2, -0.1],
            "rfs_phase_base_ewm_distance_attempt_share": [0.7, 0.5],
            "rfs_dynamic_response_ewm_adversity_round_count": [1.2, 0.8],
            # Must never enter the profile:
            "rfs_phase_base_fight_distance_attempt_share": [0.9, 0.4],
        }
    )


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fight_id": ["f1"],
            "winner_id": ["red_id"],
            "method": ["Decision - Unanimous"],
            "finish_round": [3],
        }
    )


def test_loads_two_prefight_profiles_and_separate_outcome() -> None:
    matchup = load_historical_matchup(
        _history(),
        _outcomes(),
        "f1",
    )

    assert matchup.red.fighter_id == "red_id"
    assert matchup.blue.fighter_id == "blue_id"

    assert matchup.red.prior_fight_count == 4
    assert matchup.blue.prior_fight_count == 6

    assert matchup.actual.winner_id == "red_id"
    assert matchup.actual.method == "Decision - Unanimous"

    assert (
        "rfs_phase_base_ewm_distance_attempt_share"
        in matchup.red.features
    )

    assert not any(
        column.startswith("rfs_phase_base_fight_")
        for column in matchup.red.features
    )


def test_enforces_three_prior_fight_cohort() -> None:
    history = _history()
    history.loc[
        history["fighter_id"] == "red_id",
        "rfs_traj_prior_fight_count",
    ] = 2

    with pytest.raises(
        HistoricalMatchupLoadError,
        match="minimum is 3",
    ):
        load_historical_matchup(
            history,
            _outcomes(),
            "f1",
        )


def test_rejects_missing_fighter_row() -> None:
    history = _history().iloc[[0]].copy()

    with pytest.raises(
        HistoricalMatchupLoadError,
        match="exactly two fighter rows",
    ):
        load_historical_matchup(
            history,
            _outcomes(),
            "f1",
        )
