from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.paired_effectiveness import (
    replay_paired_effectiveness,
    standing_effectiveness_spec,
    takedown_effectiveness_spec,
)


def _fights():
    rows = [
        ("2020-01-01", "f1", "A", "Alpha", "B", "Bravo", 5.0, 10.0),
        ("2020-01-01", "f2", "A", "Alpha", "C", "Charlie", 2.0, 5.0),
        ("2020-02-01", "f3", "A", "Alpha", "D", "Delta", 7.0, 10.0),
        ("2020-03-01", "f4", "A", "Alpha", "E", "Echo", 0.0, 0.0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "event_date",
            "fight_id",
            "fighter_id",
            "fighter_name",
            "opponent_id",
            "opponent_name",
            "landed",
            "attempted",
        ],
    )
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    return frame


def test_locked_paired_parameters_match_validated_study():
    config = FSRV3Config()
    assert np.isclose(config.takedown_effectiveness_rho, 0.12)
    assert np.isclose(config.takedown_effectiveness_sigma_offense, 0.35)
    assert np.isclose(config.takedown_effectiveness_sigma_defense, 0.50)
    assert np.isclose(config.standing_effectiveness_rho, 0.035)
    assert np.isclose(config.standing_effectiveness_sigma_offense, 0.30)
    assert np.isclose(config.standing_effectiveness_sigma_defense, 0.30)


def test_takedown_paired_effectiveness_emits_offense_and_defense_mean_only():
    spec = takedown_effectiveness_spec(FSRV3Config())
    history = replay_paired_effectiveness(_fights(), spec)
    assert set(history["trait"]) == {"takedown_offense", "takedown_defense"}
    assert not history["sampling_enabled"].any()
    assert (history["variance_multiplier"] == 0.0).all()
    assert history["population_baseline"].between(0.0, 1.0).all()


def test_standing_offense_same_event_prefight_state_is_delayed():
    spec = standing_effectiveness_spec(FSRV3Config())
    history = replay_paired_effectiveness(_fights(), spec)
    offense = history[
        (history["trait"] == "standing_striking_offense")
        & (history["fighter_id"] == "A")
        & (history["event_date"] == pd.Timestamp("2020-01-01"))
    ].sort_values("fight_id")
    assert len(offense) == 2
    assert np.isclose(offense.iloc[0]["pre_rating"], offense.iloc[1]["pre_rating"])
    assert np.isclose(
        offense.iloc[0]["pre_posterior_sd"],
        offense.iloc[1]["pre_posterior_sd"],
    )


def test_zero_attempts_do_not_update_effectiveness():
    history = replay_paired_effectiveness(
        _fights(),
        takedown_effectiveness_spec(FSRV3Config()),
    )
    row = history[
        (history["fight_id"] == "f4")
        & (history["trait"] == "takedown_offense")
    ].iloc[0]
    assert row["attempted"] == 0.0
    assert np.isclose(row["pre_rating"], row["post_rating"])
    assert np.isclose(row["pre_posterior_sd"], row["post_posterior_sd"])
