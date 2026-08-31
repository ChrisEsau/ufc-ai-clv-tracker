from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.rate_families import (
    replay_suppression,
    replay_tendency,
    standing_spec,
    takedown_spec,
)


def _fights():
    rows = [
        ("2020-01-01", "f1", "A", "Alpha", "B", "Bravo", 3.0, 300.0),
        ("2020-01-01", "f2", "A", "Alpha", "C", "Charlie", 1.0, 150.0),
        ("2020-02-01", "f3", "A", "Alpha", "D", "Delta", 4.0, 300.0),
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
            "numerator",
            "exposure_seconds",
        ],
    )
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    return frame


def test_locked_rate_parameters_match_validated_study():
    config = FSRV3Config()
    assert np.isclose(config.takedown_tendency_prior_seconds, 468.48)
    assert np.isclose(config.takedown_tendency_initial_alpha, 0.2432)
    assert np.isclose(config.takedown_suppression_prior_shape, 8.5281)
    assert np.isclose(config.takedown_suppression_initial_alpha, 0.4574)
    assert np.isclose(config.standing_tendency_prior_seconds, 87.78)
    assert np.isclose(config.standing_tendency_initial_alpha, 0.0824)
    assert np.isclose(config.standing_suppression_prior_shape, 28.7138)
    assert np.isclose(config.standing_suppression_initial_alpha, 0.0863)


def test_tendency_same_event_updates_are_delayed_and_variance_is_enabled():
    spec = takedown_spec(FSRV3Config())
    history = replay_tendency(_fights(), spec)
    same_date = history[history["event_date"] == pd.Timestamp("2020-01-01")]
    assert len(same_date) == 2
    assert np.isclose(same_date.iloc[0]["pre_rating"], same_date.iloc[1]["pre_rating"])
    assert np.isclose(
        same_date.iloc[0]["pre_posterior_sd"],
        same_date.iloc[1]["pre_posterior_sd"],
    )
    assert history["sampling_enabled"].all()
    assert (history["variance_multiplier"] == 1.0).all()


def test_zero_exposure_does_not_update_rate_tendency():
    history = replay_tendency(_fights(), standing_spec(FSRV3Config()))
    zero = history[history["fight_id"] == "f4"].iloc[0]
    assert zero["denominator"] == 0.0
    assert np.isnan(zero["observed_rate_15m"])
    assert np.isclose(zero["pre_rating"], zero["post_rating"])
    assert np.isclose(zero["pre_posterior_sd"], zero["post_posterior_sd"])


def test_suppression_is_a_defender_multiplier_and_variance_is_enabled():
    spec = takedown_spec(FSRV3Config())
    tendency = replay_tendency(_fights(), spec)
    suppression = replay_suppression(tendency, spec)
    first = suppression[suppression["fight_id"] == "f1"].iloc[0]
    assert first["fighter_id"] == "B"
    assert first["opponent_id"] == "A"
    assert first["trait"] == "takedown_suppression"
    assert bool(first["sampling_enabled"]) is True
    assert first["variance_multiplier"] == 1.0
    assert first["population_multiplier"] > 0.0
