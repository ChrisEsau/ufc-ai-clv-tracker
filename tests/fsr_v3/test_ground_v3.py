from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.ground import (
    replay_ground_suppression,
    replay_ground_tendency,
)
from pipeline.fsr_v3.replay.ground_effectiveness import replay_ground_effectiveness


def _fighter_fights():
    rows = [
        # Two same-date appearances for A verify delayed updates.
        ("2020-01-01", "f1", "A", "Alpha", "B", "Bravo", 5, 7, 60),
        ("2020-01-01", "f2", "A", "Alpha", "C", "Charlie", 2, 4, 30),
        ("2020-02-01", "f3", "A", "Alpha", "D", "Delta", 6, 8, 90),
        ("2020-03-01", "f4", "A", "Alpha", "E", "Echo", 0, 0, 0),
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
            "ground_landed",
            "ground_attempted",
            "own_control_seconds",
        ],
    )
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    return frame


def test_ground_tendency_same_event_updates_are_delayed():
    history = replay_ground_tendency(_fighter_fights(), FSRV3Config())
    same_date = history[history["event_date"] == pd.Timestamp("2020-01-01")]
    assert len(same_date) == 2
    assert np.isclose(same_date.iloc[0]["pre_rating"], same_date.iloc[1]["pre_rating"])
    assert np.isclose(same_date.iloc[0]["pre_posterior_sd"], same_date.iloc[1]["pre_posterior_sd"])


def test_zero_own_control_is_not_ground_tendency_evidence():
    history = replay_ground_tendency(_fighter_fights(), FSRV3Config())
    zero = history[history["fight_id"] == "f4"].iloc[0]
    assert zero["denominator"] == 0.0
    assert np.isnan(zero["observed_rate_15m_own_control"])
    assert np.isclose(zero["pre_rating"], zero["post_rating"])
    assert np.isclose(zero["pre_posterior_sd"], zero["post_posterior_sd"])


def test_ground_suppression_is_defender_trait_and_never_sampled():
    tendency = replay_ground_tendency(_fighter_fights(), FSRV3Config())
    suppression = replay_ground_suppression(tendency, FSRV3Config())
    first = suppression[suppression["fight_id"] == "f1"].iloc[0]
    assert first["fighter_id"] == "B"
    assert first["opponent_id"] == "A"
    assert first["trait"] == "ground_striking_suppression"
    assert bool(first["sampling_enabled"]) is False
    assert first["variance_multiplier"] == 0.0


def test_ground_effectiveness_is_attacker_only_and_never_sampled():
    history = replay_ground_effectiveness(_fighter_fights(), FSRV3Config())
    assert set(history["trait"]) == {"ground_striking_offense"}
    assert "ground_striking_defense" not in set(history["trait"])
    assert not history["sampling_enabled"].any()
    assert (history["variance_multiplier"] == 0.0).all()
    zero = history[history["fight_id"] == "f4"].iloc[0]
    assert np.isclose(zero["pre_rating"], zero["post_rating"])
