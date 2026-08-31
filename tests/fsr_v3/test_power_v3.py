from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.power import replay_power_from_frames


def _frames():
    keys = pd.DataFrame(
        [
            ("2019-06-01", "old", "A"),
            ("2020-01-04", "f1", "A"),
            ("2020-01-04", "f1", "B"),
            ("2020-01-04", "f2", "A"),
            ("2020-01-04", "f2", "C"),
            ("2020-02-01", "f3", "A"),
            ("2020-02-01", "f3", "D"),
        ],
        columns=["event_date", "fight_id", "fighter_id"],
    )
    observations = pd.DataFrame(
        [
            # Pre-cutoff training state.  Extra fighters stabilize the population fit.
            ("2018-01-01", "t0", "X", 100.0, 0.0),
            ("2018-02-01", "t1", "Y", 100.0, 1.0),
            ("2018-03-01", "t2", "Z", 100.0, 0.0),
            ("2019-06-01", "old", "A", 40.0, 1.0),
            # Same-date A evidence must not change A's other prefight row that day.
            ("2020-01-04", "f1", "A", 20.0, 1.0),
            ("2020-01-04", "f1", "B", 20.0, 0.0),
            ("2020-01-04", "f2", "A", 20.0, 1.0),
            ("2020-01-04", "f2", "C", 20.0, 0.0),
            ("2020-02-01", "f3", "A", 20.0, 0.0),
            ("2020-02-01", "f3", "D", 20.0, 0.0),
        ],
        columns=["date", "fight_id", "fighter_id", "sig_landed", "kd_scored"],
    )
    return keys, observations


def test_power_replay_matches_validated_mean_only_semantics():
    keys, observations = _frames()
    history = replay_power_from_frames(keys, observations, FSRV3Config())

    assert len(history) == len(keys)
    assert set(history["trait"]) == {"striking_power_v3"}
    assert not history["sampling_enabled"].any()
    assert np.allclose(history["variance_multiplier"], 0.0)

    # Pre-2020 publication stays neutral; no future-informed fighter state is backfilled.
    old = history[history["fight_id"] == "old"].iloc[0]
    assert old["pre_rating"] == 0.0
    assert old["post_rating"] == 0.0
    assert not bool(old["validated_regime"])

    # A true 2020+ cold start is exactly the population effect with sigma=.50 prior SD.
    debut = history[(history["fight_id"] == "f1") & (history["fighter_id"] == "B")].iloc[0]
    assert np.isclose(debut["pre_rating"], 0.0, atol=1e-12)
    assert np.isclose(debut["pre_posterior_sd"], 0.50, atol=0.02)


def test_power_same_date_updates_are_delayed():
    keys, observations = _frames()
    history = replay_power_from_frames(keys, observations, FSRV3Config())

    same_day = history[
        (history["event_date"] == pd.Timestamp("2020-01-04"))
        & (history["fighter_id"] == "A")
    ].sort_values("fight_id")
    assert len(same_day) == 2

    # Both prefight states are identical because neither fight can leak into the other.
    assert np.isclose(same_day.iloc[0]["pre_rating"], same_day.iloc[1]["pre_rating"])
    # Both post rows see the aggregate same-date evidence only after the date closes.
    assert np.isclose(same_day.iloc[0]["post_rating"], same_day.iloc[1]["post_rating"])
    assert same_day.iloc[0]["post_rating"] > same_day.iloc[0]["pre_rating"]

    next_fight = history[
        (history["fight_id"] == "f3") & (history["fighter_id"] == "A")
    ].iloc[0]
    assert np.isclose(next_fight["pre_rating"], same_day.iloc[0]["post_rating"])
