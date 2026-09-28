from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.feature_builder import (
    direct_feature_columns_v3,
    build_sampled_fight_feature_rows_v3,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    initialize_fighter_path_traits,
)


def _record(fid, name, **overrides):
    row = {
        "fighter_id": fid,
        "fighter_name": name,
        "standing_striking_tendency": 180.0,
        "standing_striking_suppression": 1.0,
        "standing_striking_offense": 0.2,
        "standing_striking_defense": 0.1,
        "standing_accuracy_baseline": 0.45,
        "takedown_tendency": 6.0,
        "takedown_suppression": 1.0,
        "takedown_offense": 0.2,
        "takedown_defense": 0.1,
        "takedown_completion_baseline": 0.35,
        "escape_offense": 0.1,
        "escape_defense": 0.2,
        "escape_population_mean_seconds": 45.0,
        "ground_striking_tendency": 40.0,
        "ground_striking_suppression": 1.0,
        "ground_striking_offense": 0.2,
        "ground_accuracy_baseline": 0.70,
        "ground_striking_burst_baseline": 2.2,
        "ground_striking_population_slope_15m": 35.0,
        "submission_tendency": 0.0007,
        "submission_suppression": 1.0,
        "submission_offense": 0.1,
        "submission_defense": 0.1,
    }
    row.update(overrides)
    return row


def _traits(record):
    return initialize_fighter_path_traits(
        record,
        None,
        rng=np.random.default_rng(1),
        sample_epistemic=False,
    )


def _master():
    return pd.Series(
        {
            "fight_id": "fight1",
            "event_date": pd.Timestamp("2026-01-01"),
            "total_rounds": 3,
            "r_id": "R",
            "b_id": "B",
            "r_dob": "1990-01-01",
            "b_dob": "1992-01-01",
        }
    )


def test_v3_direct_feature_schema_excludes_rejected_ground_defense():
    cols = direct_feature_columns_v3()
    assert "self_ground_striking_defense" not in cols
    assert "opp_ground_striking_defense" not in cols
    assert "ground_burst_attempts" in cols
    assert "self_ground_striking_population_slope_15m" in cols


def test_sampled_feature_rows_use_v3_multiplicative_semantics():
    red = _record(
        "R", "Red",
        standing_striking_tendency=200.0,
        takedown_tendency=8.0,
        ground_striking_tendency=50.0,
    )
    blue = _record(
        "B", "Blue",
        standing_striking_suppression=0.5,
        takedown_suppression=0.75,
        ground_striking_suppression=0.4,
    )
    frame = build_sampled_fight_feature_rows_v3(
        _master(),
        red_record=red,
        blue_record=blue,
        red_traits=_traits(red),
        blue_traits=_traits(blue),
    )
    r = frame[frame["side"] == "red"].iloc[0]
    assert r["effective_standing_rate"] == 100.0
    assert r["effective_td_rate"] == 6.0
    assert r["effective_ground_rate"] == 20.0
    assert r["ground_burst_attempts"] == 2.2


def test_sampled_trait_value_reaches_direct_feature_columns():
    red = _record("R", "Red")
    blue = _record("B", "Blue")
    red_traits = _traits(red)
    # Construct a second immutable state representing a path-level sampled draw.
    sampled_red = type(red_traits)(
        fighter_id=red_traits.fighter_id,
        values={**dict(red_traits.values), "takedown_tendency": 12.0},
        epistemic_sampled=True,
    )
    frame = build_sampled_fight_feature_rows_v3(
        _master(),
        red_record=red,
        blue_record=blue,
        red_traits=sampled_red,
        blue_traits=_traits(blue),
    )
    r = frame[frame["side"] == "red"].iloc[0]
    assert r["self_takedown_tendency"] == 12.0
    assert r["effective_td_rate"] == 12.0
