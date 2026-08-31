from __future__ import annotations

from math import exp

import numpy as np
import pandas as pd
import pytest

from pipeline.simulation.event_clock_mc_v2.feature_builder import (
    build_sampled_fight_feature_rows_v3,
    direct_feature_columns_v3,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    initialize_fighter_path_traits,
)
from pipeline.simulation.event_clock_mc_v2.inference import (
    _apply_distance_reach_translation,
)
from pipeline.simulation.event_clock_mc_v2.reach_translation import (
    DISTANCE_REACH_EDGE_CAP_INCHES,
    DISTANCE_REACH_LOG_RATE_PER_INCH,
    _measure_inches,
    directional_reach_inputs,
    distance_reach_multiplier,
)


def _record(fid: str, name: str) -> dict:
    return {
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


def _traits(record: dict):
    return initialize_fighter_path_traits(
        record,
        None,
        rng=np.random.default_rng(1),
        sample_epistemic=False,
    )


def test_canonical_numeric_master_reach_is_converted_from_cm_to_inches():
    assert _measure_inches(190.50) == pytest.approx(75.0)
    assert _measure_inches("182.88") == pytest.approx(72.0)
    assert _measure_inches("75 in") == pytest.approx(75.0)
    assert _measure_inches('75"') == pytest.approx(75.0)


def test_reach_translation_is_directional_and_reciprocal_inside_cap():
    # Canonical master reach is centimeters: 76 in vs 72 in.
    master = {"r_reach": 193.04, "b_reach": 182.88}
    red = directional_reach_inputs(master, "red")
    blue = directional_reach_inputs(master, "blue")

    assert red["self_reach_inches"] == pytest.approx(76.0)
    assert blue["self_reach_inches"] == pytest.approx(72.0)
    assert red["reach_edge_inches"] == pytest.approx(4.0)
    assert blue["reach_edge_inches"] == pytest.approx(-4.0)
    assert red["distance_reach_multiplier"] == pytest.approx(
        exp(DISTANCE_REACH_LOG_RATE_PER_INCH * 4.0)
    )
    assert red["distance_reach_multiplier"] * blue["distance_reach_multiplier"] == pytest.approx(1.0)


def test_reach_translation_caps_extreme_edges_at_six_inches():
    clipped, multiplier = distance_reach_multiplier(20.0)
    assert clipped == DISTANCE_REACH_EDGE_CAP_INCHES
    assert multiplier == pytest.approx(
        exp(DISTANCE_REACH_LOG_RATE_PER_INCH * DISTANCE_REACH_EDGE_CAP_INCHES)
    )


def test_missing_reach_is_neutral():
    red = directional_reach_inputs({"r_reach": None, "b_reach": 182.88}, "red")
    assert np.isnan(red["reach_edge_inches"])
    assert np.isnan(red["reach_edge_capped_inches"])
    assert red["distance_reach_multiplier"] == 1.0


def test_sampled_feature_builder_carries_reach_without_changing_fsr_schema():
    red = _record("R", "Red")
    blue = _record("B", "Blue")
    master = pd.Series(
        {
            "fight_id": "fight1",
            "event_date": pd.Timestamp("2026-01-01"),
            "total_rounds": 3,
            "r_id": "R",
            "b_id": "B",
            "r_dob": "1990-01-01",
            "b_dob": "1992-01-01",
            # Canonical master centimeters: 78 in vs 72 in.
            "r_reach": 198.12,
            "b_reach": 182.88,
        }
    )
    frame = build_sampled_fight_feature_rows_v3(
        master,
        red_record=red,
        blue_record=blue,
        red_traits=_traits(red),
        blue_traits=_traits(blue),
    )
    red_row = frame.loc[frame["side"].eq("red")].iloc[0]
    blue_row = frame.loc[frame["side"].eq("blue")].iloc[0]

    assert red_row["reach_edge_inches"] == pytest.approx(6.0)
    assert blue_row["reach_edge_inches"] == pytest.approx(-6.0)
    assert red_row["reach_edge_capped_inches"] == pytest.approx(6.0)
    assert "distance_reach_multiplier" not in direct_feature_columns_v3()
    assert "reach_edge_inches" not in direct_feature_columns_v3()


def test_distance_translation_preserves_accuracy_and_changes_stage9_rate_by_distance_share():
    multiplier = exp(DISTANCE_REACH_LOG_RATE_PER_INCH * 4.0)
    frame = pd.DataFrame(
        {
            "pred_distance_attempted": [80.0],
            "pred_distance_landed": [40.0],
            "pred_clinch_attempted": [20.0],
            "pred_clinch_landed": [8.0],
            "distance_reach_multiplier": [multiplier],
        }
    )
    translated = _apply_distance_reach_translation(frame)

    assert translated.loc[0, "pred_distance_attempted"] == pytest.approx(80.0 * multiplier)
    assert translated.loc[0, "pred_distance_landed"] == pytest.approx(40.0 * multiplier)
    assert (
        translated.loc[0, "pred_distance_landed"]
        / translated.loc[0, "pred_distance_attempted"]
    ) == pytest.approx(0.5)
    expected_standing_multiplier = (80.0 * multiplier + 20.0) / 100.0
    assert translated.loc[0, "standing_reach_rate_multiplier"] == pytest.approx(
        expected_standing_multiplier
    )
