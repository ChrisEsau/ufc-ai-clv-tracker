import pandas as pd

from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.phase7i_strike_exposure_audit import (
    historical_strikes,
)


def test_total_and_significant_fields_remain_separate_with_elapsed_time():
    row = {"finish_round": 2, "match_time_sec": 420}
    for side in ("r", "b"):
        row.update(
            {
                f"{side}_total_str_atmpted": 100,
                f"{side}_total_str_landed": 50,
                f"{side}_sig_str_atmpted": 60,
                f"{side}_sig_str_landed": 30,
                f"{side}_clinch_atmpted": 10,
                f"{side}_clinch_landed": 5,
                f"{side}_ground_atmpted": 5,
                f"{side}_ground_landed": 2,
            }
        )
    result = historical_strikes(pd.DataFrame([row]))
    assert result["exposure_seconds"] == 420
    assert result["total"]["attempts_per_fight"] == 200
    assert result["significant"]["attempts_per_fight"] == 120
    assert result["significant_by_phase"]["distance"]["attempts_per_fight"] == 90


def test_phase7i_keeps_all_committed_calibration_locks():
    attempts = DEFAULT_CALIBRATION.section("submission_attempts")
    submission = DEFAULT_CALIBRATION.section("submission_finish")
    assert attempts["base_30s"] == 0.045
    assert attempts["bottom_multiplier"] == 1.0
    assert submission["top_position_bonus"] == 0.0
    assert submission["bottom_position_bonus"] == 0.0
    assert submission["intercept"] == -0.60
    assert DEFAULT_CALIBRATION.section("knockdown")["midpoint_impact_ratio"] == 36.0
    assert DEFAULT_CALIBRATION.section("finish")["midpoint_impact_ratio"] == 36.0
