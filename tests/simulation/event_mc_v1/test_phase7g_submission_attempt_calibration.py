from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.phase7g_submission_attempt_calibration import calibration_for_base, historical_anchors
import pandas as pd


def test_candidate_changes_only_submission_attempt_base():
    candidate = calibration_for_base(.060)
    assert candidate.section("submission_attempts")["base_30s"] == .060
    for section, values in DEFAULT_CALIBRATION.values.items():
        for key, value in values.items():
            if (section, key) != ("submission_attempts", "base_30s"):
                assert candidate.section(section)[key] == value
    assert candidate.section("submission_attempts")["bottom_multiplier"] == 1.0
    finish = candidate.section("submission_finish")
    assert finish["top_position_bonus"] == finish["bottom_position_bonus"] == 0.0
    assert finish["intercept"] == -0.60


def test_split_historical_anchors_use_elapsed_exposure():
    cohort = pd.DataFrame({"r_sub_att": [1, 0], "b_sub_att": [0, 1], "finish_round": [2, 3], "match_time_sec": [420, 750]})
    result = historical_anchors(cohort)
    assert result["attempts_per_fight"] == 1
    assert result["attempts_per_15min"] == 2 / 1170 * 900
    assert result["fights_with_attempt_share"] == 1
