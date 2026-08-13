from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.phase7h_submission_conversion_calibration import calibration_for_intercept, historical_methods
import pandas as pd


def test_candidate_changes_only_submission_finish_intercept():
    candidate = calibration_for_intercept(-1.0)
    assert candidate.section("submission_finish")["intercept"] == -1.0
    for section, values in DEFAULT_CALIBRATION.values.items():
        for key, value in values.items():
            if (section, key) != ("submission_finish", "intercept"):
                assert candidate.section(section)[key] == value
    attempts = candidate.section("submission_attempts")
    assert attempts["base_30s"] == .045 and attempts["bottom_multiplier"] == 1.0
    finish = candidate.section("submission_finish")
    assert finish["top_position_bonus"] == finish["bottom_position_bonus"] == 0.0
    assert candidate.section("knockdown")["midpoint_impact_ratio"] == 36
    assert candidate.section("finish")["midpoint_impact_ratio"] == 36


def test_historical_methods_are_end_to_end_fight_shares():
    cohort = pd.DataFrame({"method": ["Submission", "Decision - Unanimous", "KO/TKO", "Submission"]})
    assert historical_methods(cohort) == {"KO_TKO": .25, "SUB": .5, "DEC": .25}
