from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.phase7j_strike_attempt_calibration import (
    StrikeCalibrationSink,
    calibration_for_rates,
)


def test_candidate_changes_only_authorized_strike_attempt_rates():
    candidate = calibration_for_rates(6.0, 2.8)

    for section, values in DEFAULT_CALIBRATION.values.items():
        for key, value in values.items():
            if (section, key) == ("distance", "strike_attempts_per_30s"):
                assert candidate.section(section)[key] == 6.0
            elif (section, key) == ("clinch", "strike_attempts_per_30s"):
                assert candidate.section(section)[key] == 2.8
            else:
                assert candidate.section(section)[key] == value


def test_phase7j_keeps_calibration_locks_and_sparse_sink_is_safe():
    candidate = calibration_for_rates(5.5, 2.0)

    assert candidate.section("ground")["strike_attempts_per_30s"] == 1.6
    assert candidate.section("distance")["strike_accuracy"] == 0.40
    assert candidate.section("clinch")["strike_accuracy"] == 0.68
    assert candidate.section("ground")["strike_accuracy"] == 0.70
    assert candidate.section("submission_attempts")["base_30s"] == 0.045
    assert candidate.section("submission_attempts")["bottom_multiplier"] == 1.0
    assert candidate.section("submission_finish")["intercept"] == -0.60
    assert candidate.section("submission_finish")["top_position_bonus"] == 0.0
    assert candidate.section("submission_finish")["bottom_position_bonus"] == 0.0
    assert candidate.section("knockdown")["midpoint_impact_ratio"] == 36.0
    assert candidate.section("finish")["midpoint_impact_ratio"] == 36.0
    assert StrikeCalibrationSink().finalize()["attempts"] == {}


def test_promoted_global_strike_rates_are_the_only_new_defaults():
    assert DEFAULT_CALIBRATION.section("distance")["strike_attempts_per_30s"] == 6.0
    assert DEFAULT_CALIBRATION.section("clinch")["strike_attempts_per_30s"] == 3.6
    assert DEFAULT_CALIBRATION.section("ground")["strike_attempts_per_30s"] == 1.6
