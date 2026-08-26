import pandas as pd

from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.phase7k_takedown_decomposition import (
    historical_takedowns,
    source_totals,
)


def test_round_td_aggregation_uses_both_fighters_and_elapsed_fight_time():
    cohort = pd.DataFrame({
        "fight_id": ["a", "b"], "finish_round": [2, 1], "match_time_sec": [420, 120]
    })
    rounds = pd.DataFrame([
        {"fight_id": "a", "corner": "red", "round": 1, "td_attempted": 2, "td_landed": 1},
        {"fight_id": "a", "corner": "blue", "round": 1, "td_attempted": 1, "td_landed": 0},
        {"fight_id": "a", "corner": "red", "round": 2, "td_attempted": 0, "td_landed": 0},
        {"fight_id": "a", "corner": "blue", "round": 2, "td_attempted": 1, "td_landed": 1},
        {"fight_id": "b", "corner": "red", "round": 1, "td_attempted": 0, "td_landed": 0},
        {"fight_id": "b", "corner": "blue", "round": 1, "td_attempted": 0, "td_landed": 0},
    ])
    result = historical_takedowns(cohort, rounds)
    assert result["total_attempts"] == 4
    assert result["total_landed"] == 2
    assert result["exposure_seconds"] == 540
    assert result["attempts_per_15min"] == 4 / 540 * 900
    assert result["landed_per_15min"] == 2 / 540 * 900
    assert result["success_percentage"] == 0.5
    assert result["fights_with_attempt_share"] == 0.5
    assert result["fights_with_landed_share"] == 0.5
    assert result["zero_attempt_share"] == 0.5
    assert result["multi_attempt_share"] == 0.5


def test_zero_attempt_success_is_zero_safe():
    cohort = pd.DataFrame({"fight_id": ["z"], "finish_round": [1], "match_time_sec": [60]})
    rounds = pd.DataFrame([
        {"fight_id": "z", "corner": corner, "round": 1, "td_attempted": 0, "td_landed": 0}
        for corner in ("red", "blue")
    ])
    assert historical_takedowns(cohort, rounds)["success_percentage"] == 0.0


def test_simulator_total_td_accounting_equals_distance_plus_clinch():
    attempts, landed = source_totals(3, 2, 1, 2)
    assert attempts == 5
    assert landed == 3


def test_phase7k_keeps_all_calibration_locks():
    assert DEFAULT_CALIBRATION.section("distance")["strike_attempts_per_30s"] == 6.0
    assert DEFAULT_CALIBRATION.section("clinch")["strike_attempts_per_30s"] == 3.6
    assert DEFAULT_CALIBRATION.section("ground")["strike_attempts_per_30s"] == 1.6
    assert DEFAULT_CALIBRATION.section("distance")["td_attempt_base_30s"] == 0.16
    assert DEFAULT_CALIBRATION.section("clinch")["td_attempt_base_30s"] == 0.24
    assert DEFAULT_CALIBRATION.section("distance")["td_success_logit_offset"] == -0.85
    assert DEFAULT_CALIBRATION.section("submission_attempts")["base_30s"] == 0.045
    assert DEFAULT_CALIBRATION.section("submission_finish")["intercept"] == -0.60
    assert DEFAULT_CALIBRATION.section("knockdown")["midpoint_impact_ratio"] == 36.0
    assert DEFAULT_CALIBRATION.section("finish")["midpoint_impact_ratio"] == 36.0
