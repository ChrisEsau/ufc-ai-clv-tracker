import pandas as pd
import pytest

from pipeline.simulation.event_mc_v1.diagnostics.phase7d2_kd_reconciliation import historical_anchors


def test_historical_anchors_keep_kd_targets_separate_and_use_elapsed_time():
    cohort = pd.DataFrame({
        "finish_round": [2, 3], "match_time_sec": [420, 750],
        "r_total_str_landed": [50, 40], "b_total_str_landed": [30, 30],
        "r_kd": [1, 0], "b_kd": [0, 1],
    })
    result = historical_anchors(cohort)
    assert result["kd_per_fight"] == 1
    assert result["kd_per_100_landed"] == pytest.approx(1.3333333333)
    assert result["kd_per_15min"] == pytest.approx(2 / 1170 * 900)
    assert result["landed_per_fight"] == 75
    assert result["mean_fight_duration"] == 585
