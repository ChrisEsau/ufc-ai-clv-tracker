import pandas as pd
import pytest

from pipeline.simulation.event_mc_v1.diagnostics.population_validation import (
    build_cohort,
    compute_metrics,
    observed_duration_seconds,
    simulate_fight,
)


def test_one_historical_fight_produces_compact_resolved_row():
    cohort, fsr = build_cohort(start_year=2020, limit=1)
    row = simulate_fight(cohort.iloc[0], fsr, paths=2, seed=91)
    assert row["red_win_probability"] + row["blue_win_probability"] == pytest.approx(1)
    assert sum(row[f"{method}_probability"] for method in ("ko_tko", "sub", "dec")) == pytest.approx(1)
    assert not any("trace" in key for key in row)
    repeat = simulate_fight(cohort.iloc[0], fsr, paths=2, seed=91)
    assert row == repeat


def test_metrics_on_controlled_rows_and_exposure_denominator():
    rows = pd.DataFrame([
        {"fight_id":"a", "actual_red_win":1, "red_win_probability":.8, "actual_method":"KO_TKO", "ko_tko_probability":.7, "sub_probability":.1, "dec_probability":.2, "historical_kd":1, "actual_duration_seconds":300, "actual_finish_round":1, "kd_per_path":.5, "zero_kd_share":.5, "multi_kd_share":0, "historical_sub_attempts":0, "submission_attempts_per_path":.1, "simulated_nondecision_paths":1, "simulated_finish_time_sum_seconds":100, "simulated_total_kd":1, "simulated_total_exposure_seconds":1000, "simulated_paths_with_submission_attempt":1, "simulated_total_path_count":10, "sim_finish_r1_count":1, "sim_finish_r2_count":0, "sim_finish_r3_count":0, "sim_finish_r4_count":0, "sim_finish_r5_count":0},
        {"fight_id":"b", "actual_red_win":0, "red_win_probability":.2, "actual_method":"DEC", "ko_tko_probability":.2, "sub_probability":.1, "dec_probability":.7, "historical_kd":0, "actual_duration_seconds":900, "actual_finish_round":3, "kd_per_path":.5, "zero_kd_share":.5, "multi_kd_share":0, "historical_sub_attempts":2, "submission_attempts_per_path":.3, "simulated_nondecision_paths":9, "simulated_finish_time_sum_seconds":2700, "simulated_total_kd":3, "simulated_total_exposure_seconds":3000, "simulated_paths_with_submission_attempt":5, "simulated_total_path_count":10, "sim_finish_r1_count":0, "sim_finish_r2_count":9, "sim_finish_r3_count":0, "sim_finish_r4_count":0, "sim_finish_r5_count":0},
    ])
    metrics = compute_metrics(rows)
    assert metrics["winner_accuracy"] == 1
    assert metrics["brier_score"] == pytest.approx(.04)
    assert metrics["mean_probability_actual_winner"] == pytest.approx(.8)
    assert metrics["historical_kd_per_15_minutes"] == pytest.approx(.75)
    assert metrics["simulated_kd_per_15_minutes"] == pytest.approx(.9)
    assert metrics["simulated_finish_time_mean"] == pytest.approx(280)
    assert metrics["simulated_finish_round_shares"] == {"1":.1,"2":.9,"3":0,"4":0,"5":0}
    assert metrics["simulated_share_with_submission_attempt"] == pytest.approx(.3)
    assert metrics["historical_method_shares"] == {"KO_TKO":.5, "SUB":0.0, "DEC":.5}


@pytest.mark.parametrize(("finish_round", "elapsed"), [(1, 120), (2, 420), (3, 750)])
def test_observed_duration_uses_authoritative_elapsed_master_time(finish_round, elapsed):
    assert observed_duration_seconds({"finish_round": finish_round, "match_time_sec": elapsed}) == elapsed


def test_legacy_final_round_clock_requires_explicit_semantics():
    row = {"finish_round": 2, "match_time_sec": 273}
    assert observed_duration_seconds(row) == 273
    assert observed_duration_seconds(row, match_time_semantics="legacy_final_round") == 573
    with pytest.raises(ValueError, match="unsupported match_time_semantics"):
        observed_duration_seconds(row, match_time_semantics="guess")
