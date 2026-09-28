import inspect

import pandas as pd
import pytest

from pipeline.common.paths import FSR_V2_PREFIGHT_SNAPSHOTS_PATH, MASTER_PATH
from pipeline.simulation.event_mc_v1.components.fsr_v2 import FSR_V2_SIMULATOR_FIELDS, FSR_V2_TRAIT_FIELDS
from pipeline.simulation.event_mc_v1.diagnostics import population_validation
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import (
    _fight,
    _add_prior_ufc_fight_counts,
    build_cohort,
    compute_metrics,
    observed_duration_seconds,
    simulate_fight,
)


def canonical_row(fighter_id, fight_id="target", event_date="2020-01-02", **changes):
    values = {name: 0.0 for name in FSR_V2_TRAIT_FIELDS}
    values.update({"fighter_id": fighter_id, "fighter_name": f"name-{fighter_id}",
        "fight_id": fight_id, "event_date": pd.Timestamp(event_date),
        "standing_striking_tendency": .08, "takedown_tendency": .02,
        "ground_striking_tendency": .05, "submission_tendency": .01,
        "head_strike_tendency": .7, "body_strike_tendency": .3, "leg_strike_tendency": .2,
        "stamina_capacity": 100.0, "stamina_depletion_resistance": 61.0,
        "stamina_performance_resilience": 62.0, "striking_power": 63.0,
        "damage_durability": 64.0, "knockdown_resistance": 65.0,
        "standing_accuracy_baseline": .47, "takedown_completion_baseline": .38,
        "ground_accuracy_baseline": .56, "submission_conversion_baseline": .21,
        "escape_population_mean_seconds": 42.0})
    values.update(changes)
    return values


def master_row(fight_id, date, red, blue, **changes):
    values = {"fight_id": fight_id, "date": date, "r_id": red, "b_id": blue,
        "r_name": f"master-{red}", "b_name": f"master-{blue}", "winner": f"master-{red}",
        "division": "Lightweight", "total_rounds": 3, "method": "Decision",
        "finish_round": 3, "match_time_sec": 900, "r_kd": 0, "b_kd": 0,
        "r_sub_att": 0, "b_sub_att": 0}
    values.update(changes)
    return values


def test_one_historical_fight_produces_compact_resolved_row(monkeypatch):
    masters = [master_row(f"prior-{i}", f"2019-0{i+1}-01", "red", "blue") for i in range(3)]
    masters.append(master_row("target", "2020-01-02", "red", "blue"))
    snapshots = pd.DataFrame([canonical_row("red"), canonical_row("blue")])
    def fake_read(path, *args, **kwargs):
        return pd.DataFrame(masters) if path == MASTER_PATH else snapshots
    monkeypatch.setattr(pd, "read_parquet", fake_read)
    cohort, fsr = build_cohort(start_year=2020, limit=1)
    row = simulate_fight(cohort.iloc[0], fsr, paths=2, seed=91)
    assert row["red_win_probability"] + row["blue_win_probability"] == pytest.approx(1)
    assert sum(row[f"{method}_probability"] for method in ("ko_tko", "sub", "dec")) == pytest.approx(1)
    assert not any("trace" in key for key in row)
    repeat = simulate_fight(cohort.iloc[0], fsr, paths=2, seed=91)
    assert row == repeat


def test_population_harness_uses_only_canonical_fsr_v2_path():
    source = inspect.getsource(population_validation)
    assert "FSR_32_PATH" not in source
    assert "FSR_V2_PREFIGHT_SNAPSHOTS_PATH" in source


def test_cohort_prior_counts_are_strictly_prior_date_and_both_corners(monkeypatch):
    masters = [
        master_row("r1", "2019-01-01", "red", "x1"),
        master_row("r2", "2019-02-01", "red", "x2"),
        master_row("r3", "2019-03-01", "red", "x3"),
        master_row("b1", "2019-01-01", "blue", "y1"),
        master_row("b2", "2019-02-01", "blue", "y2"),
        master_row("b3", "2020-01-02", "blue", "y3"),
        # This same-date bout must not make blue eligible for target.
        master_row("target", "2020-01-02", "red", "blue"),
        master_row("eligible", "2020-02-01", "red", "blue"),
    ]
    snapshots = pd.DataFrame([canonical_row("red", "eligible", "2020-02-01"),
                              canonical_row("blue", "eligible", "2020-02-01")])
    seen = []
    def fake_read(path, *args, **kwargs):
        seen.append(path)
        return pd.DataFrame(masters) if path == MASTER_PATH else snapshots
    monkeypatch.setattr(pd, "read_parquet", fake_read)
    cohort, _ = build_cohort(start_year=2020)
    assert cohort["fight_id"].tolist() == ["eligible"]
    assert cohort.iloc[0]["r_prior_ufc_fights"] == 4
    assert cohort.iloc[0]["b_prior_ufc_fights"] == 4
    counted = _add_prior_ufc_fight_counts(pd.DataFrame(masters))
    target = counted[counted["fight_id"] == "target"].iloc[0]
    assert target["b_prior_ufc_fights"] == 2
    assert seen == [MASTER_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH]


def test_resolver_uses_id_date_not_names_and_builds_canonical_matchup():
    row = pd.Series(master_row("target", "2020-01-02", "red", "blue",
                               r_name="not snapshot name", b_name="also wrong",
                               event_date=pd.Timestamp("2020-01-02")))
    fsr = pd.DataFrame([canonical_row("red"), canonical_row("blue")])
    fight = _fight(row, fsr)
    assert fight.fsr_v2_matchup is not None
    assert fight.fsr_v2_matchup.red.fighter_id == "red"
    assert fight.red_name == "name-red"
    assert len(FSR_V2_SIMULATOR_FIELDS) == 27
    assert set(fight.fsr_v2_matchup.red.audit_traits()) == set(FSR_V2_SIMULATOR_FIELDS)
    with pytest.raises(ValueError, match="found 0"):
        _fight(row, fsr.assign(event_date=pd.Timestamp("2020-01-03")))
    with pytest.raises(ValueError, match="found 2"):
        _fight(row, pd.concat([fsr, fsr.iloc[[0]]], ignore_index=True))


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
