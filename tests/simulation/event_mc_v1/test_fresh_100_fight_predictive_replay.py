from dataclasses import fields

import pytest

from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.fresh_100_fight_predictive_replay import (
    CUTOFF,
    EXPECTED_FSR_SHA256,
    JOINT_CLASSES,
    _simulate_one,
    build_simulation_inputs,
    fsr_sha256,
    probabilities_from_counts,
    select_fresh_cohort,
)
from pipeline.simulation.event_mc_v1.diagnostics.phase7b_kd_calibration import temporal_cohorts
from pipeline.simulation.event_mc_v1.diagnostics.phase7m_td_success_calibration import current_calibration_values


@pytest.fixture(scope="module")
def fresh():
    return select_fresh_cohort(100)


def test_fresh_cohort_is_strictly_post_cutoff_and_disjoint(fresh):
    cohort, _, metadata = fresh
    train, holdout, _ = temporal_cohorts(100, 50)
    calibration_ids = set(train.fight_id.astype(str)) | set(holdout.fight_id.astype(str))
    assert len(cohort) == 100
    assert (cohort.event_date > CUTOFF).all()
    assert set(cohort.fight_id.astype(str)).isdisjoint(calibration_ids)
    assert metadata["calibration_overlap_count"] == 0


def test_simulation_inputs_do_not_carry_actual_outcomes(fresh):
    cohort, fsr, _ = fresh
    fight = build_simulation_inputs(cohort.head(1), fsr)[0]
    input_fields = {item.name for item in fields(fight)}
    assert input_fields.isdisjoint({"winner", "method", "finish_round", "match_time_sec"})


def test_joint_probabilities_and_marginals_are_coherent_and_reproducible():
    counts = {"red_KO_TKO": 2, "red_SUB": 3, "red_DEC": 5,
              "blue_KO_TKO": 7, "blue_SUB": 11, "blue_DEC": 12}
    first = probabilities_from_counts(counts, 40)
    second = probabilities_from_counts(counts, 40)
    assert first == second
    assert sum(first["joint"].values()) == pytest.approx(1)
    assert first["red"] + first["blue"] == pytest.approx(1)
    assert sum(first["methods"].values()) == pytest.approx(1)
    assert first["red"] == pytest.approx(sum(first["joint"][f"red_{m}"] for m in ("KO_TKO", "SUB", "DEC")))
    for method in ("KO_TKO", "SUB", "DEC"):
        assert first["methods"][method] == pytest.approx(first["joint"][f"red_{method}"] + first["joint"][f"blue_{method}"])
    assert set(first["joint"]) == set(JOINT_CLASSES)


def test_per_fight_path_seed_is_reproducible(fresh):
    cohort, fsr, _ = fresh
    fight = build_simulation_inputs(cohort.head(1), fsr)[0]
    arguments = (0, fight, 2, 20260813)
    assert _simulate_one(arguments) == _simulate_one(arguments)


def test_active_calibration_and_fsr_are_frozen():
    assert fsr_sha256() == EXPECTED_FSR_SHA256
    assert current_calibration_values(DEFAULT_CALIBRATION) == {
        "distance_strike_attempts_per_30s": 6.0, "clinch_strike_attempts_per_30s": 3.6,
        "ground_strike_attempts_per_30s": 1.6, "distance_strike_accuracy": .40,
        "clinch_strike_accuracy": .68, "ground_strike_accuracy": .70,
        "distance_td_attempt_base_30s": .16, "clinch_td_attempt_base_30s": .24,
        "td_success_logit_offset": -.85, "submission_attempt_base_30s": .045,
        "submission_bottom_multiplier": 1.0, "submission_conversion_intercept": -.60,
        "submission_top_bonus": 0.0, "submission_bottom_bonus": 0.0,
        "kd_midpoint": 36.0, "finish_midpoint": 36.0,
    }
