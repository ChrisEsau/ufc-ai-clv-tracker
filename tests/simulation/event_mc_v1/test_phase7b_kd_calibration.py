from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.phase7b_kd_calibration import TARGET_KD_PER_15MIN, calibration_for_midpoint, evaluate, temporal_cohorts


def test_in_memory_override_changes_only_knockdown_midpoint():
    candidate = calibration_for_midpoint(48)
    assert candidate.section("knockdown")["midpoint_impact_ratio"] == 48
    for section, values in DEFAULT_CALIBRATION.values.items():
        for key, value in values.items():
            if (section, key) != ("knockdown", "midpoint_impact_ratio"):
                assert candidate.section(section)[key] == value


def test_kd_exposure_target_uses_elapsed_master_time():
    assert TARGET_KD_PER_15MIN == 0.4398013629880078


def test_candidate_evaluation_is_same_seed_deterministic():
    train, _, fsr = temporal_cohorts(train_limit=1, holdout_limit=1)
    first = evaluate(train, fsr, 32, 1, 77)
    second = evaluate(train, fsr, 32, 1, 77)
    first.pop("runtime_seconds"); second.pop("runtime_seconds")
    assert first == second
    assert first["landed_per_path"] >= 0
    assert first["kd_per_path"] >= 0
    assert first["mean_fight_duration"] > 0
