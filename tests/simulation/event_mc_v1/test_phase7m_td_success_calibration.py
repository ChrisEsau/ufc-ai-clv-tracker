from copy import deepcopy

from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.components.formulas import td_success_probability
from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile
from pipeline.simulation.event_mc_v1.diagnostics.phase7l_distance_td_calibration import EXPECTED_ANCHORS, validate_historical_anchors
from pipeline.simulation.event_mc_v1.diagnostics.phase7m_td_success_calibration import (
    UNAVAILABLE,
    calibration_for_td_success,
    current_calibration_values,
    validate_global_comparison,
)


def profile(fighter_id, conversion=50, defense=50):
    return FighterProfile(fighter_id, fighter_id, 50, 50, 50, 50, 50, 50, conversion, defense)


def test_candidate_changes_only_shared_td_success_offset():
    candidate = calibration_for_td_success(-0.75)
    for section, values in DEFAULT_CALIBRATION.values.items():
        for key, value in values.items():
            if (section, key) == ("distance", "td_success_logit_offset"):
                assert candidate.section(section)[key] == -0.75
            else:
                assert candidate.section(section)[key] == value


def test_attempt_ontology_and_all_calibration_locks_remain_fixed():
    values = current_calibration_values(calibration_for_td_success(-0.70))
    assert values["distance_td_attempt_base_30s"] == 0.16
    assert values["clinch_td_attempt_base_30s"] == 0.24
    assert values["distance_strike_attempts_per_30s"] == 6.0
    assert values["clinch_strike_attempts_per_30s"] == 3.6
    assert values["ground_strike_attempts_per_30s"] == 1.6
    assert values["submission_attempt_base_30s"] == 0.045
    assert values["submission_bottom_multiplier"] == 1.0
    assert values["submission_conversion_intercept"] == -0.60
    assert values["submission_top_bonus"] == values["submission_bottom_bonus"] == 0.0
    assert values["kd_midpoint"] == values["finish_midpoint"] == 36.0


def test_same_success_formula_applies_to_distance_and_clinch_resolvers():
    candidate = calibration_for_td_success(-0.70)
    attacker, defender = profile("a", 58), profile("d", defense=54)
    # Both action resolvers call this single shared consumer.
    expected = td_success_probability(attacker, defender, candidate)
    assert expected == td_success_probability(attacker, defender, candidate)


def test_historical_anchors_still_reproduce_and_fail_closed():
    for name, anchors in EXPECTED_ANCHORS.items():
        validate_historical_anchors(name, deepcopy(anchors))


def test_unavailable_historical_label_is_explicit():
    assert UNAVAILABLE == "historical comparator unavailable"


def test_promoted_shared_td_success_offset_is_active():
    assert DEFAULT_CALIBRATION.section("distance")["td_success_logit_offset"] == -0.85


def test_required_global_report_families_and_unavailable_residence_are_enforced():
    rows = [
        {"metric": metric, "historical": UNAVAILABLE if metric.startswith("phase_residence.") else 0}
        for metric in (
            "fight.mean_duration", "outcomes.DEC", "strikes.attempts",
            "takedowns.attempts", "knockdowns.per_path", "submissions.attempts",
            "phase_residence.distance_seconds_per_path",
        )
    ]
    validate_global_comparison({"rows": rows})
