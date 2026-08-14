from copy import deepcopy

import pytest

from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.phase7l_distance_td_calibration import (
    EXPECTED_ANCHORS,
    calibration_for_distance_td,
    validate_historical_anchors,
)


def test_candidate_changes_only_distance_td_attempt_base():
    candidate = calibration_for_distance_td(0.14)
    for section, values in DEFAULT_CALIBRATION.values.items():
        for key, value in values.items():
            if (section, key) == ("distance", "td_attempt_base_30s"):
                assert candidate.section(section)[key] == 0.14
            else:
                assert candidate.section(section)[key] == value


def test_phase7l_calibration_locks_remain_fixed():
    candidate = calibration_for_distance_td(0.16)
    assert candidate.section("clinch")["td_attempt_base_30s"] == 0.24
    assert candidate.section("distance")["td_success_logit_offset"] == -0.85
    assert candidate.section("distance")["strike_attempts_per_30s"] == 6.0
    assert candidate.section("clinch")["strike_attempts_per_30s"] == 3.6
    assert candidate.section("submission_finish")["intercept"] == -0.60
    assert candidate.section("knockdown")["midpoint_impact_ratio"] == 36.0
    assert candidate.section("finish")["midpoint_impact_ratio"] == 36.0


def test_promoted_distance_td_base_is_active():
    assert DEFAULT_CALIBRATION.section("distance")["td_attempt_base_30s"] == 0.16


def test_historical_anchor_validation_fails_closed():
    validate_historical_anchors("train", deepcopy(EXPECTED_ANCHORS["train"]))
    changed = deepcopy(EXPECTED_ANCHORS["train"])
    changed["attempts_per_fight"] += 0.01
    with pytest.raises(RuntimeError, match="historical TD anchors changed"):
        validate_historical_anchors("train", changed)
