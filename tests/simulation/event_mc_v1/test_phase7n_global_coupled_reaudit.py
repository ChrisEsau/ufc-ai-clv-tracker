from copy import deepcopy

import pytest

from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.phase7l_distance_td_calibration import (
    EXPECTED_ANCHORS,
    validate_historical_anchors,
)
from pipeline.simulation.event_mc_v1.diagnostics.phase7m_td_success_calibration import (
    UNAVAILABLE,
    current_calibration_values,
)
from pipeline.simulation.event_mc_v1.diagnostics.phase7n_global_coupled_reaudit import (
    READINESS_LINE,
    classify_relative_error,
    validate_significant_anchor,
)


def test_phase7n_global_locks_match_phase7m_state():
    values = current_calibration_values(DEFAULT_CALIBRATION)
    assert values == {
        "distance_strike_attempts_per_30s": 6.0,
        "clinch_strike_attempts_per_30s": 3.6,
        "ground_strike_attempts_per_30s": 1.6,
        "distance_strike_accuracy": 0.40,
        "clinch_strike_accuracy": 0.68,
        "ground_strike_accuracy": 0.70,
        "distance_td_attempt_base_30s": 0.16,
        "clinch_td_attempt_base_30s": 0.24,
        "td_success_logit_offset": -0.85,
        "submission_attempt_base_30s": 0.045,
        "submission_bottom_multiplier": 1.0,
        "submission_conversion_intercept": -0.60,
        "submission_top_bonus": 0.0,
        "submission_bottom_bonus": 0.0,
        "kd_midpoint": 36.0,
        "finish_midpoint": 36.0,
    }


def test_historical_provenance_checks_fail_closed():
    for name, anchors in EXPECTED_ANCHORS.items():
        validate_historical_anchors(name, deepcopy(anchors))
    with pytest.raises(RuntimeError, match="significant-strike anchor changed"):
        validate_significant_anchor(
            "train", {"significant": {"attempts_per_15min": 999.0}}
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "CLOSE"), (5, "CLOSE"), (5.01, "MODERATE"), (10, "MODERATE"),
     (10.01, "MATERIAL"), (20, "MATERIAL"), (20.01, "LARGE"), (None, None)],
)
def test_mismatch_classification_is_deterministic(value, expected):
    assert classify_relative_error(value) == expected


def test_readiness_and_unavailable_labels_are_explicit():
    assert READINESS_LINE.endswith("NO")
    assert UNAVAILABLE == "historical comparator unavailable"
