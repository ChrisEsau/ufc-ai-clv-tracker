from __future__ import annotations

from scripts.experimental import fsr_reversal_v1 as reversal


def test_no_opponent_control_means_no_reversal_update():
    obs, quality, rate = reversal.observation(0.0, 0.0, [])
    assert obs is None
    assert quality == 0.0
    assert rate is None


def test_zero_reversal_under_control_is_negative_evidence():
    obs, quality, rate = reversal.observation(0.0, 180.0, [])
    assert obs == 0.0
    assert 0.0 < quality < 1.0
    assert rate is None


def test_positive_reversal_is_positive_evidence():
    pool = [1.0, 2.0, 4.0, 8.0]
    obs, quality, rate = reversal.observation(1.0, 180.0, pool)
    assert obs is not None
    assert obs >= 0.60
    assert 0.0 < quality < 1.0
    assert rate == 5.0


def test_higher_reversal_rating_beats_same_control_opponent():
    baseline = 0.20
    low = reversal.expected_matchup(48.0, 50.0, baseline)
    high = reversal.expected_matchup(54.0, 50.0, baseline)
    assert high > low


def test_stronger_control_opponent_reduces_expected_reversal_probability():
    baseline = 0.20
    weak_control = reversal.expected_matchup(52.0, 48.0, baseline)
    strong_control = reversal.expected_matchup(52.0, 56.0, baseline)
    assert weak_control > strong_control
