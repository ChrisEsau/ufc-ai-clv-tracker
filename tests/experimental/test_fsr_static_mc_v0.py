from __future__ import annotations

import pandas as pd

from scripts.experimental.fsr_static_mc_v0 import (
    SEGMENTS_PER_ROUND,
    StaticFSRMCV0,
    _style_preferences,
)


def _profile(fid: str, **overrides) -> pd.Series:
    data = {
        "fighter_id": fid,
        "distance_striking_pressure": 50.0,
        "distance_striking_precision": 50.0,
        "distance_striking_defense": 50.0,
        "clinch_striking_pressure": 50.0,
        "clinch_striking_precision": 50.0,
        "clinch_striking_defense": 50.0,
        "ground_striking_pressure": 50.0,
        "ground_striking_precision": 50.0,
        "ground_striking_defense": 50.0,
        "wrestling_entry": 50.0,
        "wrestling_conversion": 50.0,
        "td_defense": 50.0,
        "control_imposition": 50.0,
        "control_resistance": 50.0,
        "submission_pressure": 50.0,
        "reversal_ability": 50.0,
    }
    data.update(overrides)
    return pd.Series(data)


def test_balanced_profile_has_zero_relative_style_preferences():
    d, c, w = _style_preferences(_profile("a"))
    assert d == 0.0
    assert c == 0.0
    assert w == 0.0


def test_v0_is_deterministic_for_fixed_seed():
    red = _profile("red")
    blue = _profile("blue")

    first = StaticFSRMCV0(red, blue, rounds=1, seed=123).run()
    second = StaticFSRMCV0(red, blue, rounds=1, seed=123).run()

    assert first.events == second.events
    assert first.stats == second.stats


def test_v0_emits_ten_segments_per_round_and_valid_phases():
    red = _profile("red")
    blue = _profile("blue")
    path = StaticFSRMCV0(red, blue, rounds=2, seed=5).run()

    assert len(path.events) == 2 * SEGMENTS_PER_ROUND
    valid = {"DISTANCE", "CLINCH", "GROUND"}
    assert all(e["phase_start"] in valid for e in path.events)
    assert all(e["phase_end"] in valid for e in path.events)


def test_td_success_probability_improves_with_conversion_edge():
    weak = _profile("weak", wrestling_conversion=42.0)
    strong = _profile("strong", wrestling_conversion=58.0)
    defender = _profile("def", td_defense=50.0)

    weak_sim = StaticFSRMCV0(weak, defender, rounds=1, seed=1)
    strong_sim = StaticFSRMCV0(strong, defender, rounds=1, seed=1)

    assert strong_sim._td_success_prob(0) > weak_sim._td_success_prob(0)


def test_wrestling_entry_increases_td_attempt_hazard():
    low = _profile("low", wrestling_entry=42.0)
    high = _profile("high", wrestling_entry=58.0)
    opponent = _profile("opp")

    low_sim = StaticFSRMCV0(low, opponent, rounds=1, seed=1)
    high_sim = StaticFSRMCV0(high, opponent, rounds=1, seed=1)

    assert high_sim._td_attempt_hazard(0, "DISTANCE") > low_sim._td_attempt_hazard(0, "DISTANCE")
