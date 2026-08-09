from __future__ import annotations

import pandas as pd

from scripts.experimental.fsr_static_mc_v0 import (
    DISTANCE_CLINCH_BASE,
    DISTANCE_CLINCH_BASE_30S,
    SEGMENT_SECONDS,
    SEGMENTS_PER_ROUND,
    StaticFSRMCV0,
    _rescale_interval_prob,
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


def test_v0_emits_thirty_ten_second_segments_per_round_and_valid_phases():
    red = _profile("red")
    blue = _profile("blue")
    path = StaticFSRMCV0(red, blue, rounds=2, seed=5).run()

    assert SEGMENT_SECONDS == 10
    assert SEGMENTS_PER_ROUND == 30
    assert len(path.events) == 2 * SEGMENTS_PER_ROUND
    valid = {"DISTANCE", "CLINCH", "GROUND"}
    assert all(e["phase_start"] in valid for e in path.events)
    assert all(e["phase_end"] in valid for e in path.events)


def test_30_second_hazard_is_rescaled_to_equivalent_10_second_hazard():
    expected = _rescale_interval_prob(DISTANCE_CLINCH_BASE_30S, 30, 10)
    assert DISTANCE_CLINCH_BASE == expected
    reconstructed_30s = 1.0 - (1.0 - DISTANCE_CLINCH_BASE) ** 3
    assert abs(reconstructed_30s - DISTANCE_CLINCH_BASE_30S) < 1e-12


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


def test_successful_takedown_sets_attacker_as_ground_controller():
    attacker = _profile("attacker", wrestling_conversion=90.0)
    defender = _profile("defender", td_defense=10.0)
    sim = StaticFSRMCV0(attacker, defender, rounds=1, seed=1)

    # With this extreme edge, success probability is effectively certain for
    # the deterministic unit test.
    note = sim._attempt_takedown(0, "DISTANCE")

    assert "TD SUCCESS" in note
    assert sim.phase == "GROUND"
    assert sim.ground_controller == 0


def test_reversal_swaps_ground_controller_when_reversal_occurs():
    top = _profile("top", control_imposition=10.0)
    bottom = _profile("bottom", reversal_ability=90.0, control_resistance=90.0)
    sim = StaticFSRMCV0(top, bottom, rounds=1, seed=2)
    sim.phase = "GROUND"
    sim.ground_controller = 0

    # Force the exit and reversal decisions while leaving the ownership logic
    # itself under test.
    sim._ground_exit_hazard = lambda controller: 1.0
    sim._reversal_probability = lambda bottom_i, controller_i: 1.0
    sim._maybe_submission_attempt = lambda fighter, **kwargs: False

    note = sim._ground_transition()

    assert "REVERSAL" in note
    assert sim.phase == "GROUND"
    assert sim.ground_controller == 1
    assert sim.stats[1].reversals == 1
