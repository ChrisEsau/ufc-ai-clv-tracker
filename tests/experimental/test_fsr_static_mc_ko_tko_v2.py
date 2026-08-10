from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko


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
        "striking_power": 50.0,
        "knockdown_resistance": 50.0,
        "damage_durability": 50.0,
    }
    data.update(overrides)
    return pd.Series(data)


def _params() -> ko.KOParameters:
    return ko.KOParameters(
        name="test",
        base_logit=-12.0,
        shock_coefficient=40.0,
        shock_curvature=8.0,
        depletion_coefficient=3.0,
        current_kd_logit_bonus=2.0,
        recent_kd_logit_bonus=1.0,
    )


def test_large_shock_is_much_more_dangerous_than_ordinary_shock() -> None:
    sim = ko.StaticFSRMCKOTKOV2(_profile("a"), _profile("b"), ko_params=_params(), rounds=1, seed=1)
    state = sim.damage_state[1]
    ordinary = sim._ko_probability(
        1,
        state.reservoir_capacity * 0.01,
        reservoir_fraction_before=1.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )
    severe = sim._ko_probability(
        1,
        state.reservoir_capacity * 0.10,
        reservoir_fraction_before=1.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )
    assert severe > ordinary * 100.0


def test_depletion_increases_same_shock_finish_probability() -> None:
    sim = ko.StaticFSRMCKOTKOV2(_profile("a"), _profile("b"), ko_params=_params(), rounds=1, seed=2)
    state = sim.damage_state[1]
    strike_damage = state.reservoir_capacity * 0.06
    fresh = sim._ko_probability(
        1,
        strike_damage,
        reservoir_fraction_before=1.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )
    state.reservoir_current = state.reservoir_capacity * 0.25
    depleted = sim._ko_probability(
        1,
        strike_damage,
        reservoir_fraction_before=0.25,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )
    assert depleted > fresh


def test_current_and_recent_kd_are_secondary_finish_modifiers() -> None:
    sim = ko.StaticFSRMCKOTKOV2(_profile("a"), _profile("b"), ko_params=_params(), rounds=1, seed=3)
    state = sim.damage_state[1]
    strike_damage = state.reservoir_capacity * 0.07
    baseline = sim._ko_probability(
        1,
        strike_damage,
        reservoir_fraction_before=1.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )
    current_kd = sim._ko_probability(
        1,
        strike_damage,
        reservoir_fraction_before=1.0,
        knockdown_on_strike=True,
        recent_kd_before=False,
    )
    recent_kd = sim._ko_probability(
        1,
        strike_damage,
        reservoir_fraction_before=1.0,
        knockdown_on_strike=False,
        recent_kd_before=True,
    )
    assert current_kd > baseline
    assert recent_kd > baseline


def test_kd_does_not_add_bonus_reservoir_damage() -> None:
    sim = ko.StaticFSRMCKOTKOV2(_profile("a"), _profile("b"), ko_params=_params(), rounds=1, seed=4)
    before = sim.damage_state[1].reservoir_current
    sim._draw_strike_damage = lambda attacker: 7.5
    sim._knockdown_probability = lambda defender, strike_damage: 1.0
    sim._ko_probability = lambda *args, **kwargs: 0.0
    total_damage, knockdowns = sim._apply_landed_strikes(0, 1)
    assert total_damage == 7.5
    assert knockdowns == 1
    assert sim.damage_state[1].reservoir_current == before - 7.5


def test_fixed_seed_is_deterministic() -> None:
    red = _profile("red", striking_power=58.0)
    blue = _profile("blue", damage_durability=60.0, knockdown_resistance=57.0)
    sim1 = ko.StaticFSRMCKOTKOV2(red, blue, ko_params=_params(), rounds=1, seed=123)
    path1 = sim1.run()
    sim2 = ko.StaticFSRMCKOTKOV2(red, blue, ko_params=_params(), rounds=1, seed=123)
    path2 = sim2.run()
    assert path1.events == path2.events
    assert path1.finish == path2.finish
    assert sim1.stats == sim2.stats
