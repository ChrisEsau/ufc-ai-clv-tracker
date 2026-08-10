from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experimental import fsr_static_mc_damage_v1_ground017 as ground017
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_v0 as base


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


def _sim(seed: int = 7) -> ko.StaticFSRMCKOTKOV2:
    return ko.StaticFSRMCKOTKOV2(_profile("red"), _profile("blue"), rounds=1, seed=seed)


def test_ground_017_variant_is_isolated_from_frozen_v0_constant() -> None:
    assert ground017.GROUND_EXIT_BASE_30S_SHADOW == pytest.approx(0.17)
    assert base.GROUND_EXIT_BASE_30S == pytest.approx(0.20)


def test_v2_has_no_generic_ko_probability_curve() -> None:
    assert not hasattr(ko.StaticFSRMCKOTKOV2, "_ko_probability")


def test_reservoir_exhaustion_causes_deterministic_finish(monkeypatch) -> None:
    sim = _sim()
    sim.damage_state[1].reservoir_current = 1.0
    monkeypatch.setattr(sim, "_draw_strike_damage", lambda attacker: 2.0)
    monkeypatch.setattr(sim, "_knockdown_probability", lambda defender, strike_damage: 0.0)

    damage_done, knockdowns = sim._apply_landed_strikes(0, 1)

    assert damage_done == pytest.approx(2.0)
    assert knockdowns == 0
    assert sim.damage_state[1].reservoir_current == pytest.approx(0.0)
    assert sim.finish is not None
    assert sim.finish.winner == 0
    assert sim.finish.loser == 1
    assert sim.finish.method == "KO/TKO"


def test_kd_causing_strike_gets_no_retroactive_bonus(monkeypatch) -> None:
    sim = _sim()
    before = sim.damage_state[1].reservoir_current
    monkeypatch.setattr(sim, "_draw_strike_damage", lambda attacker: 7.5)
    monkeypatch.setattr(sim, "_knockdown_probability", lambda defender, strike_damage: 1.0)

    total_damage, knockdowns = sim._apply_landed_strikes(0, 1)

    assert total_damage == pytest.approx(7.5)
    assert knockdowns == 1
    assert sim.damage_state[1].reservoir_current == pytest.approx(before - 7.5)


def test_recent_kd_amplifies_followup_damage(monkeypatch) -> None:
    sim = _sim()
    sim.damage_state[1].recent_knockdown_segments = 1
    before = sim.damage_state[1].reservoir_current
    monkeypatch.setattr(sim, "_draw_strike_damage", lambda attacker: 2.0)
    monkeypatch.setattr(sim, "_knockdown_probability", lambda defender, strike_damage: 0.0)

    total_damage, knockdowns = sim._apply_landed_strikes(0, 1)

    assert knockdowns == 0
    assert total_damage == pytest.approx(4.0)
    assert sim.damage_state[1].reservoir_current == pytest.approx(before - 4.0)
    assert sim.finish is None


def test_fixed_seed_is_deterministic() -> None:
    red = _profile("red", striking_power=58.0)
    blue = _profile("blue", damage_durability=60.0, knockdown_resistance=57.0)
    sim1 = ko.StaticFSRMCKOTKOV2(red, blue, rounds=1, seed=123)
    path1 = sim1.run()
    sim2 = ko.StaticFSRMCKOTKOV2(red, blue, rounds=1, seed=123)
    path2 = sim2.run()
    assert path1.events == path2.events
    assert path1.finish == path2.finish
    assert sim1.stats == sim2.stats
