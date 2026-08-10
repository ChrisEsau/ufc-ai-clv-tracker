from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experimental import fsr_static_mc_ko_tko_v3_hybrid as hybrid


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
        "recovery_ability": 50.0,
    }
    data.update(overrides)
    return pd.Series(data)


def _sim(**kwargs) -> hybrid.StaticFSRMCKOTKOV3Hybrid:
    return hybrid.StaticFSRMCKOTKOV3Hybrid(
        _profile("red"),
        _profile("blue"),
        rounds=3,
        seed=7,
        **kwargs,
    )


def test_acute_ko_can_finish_with_substantial_reservoir_remaining(monkeypatch) -> None:
    sim = _sim()
    monkeypatch.setattr(sim, "_draw_strike_damage", lambda attacker: 8.0)
    monkeypatch.setattr(sim, "_knockdown_probability", lambda defender, strike_damage: 1.0)
    monkeypatch.setattr(
        sim,
        "_acute_ko_probability_given_kd",
        lambda defender, strike_damage: 1.0,
    )

    sim._apply_landed_strikes(0, 1)

    assert sim.finish is not None
    assert sim.finish.finish_route == "acute_ko"
    assert sim.finish.knockdown_on_strike is True
    assert sim.damage_state[1].reservoir_current == pytest.approx(92.0)
    assert sim.finish.reservoir_fraction_after == pytest.approx(0.92)


def test_post_kd_tko_can_finish_without_new_knockdown(monkeypatch) -> None:
    sim = _sim()
    sim.damage_state[1].recent_knockdown_segments = 2
    monkeypatch.setattr(sim, "_draw_strike_damage", lambda attacker: 1.0)
    monkeypatch.setattr(sim, "_knockdown_probability", lambda defender, strike_damage: 0.0)
    monkeypatch.setattr(
        sim,
        "_post_kd_tko_probability",
        lambda attacker, defender, strike_damage: 1.0,
    )

    sim._apply_landed_strikes(0, 1)

    assert sim.finish is not None
    assert sim.finish.finish_route == "post_kd_tko"
    assert sim.finish.knockdown_on_strike is False
    assert sim.finish.recent_kd_before is True
    assert sim.damage_state[1].reservoir_current == pytest.approx(99.0)


def test_recent_kd_does_not_multiply_followup_strike_damage(monkeypatch) -> None:
    sim = _sim()
    sim.damage_state[1].recent_knockdown_segments = 2
    monkeypatch.setattr(sim, "_draw_strike_damage", lambda attacker: 3.0)
    monkeypatch.setattr(sim, "_knockdown_probability", lambda defender, strike_damage: 0.0)
    monkeypatch.setattr(
        sim,
        "_post_kd_tko_probability",
        lambda attacker, defender, strike_damage: 0.0,
    )

    sim._apply_landed_strikes(0, 1)

    assert sim.finish is None
    assert sim.damage_state[1].reservoir_current == pytest.approx(97.0)
    assert sim.stats[0].damage_dealt == pytest.approx(3.0)
    assert sim.stats[1].damage_absorbed == pytest.approx(3.0)


def test_reservoir_exhaustion_remains_terminal_safeguard(monkeypatch) -> None:
    sim = _sim()
    monkeypatch.setattr(sim, "_draw_strike_damage", lambda attacker: 150.0)
    monkeypatch.setattr(sim, "_knockdown_probability", lambda defender, strike_damage: 0.0)

    sim._apply_landed_strikes(0, 1)

    assert sim.finish is not None
    assert sim.finish.finish_route == "cumulative_exhaustion"
    assert sim.damage_state[1].reservoir_current == pytest.approx(0.0)


def test_between_round_recovery_restores_damage_and_expires_recent_kd() -> None:
    sim = _sim()
    sim.damage_state[0].reservoir_current = 50.0
    sim.damage_state[0].recent_knockdown_segments = 2

    sim._apply_between_round_recovery(1)

    # At recovery_ability=50, 20% of the missing 50 reservoir is restored.
    assert sim.damage_state[0].reservoir_current == pytest.approx(60.0)
    assert sim.total_round_recovery[0] == pytest.approx(10.0)
    assert sim.damage_state[0].recent_knockdown_segments == 0
    assert sim.round_recovery_events[0]["recent_kd_before_break"] is True
    assert sim.round_recovery_events[0]["recent_kd_after_break"] is False


def test_age_adjustment_is_inherited_and_remains_narrow() -> None:
    sim = hybrid.StaticFSRMCKOTKOV3Hybrid(
        _profile("red", knockdown_resistance=60.0, damage_durability=60.0),
        _profile("blue"),
        rounds=3,
        seed=7,
        red_age=40.0,
        blue_age=30.0,
    )

    assert float(sim.raw_fighters[0]["knockdown_resistance"]) == pytest.approx(60.0)
    assert float(sim.raw_fighters[0]["damage_durability"]) == pytest.approx(60.0)
    assert float(sim.fighters[0]["knockdown_resistance"]) == pytest.approx(40.0)
    assert float(sim.fighters[0]["damage_durability"]) == pytest.approx(40.0)
    assert float(sim.fighters[0]["recovery_ability"]) == pytest.approx(50.0)


def test_hybrid_requires_existing_recovery_trait() -> None:
    red = _profile("red").drop(labels=["recovery_ability"])
    with pytest.raises(ValueError, match="recovery_ability"):
        hybrid.StaticFSRMCKOTKOV3Hybrid(red, _profile("blue"), rounds=3, seed=7)
