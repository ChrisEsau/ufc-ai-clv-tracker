from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experimental import fsr_static_mc_damage_v1 as damage


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


def test_selected_shadow_calibration_constants_are_exposed() -> None:
    assert damage.STRIKE_DAMAGE_SCALE == 0.50
    assert damage.KD_BASE_LOGIT == -8.635900
    assert damage.KD_SHOCK_COEFFICIENT == 80.0
    assert damage.KD_RESISTANCE_SCALE == 32.0
    assert damage.KD_DEPLETION_COEFFICIENT == 1.50
    assert damage.KD_RECENT_KD_LOGIT_BONUS == 0.50
    assert damage.RECENT_KD_SEGMENTS == 3


def test_capacity_mapping_is_centered_monotonic_and_bounded() -> None:
    low = damage.reservoir_capacity_from_durability(10.0)
    mid = damage.reservoir_capacity_from_durability(50.0)
    high = damage.reservoir_capacity_from_durability(90.0)

    assert low < mid < high
    assert mid == damage.AVERAGE_RESERVOIR_CAPACITY
    assert damage.MIN_RESERVOIR_CAPACITY <= low
    assert high <= damage.MAX_RESERVOIR_CAPACITY


def test_fight_starts_with_full_fighter_specific_reservoirs() -> None:
    sim = damage.StaticFSRMCDamageV1(
        _profile("low", damage_durability=30.0),
        _profile("high", damage_durability=70.0),
        rounds=1,
        seed=1,
    )

    assert sim.damage_state[0].reservoir_current == sim.damage_state[0].reservoir_capacity
    assert sim.damage_state[1].reservoir_current == sim.damage_state[1].reservoir_capacity
    assert sim.damage_state[1].reservoir_capacity > sim.damage_state[0].reservoir_capacity


def test_landed_strike_depletes_defender_reservoir_without_bonus_kd_damage() -> None:
    sim = damage.StaticFSRMCDamageV1(_profile("a"), _profile("b"), rounds=1, seed=2)
    before = sim.damage_state[1].reservoir_current

    sim._draw_strike_damage = lambda attacker: 7.5
    sim._knockdown_probability = lambda defender, strike_damage: 1.0
    total_damage, knockdowns = sim._apply_landed_strikes(0, 1)

    assert total_damage == 7.5
    assert knockdowns == 1
    assert sim.damage_state[1].reservoir_current == before - 7.5
    assert sim.damage_state[1].recent_knockdown
    assert sim.stats[0].knockdowns_scored == 1
    assert sim.stats[1].knockdowns_absorbed == 1


def test_selected_damage_scale_is_applied_to_draws() -> None:
    sim = damage.StaticFSRMCDamageV1(_profile("a"), _profile("b"), rounds=1, seed=22)
    sim.rng = type(
        "FixedRNG",
        (),
        {
            "gamma": staticmethod(lambda shape, scale: 4.0),
            "random": staticmethod(lambda: 1.0),
        },
    )()

    assert sim._draw_strike_damage(0) == 4.0 * damage.STRIKE_DAMAGE_SCALE


def test_higher_knockdown_resistance_reduces_same_shock_probability() -> None:
    attacker = _profile("attacker")
    low = damage.StaticFSRMCDamageV1(
        attacker,
        _profile("low", knockdown_resistance=30.0),
        rounds=1,
        seed=3,
    )
    high = damage.StaticFSRMCDamageV1(
        attacker,
        _profile("high", knockdown_resistance=70.0),
        rounds=1,
        seed=3,
    )

    assert low._knockdown_probability(1, 8.0) > high._knockdown_probability(1, 8.0)


def test_lower_reservoir_fraction_increases_same_shock_probability() -> None:
    sim = damage.StaticFSRMCDamageV1(_profile("a"), _profile("b"), rounds=1, seed=4)
    fresh = sim._knockdown_probability(1, 6.0)
    sim.damage_state[1].reservoir_current *= 0.35
    depleted = sim._knockdown_probability(1, 6.0)

    assert depleted > fresh


def test_recent_knockdown_increases_followup_kd_probability_and_expires() -> None:
    sim = damage.StaticFSRMCDamageV1(_profile("a"), _profile("b"), rounds=1, seed=5)
    baseline = sim._knockdown_probability(1, 6.0)
    sim.damage_state[1].recent_knockdown_segments = damage.RECENT_KD_SEGMENTS
    vulnerable = sim._knockdown_probability(1, 6.0)

    assert vulnerable > baseline

    for _ in range(damage.RECENT_KD_SEGMENTS):
        sim._advance_damage_timers()
    assert not sim.damage_state[1].recent_knockdown


def test_power_changes_upper_tail_probability_not_reservoir_capacity() -> None:
    low_power = damage.StaticFSRMCDamageV1(
        _profile("low", striking_power=30.0),
        _profile("opp"),
        rounds=1,
        seed=6,
    )
    high_power = damage.StaticFSRMCDamageV1(
        _profile("high", striking_power=70.0),
        _profile("opp"),
        rounds=1,
        seed=6,
    )

    assert high_power._tail_probability(0) > low_power._tail_probability(0)
    assert high_power.damage_state[0].reservoir_capacity == low_power.damage_state[0].reservoir_capacity


def test_fixed_seed_remains_deterministic_with_damage_state() -> None:
    red = _profile("red", striking_power=58.0, damage_durability=54.0)
    blue = _profile("blue", knockdown_resistance=57.0, damage_durability=61.0)

    sim1 = damage.StaticFSRMCDamageV1(red, blue, rounds=1, seed=123)
    path1 = sim1.run()
    sim2 = damage.StaticFSRMCDamageV1(red, blue, rounds=1, seed=123)
    path2 = sim2.run()

    assert path1.events == path2.events
    assert sim1.stats == sim2.stats
    assert sim1.damage_state == sim2.damage_state
