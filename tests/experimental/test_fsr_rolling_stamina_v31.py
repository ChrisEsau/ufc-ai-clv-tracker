from __future__ import annotations

import pandas as pd
import pytest

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_static_mc_ko_tko_v3_stamina as v3
from scripts.experimental import fsr_static_mc_ko_tko_v3_1_rolling_fsr as rolling


def _profile(power: float = 60.0, resilience: float = 50.0) -> pd.Series:
    return pd.Series(
        {
            "fighter_id": "x",
            "striking_power": power,
            "distance_striking_pressure": 55.0,
            "distance_striking_precision": 55.0,
            "distance_striking_defense": 55.0,
            "clinch_striking_pressure": 52.0,
            "clinch_striking_precision": 52.0,
            "clinch_striking_defense": 52.0,
            "ground_striking_pressure": 51.0,
            "ground_striking_precision": 51.0,
            "ground_striking_defense": 51.0,
            "wrestling_entry": 53.0,
            "wrestling_conversion": 53.0,
            "td_defense": 53.0,
            "control_imposition": 53.0,
            "control_resistance": 53.0,
            "submission_pressure": 53.0,
            "reversal_ability": 53.0,
            fsr32.STAMINA_CAPACITY: 100.0,
            fsr32.STAMINA_DEPLETION_RESISTANCE: 50.0,
            fsr32.STAMINA_PERFORMANCE_RESILIENCE: resilience,
            fsr32.STAMINA_RECOVERY_ABILITY: 50.0,
        }
    )


def _bare_sim(power: float = 60.0, resilience: float = 50.0):
    sim = object.__new__(rolling.StaticFSRMCKOTKOV31RollingFSR)
    p0 = _profile(power=power, resilience=resilience)
    p1 = _profile(power=power, resilience=resilience)
    sim.base_fighters = [p0.copy(deep=True), p1.copy(deep=True)]
    sim.fighters = [p0.copy(deep=True), p1.copy(deep=True)]
    sim.stamina_state = [
        v3.StaminaState(capacity=100.0, current=100.0),
        v3.StaminaState(capacity=100.0, current=100.0),
    ]
    sim.pending_stamina_costs = [[], []]
    sim.total_stamina_spent = [0.0, 0.0]
    sim.total_stamina_recovered = [0.0, 0.0]
    sim.stamina_events = []
    return sim


def test_effective_power_is_fresh_at_full_stamina_and_only_declines() -> None:
    sim = _bare_sim(power=60.0)
    fresh = sim._effective_profile(0)
    assert fresh["striking_power"] == pytest.approx(60.0)

    sim.stamina_state[0].current = 50.0
    tired = sim._effective_profile(0)
    assert tired["striking_power"] < fresh["striking_power"]
    assert tired["striking_power"] <= sim.base_fighters[0]["striking_power"]


def test_non_power_fsr_traits_are_not_fatigued() -> None:
    sim = _bare_sim(power=60.0)
    sim.stamina_state[0].current = 35.0
    tired = sim._effective_profile(0)

    for trait in (
        "distance_striking_pressure",
        "distance_striking_precision",
        "distance_striking_defense",
        "wrestling_entry",
        "wrestling_conversion",
        "td_defense",
        "control_imposition",
        "control_resistance",
        "submission_pressure",
        "reversal_ability",
    ):
        assert tired[trait] == pytest.approx(sim.base_fighters[0][trait])


def test_below_50_power_also_declines_with_fatigue() -> None:
    sim = _bare_sim(power=40.0)
    sim.stamina_state[0].current = 50.0
    tired = sim._effective_profile(0)
    assert tired["striking_power"] < 40.0


def test_fatigue_curve_protects_fresh_power_and_accelerates() -> None:
    sim = _bare_sim(power=60.0, resilience=50.0)

    penalties = {}
    for stamina in (90.0, 80.0, 70.0, 50.0, 30.0):
        sim.stamina_state[0].current = stamina
        penalties[int(stamina)] = sim.fatigue_penalty(0)

    assert 0.0 < penalties[90] < penalties[80] < penalties[70] < penalties[50] < penalties[30]
    assert penalties[90] == pytest.approx(45.0 * (0.10 ** 2.50), rel=1e-6)
    assert penalties[80] == pytest.approx(45.0 * (0.20 ** 2.50), rel=1e-6)
    assert penalties[50] == pytest.approx(45.0 * (0.50 ** 2.50), rel=1e-6)
    assert penalties[90] < 0.2
    assert penalties[80] < 1.0
    assert (penalties[50] - penalties[70]) > (penalties[70] - penalties[90])


def test_higher_resilience_reduces_power_penalty() -> None:
    low = _bare_sim(power=60.0, resilience=40.0)
    high = _bare_sim(power=60.0, resilience=60.0)
    low.stamina_state[0].current = 50.0
    high.stamina_state[0].current = 50.0
    assert high.fatigue_penalty(0) < low.fatigue_penalty(0)


def test_action_cost_is_deferred_until_segment_flush() -> None:
    sim = _bare_sim()
    before = sim.stamina_state[0].current
    sim._spend_stamina(0, 10.0, "test_action")
    assert sim.stamina_state[0].current == pytest.approx(before)
    assert sim.pending_stamina_costs[0] == [(10.0, "test_action")]

    sim._flush_pending_stamina_costs()
    assert sim.stamina_state[0].current < before
    assert sim.pending_stamina_costs[0] == []


def test_structural_and_stamina_traits_are_not_fatigued() -> None:
    sim = _bare_sim()
    sim.base_fighters[0]["knockdown_resistance"] = 61.0
    sim.base_fighters[0]["damage_durability"] = 66.0
    sim.stamina_state[0].current = 20.0
    tired = sim._effective_profile(0)
    assert tired["knockdown_resistance"] == pytest.approx(61.0)
    assert tired["damage_durability"] == pytest.approx(66.0)
    assert tired[fsr32.STAMINA_PERFORMANCE_RESILIENCE] == pytest.approx(50.0)
