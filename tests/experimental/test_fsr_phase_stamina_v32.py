from __future__ import annotations

import pandas as pd
import pytest

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_static_mc_ko_tko_v3_stamina as v3
from scripts.experimental import fsr_static_mc_ko_tko_v3_2_phase_stamina as v32


def _profile() -> pd.Series:
    return pd.Series(
        {
            "fighter_id": "x",
            "striking_power": 60.0,
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
            fsr32.STAMINA_PERFORMANCE_RESILIENCE: 50.0,
            fsr32.STAMINA_RECOVERY_ABILITY: 50.0,
        }
    )


def _bare_sim():
    sim = object.__new__(v32.StaticFSRMCKOTKOV32PhaseStamina)
    p0 = _profile()
    p1 = _profile()
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


def test_v32_preserves_v31_power_curve_and_power_only_contract() -> None:
    assert v32.FATIGUE_CURVE_EXPONENT == pytest.approx(2.50)
    assert v32.FATIGUE_SENSITIVE_TRAITS == {"striking_power"}


def test_ground_bottom_resistance_is_queued_for_controlled_fighter() -> None:
    sim = _bare_sim()
    before = sim.stamina_state[1].current
    sim._queue_controlled_fighter_cost(
        controller=0,
        per_second=v32.STAMINA_COST_GROUND_BOTTOM_RESISTANCE_PER_SECOND,
        reason="ground_bottom_resistance",
    )

    expected = 10.0 * v32.STAMINA_COST_GROUND_BOTTOM_RESISTANCE_PER_SECOND
    assert sim.pending_stamina_costs[1] == [(expected, "ground_bottom_resistance")]
    assert sim.pending_stamina_costs[0] == []
    assert sim.stamina_state[1].current == pytest.approx(before)

    sim._flush_pending_stamina_costs()
    assert sim.stamina_state[1].current < before


def test_clinch_resistance_is_queued_for_controlled_fighter() -> None:
    sim = _bare_sim()
    sim._queue_controlled_fighter_cost(
        controller=1,
        per_second=v32.STAMINA_COST_CLINCH_RESISTANCE_PER_SECOND,
        reason="clinch_resistance",
    )

    expected = 10.0 * v32.STAMINA_COST_CLINCH_RESISTANCE_PER_SECOND
    assert sim.pending_stamina_costs[0] == [(expected, "clinch_resistance")]
    assert sim.pending_stamina_costs[1] == []


def test_new_defensive_cost_order_matches_candidate_hierarchy() -> None:
    assert v32.STAMINA_COST_TAKEDOWN_DEFENSE > (
        10.0 * v32.STAMINA_COST_GROUND_BOTTOM_RESISTANCE_PER_SECOND
    )
    assert v32.STAMINA_COST_SUBMISSION_DEFENSE > (
        10.0 * v32.STAMINA_COST_CLINCH_RESISTANCE_PER_SECOND
    )
    assert v32.STAMINA_COST_GROUND_BOTTOM_RESISTANCE_PER_SECOND > (
        v32.STAMINA_COST_GROUND_CONTROL_PER_SECOND
    )
    assert v32.STAMINA_COST_CLINCH_RESISTANCE_PER_SECOND >= (
        v32.STAMINA_COST_CLINCH_CONTROL_PER_SECOND
    )


def test_new_costs_are_additive_and_do_not_change_existing_constants() -> None:
    assert v32.STAMINA_COST_STRIKE_ATTEMPT == pytest.approx(0.70)
    assert v32.STAMINA_COST_TD_ATTEMPT == pytest.approx(3.00)
    assert v32.STAMINA_COST_TD_SUCCESS == pytest.approx(1.00)
    assert v32.STAMINA_COST_CLINCH_CONTROL_PER_SECOND == pytest.approx(0.025)
    assert v32.STAMINA_COST_GROUND_CONTROL_PER_SECOND == pytest.approx(0.025)
    assert v32.STAMINA_COST_ESCAPE == pytest.approx(1.50)
    assert v32.STAMINA_COST_REVERSAL == pytest.approx(2.50)
