from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery


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


def _sim(red_recovery: float = 50.0, blue_recovery: float = 50.0):
    strong = collapse.CollapseCandidate("strong", 5.0, 2.0)
    return recovery.StaticFSRMCKOTKOV2RoundRecovery(
        _profile("red", recovery_ability=red_recovery),
        _profile("blue", recovery_ability=blue_recovery),
        collapse=strong,
        rounds=3,
        seed=7,
    )


def test_recovery_curve_reference_points() -> None:
    assert recovery.round_recovery_fraction(10.0) == pytest.approx(0.05)
    assert recovery.round_recovery_fraction(50.0) == pytest.approx(0.20)
    assert recovery.round_recovery_fraction(90.0) == pytest.approx(0.35)


def test_between_round_recovery_restores_fraction_of_missing_reservoir() -> None:
    sim = _sim()
    sim.damage_state[0].reservoir_current = 50.0
    sim._apply_between_round_recovery(1)

    # Capacity is 100 at durability 50. Missing=50, so 20% recovery restores 10.
    assert sim.damage_state[0].reservoir_current == pytest.approx(60.0)
    assert sim.total_round_recovery[0] == pytest.approx(10.0)


def test_higher_recovery_ability_restores_more_from_same_state() -> None:
    sim = _sim(red_recovery=30.0, blue_recovery=70.0)
    sim.damage_state[0].reservoir_current = 50.0
    sim.damage_state[1].reservoir_current = 50.0
    sim._apply_between_round_recovery(1)

    assert sim.total_round_recovery[1] > sim.total_round_recovery[0]


def test_recovery_never_exceeds_capacity() -> None:
    sim = _sim(red_recovery=90.0)
    sim.damage_state[0].reservoir_current = 99.0
    sim._apply_between_round_recovery(1)
    assert sim.damage_state[0].reservoir_current <= sim.damage_state[0].reservoir_capacity


def test_recovery_does_not_modify_fsr_trait_values() -> None:
    sim = _sim(red_recovery=70.0)
    before = float(sim.fighters[0]["recovery_ability"])
    sim.damage_state[0].reservoir_current = 50.0
    sim._apply_between_round_recovery(1)
    assert float(sim.fighters[0]["recovery_ability"]) == before
