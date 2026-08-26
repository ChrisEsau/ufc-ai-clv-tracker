from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experimental import fsr_static_mc_ko_tko_v1 as ko


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


def test_more_acute_shock_increases_finish_probability() -> None:
    sim = ko.StaticFSRMCKOTKOV1(_profile("a"), _profile("b"), rounds=1, seed=1)

    low = sim._ko_probability(
        1,
        1.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )
    high = sim._ko_probability(
        1,
        10.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )

    assert high > low


def test_depleted_reservoir_increases_finish_probability() -> None:
    sim = ko.StaticFSRMCKOTKOV1(_profile("a"), _profile("b"), rounds=1, seed=2)
    fresh = sim._ko_probability(
        1,
        3.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )

    sim.damage_state[1].reservoir_current = 20.0
    depleted = sim._ko_probability(
        1,
        3.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )

    assert depleted > fresh


def test_knockdown_and_recent_kd_raise_finish_hazard() -> None:
    sim = ko.StaticFSRMCKOTKOV1(_profile("a"), _profile("b"), rounds=1, seed=3)

    normal = sim._ko_probability(
        1,
        4.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )
    current_kd = sim._ko_probability(
        1,
        4.0,
        knockdown_on_strike=True,
        recent_kd_before=False,
    )
    hurt_followup = sim._ko_probability(
        1,
        4.0,
        knockdown_on_strike=True,
        recent_kd_before=True,
    )

    assert current_kd > normal
    assert hurt_followup > current_kd


def test_zero_reservoir_is_dangerous_but_not_deterministic() -> None:
    sim = ko.StaticFSRMCKOTKOV1(_profile("a"), _profile("b"), rounds=1, seed=4)
    sim.damage_state[1].reservoir_current = 0.0

    p = sim._ko_probability(
        1,
        3.0,
        knockdown_on_strike=True,
        recent_kd_before=True,
    )

    assert 0.0 < p < 1.0


def test_catastrophic_finish_hazard_exists_above_zero_reservoir() -> None:
    sim = ko.StaticFSRMCKOTKOV1(_profile("a"), _profile("b"), rounds=1, seed=5)
    assert sim.damage_state[1].reservoir_fraction == 1.0

    ordinary = sim._ko_probability(
        1,
        1.0,
        knockdown_on_strike=False,
        recent_kd_before=False,
    )
    bomb = sim._ko_probability(
        1,
        20.0,
        knockdown_on_strike=True,
        recent_kd_before=False,
    )

    assert bomb > ordinary
    assert bomb > 0.0
    assert sim.damage_state[1].reservoir_fraction == 1.0


def test_segment_finish_resolution_returns_one_competing_winner() -> None:
    sim = ko.StaticFSRMCKOTKOV1(_profile("red"), _profile("blue"), rounds=1, seed=6)
    sim._segment_finish_candidates = [
        {
            "attacker": 0,
            "defender": 1,
            "probability": 0.99,
            "strike_damage": 8.0,
            "shock_fraction": 0.08,
            "reservoir_fraction_after": 0.70,
            "knockdown_on_strike": True,
            "recent_kd_before": False,
        },
        {
            "attacker": 1,
            "defender": 0,
            "probability": 0.99,
            "strike_damage": 7.0,
            "shock_fraction": 0.07,
            "reservoir_fraction_after": 0.72,
            "knockdown_on_strike": True,
            "recent_kd_before": False,
        },
    ]

    finish = sim._resolve_segment_finish()

    assert finish is not None
    assert {finish.winner, finish.loser} == {0, 1}
    assert finish.winner != finish.loser


def test_run_truncates_path_when_stoppage_is_forced() -> None:
    sim = ko.StaticFSRMCKOTKOV1(_profile("red"), _profile("blue"), rounds=3, seed=7)

    original_generate = sim._generate_striking

    def forced_generate(phase: str) -> list[str]:
        notes = original_generate(phase)
        if sim.finish is None:
            sim.finish = ko.FinishResult(
                winner=0,
                loser=1,
                method="KO/TKO",
                probability=1.0,
                strike_damage=10.0,
                shock_fraction=0.10,
                reservoir_fraction_after=0.50,
                knockdown_on_strike=True,
                recent_kd_before=False,
            )
        return notes

    sim._generate_striking = forced_generate
    path = sim.run()

    assert path.finish is not None
    assert len(path.events) == 1
    assert path.events[0]["finish"] is True
    assert path.finish.round == 1
    assert path.finish.segment == 1
