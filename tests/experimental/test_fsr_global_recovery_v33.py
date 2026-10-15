from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_static_mc_ko_tko_v3_stamina as v3
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33


def test_fsr32_has_no_recovery_fields() -> None:
    assert "recovery_ability" in fsr32.REMOVED_RECOVERY_COLUMNS
    assert "stamina_recovery_ability" in fsr32.REMOVED_RECOVERY_COLUMNS
    assert "stamina_recovery_ability" not in fsr32.STAMINA_COLUMNS


def test_builder_removes_recovery_fields() -> None:
    source = pd.DataFrame([{
        "fight_id": "f1",
        "fighter_id": "x1",
        "fatigue_accumulation_resistance": 48.0,
        "fatigue_performance_resilience": 52.0,
        "recovery_ability": 77.0,
        "stamina_recovery_ability": 77.0,
    }])
    built = fsr32.build_fsr_32_database(source)
    assert "recovery_ability" not in built.columns
    assert "stamina_recovery_ability" not in built.columns
    assert built.loc[0, fsr32.STAMINA_CAPACITY] == pytest.approx(100.0)


def test_v33_global_round_recovery() -> None:
    sim = object.__new__(v33.StaticFSRMCKOTKOV33GlobalRecovery)
    sim.damage_state = [
        SimpleNamespace(reservoir_capacity=100.0, reservoir_current=50.0),
        SimpleNamespace(reservoir_capacity=100.0, reservoir_current=80.0),
    ]
    sim.total_round_recovery = [0.0, 0.0]
    sim.round_recovery_events = []
    sim.stamina_state = [
        v3.StaminaState(capacity=100.0, current=50.0),
        v3.StaminaState(capacity=100.0, current=80.0),
    ]
    sim.total_stamina_recovered = [0.0, 0.0]
    sim.stamina_round_events = []

    sim._apply_between_round_recovery(1)

    assert sim.damage_state[0].reservoir_current == pytest.approx(60.0)
    assert sim.damage_state[1].reservoir_current == pytest.approx(84.0)
    assert sim.stamina_state[0].current == pytest.approx(70.0)
    assert sim.stamina_state[1].current == pytest.approx(88.0)


def test_v33_global_recovery_constants() -> None:
    assert v33.GLOBAL_DAMAGE_RECOVERY_FRACTION == pytest.approx(0.20)
    assert v33.GLOBAL_STAMINA_RECOVERY_FRACTION == pytest.approx(0.40)
