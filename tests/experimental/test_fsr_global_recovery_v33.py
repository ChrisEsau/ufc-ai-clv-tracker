from __future__ import annotations

import pandas as pd
import pytest

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_ko_tko_v3_stamina as v3
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33


def test_fsr32_stamina_contract_no_longer_contains_recovery_alias() -> None:
    assert "stamina_recovery_ability" not in fsr32.STAMINA_COLUMNS
    assert "recovery_ability" not in fsr32.STAMINA_SOURCE_COLUMNS.values()


def test_builder_drops_deprecated_stamina_recovery_column() -> None:
    source = pd.DataFrame(
        [
            {
                "fight_id": "fight-1",
                "fighter_id": "fighter-1",
                "fatigue_accumulation_resistance": 48.0,
                "fatigue_performance_resilience": 52.0,
                "recovery_ability": 77.0,
                "stamina_recovery_ability": 77.0,
            }
        ]
    )

    built = fsr32.build_fsr_32_database(source)

    assert "stamina_recovery_ability" not in built.columns
    assert built.loc[0, fsr32.STAMINA_CAPACITY] == pytest.approx(100.0)
    assert built.loc[0, fsr32.STAMINA_DEPLETION_RESISTANCE] == pytest.approx(48.0)
    assert built.loc[0, fsr32.STAMINA_PERFORMANCE_RESILIENCE] == pytest.approx(52.0)


def test_v33_restores_40_percent_of_missing_stamina(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate stamina recovery from the independent locked damage-recovery system.
    monkeypatch.setattr(
        recovery.StaticFSRMCKOTKOV2RoundRecovery,
        "_apply_between_round_recovery",
        lambda self, completed_round: None,
    )

    sim = object.__new__(v33.StaticFSRMCKOTKOV33GlobalRecovery)
    sim.stamina_state = [
        v3.StaminaState(capacity=100.0, current=50.0),
        v3.StaminaState(capacity=100.0, current=80.0),
    ]
    sim.total_stamina_recovered = [0.0, 0.0]
    sim.stamina_round_events = []

    sim._apply_between_round_recovery(completed_round=1)

    # 40% of missing: 50 + .40*50 = 70; 80 + .40*20 = 88.
    assert sim.stamina_state[0].current == pytest.approx(70.0)
    assert sim.stamina_state[1].current == pytest.approx(88.0)
    assert sim.total_stamina_recovered == pytest.approx([20.0, 8.0])
    assert all(event["recovery_mode"] == "global" for event in sim.stamina_round_events)
    assert all(
        event["fraction_of_missing"] == pytest.approx(0.40)
        for event in sim.stamina_round_events
    )


def test_v33_global_recovery_fraction_is_locked_candidate() -> None:
    assert v33.GLOBAL_STAMINA_RECOVERY_FRACTION == pytest.approx(0.40)
