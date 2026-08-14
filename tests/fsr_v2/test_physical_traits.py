import pandas as pd
import pytest

from pipeline.fsr_v2.physical import (
    PHYSICAL_COLUMNS,
    STAMINA_CAPACITY,
    attach_physical_latest,
    attach_physical_prefight,
)


def _physical_row(*, fight_id: str = "f1", fighter_id: str = "a") -> dict:
    return {
        "fight_id": fight_id,
        "fighter_id": fighter_id,
        "stamina_capacity": 100.0,
        "stamina_depletion_resistance": 55.0,
        "stamina_performance_resilience": 57.0,
        "striking_power": 61.0,
        "damage_durability": 63.0,
        "knockdown_resistance": 65.0,
    }


def test_physical_contract_preserves_fsr32_simulator_fields():
    assert STAMINA_CAPACITY == 100.0
    assert PHYSICAL_COLUMNS == (
        "stamina_capacity",
        "stamina_depletion_resistance",
        "stamina_performance_resilience",
        "striking_power",
        "damage_durability",
        "knockdown_resistance",
    )


def test_attach_physical_prefight_adds_all_fields_one_to_one():
    core = pd.DataFrame(
        [{"fight_id": "f1", "fighter_id": "a", "standing_striking_tendency": 0.2}]
    )
    physical = pd.DataFrame([_physical_row()])

    out = attach_physical_prefight(core, physical)

    assert len(out) == 1
    for column in PHYSICAL_COLUMNS:
        assert column in out.columns
    assert out.loc[0, "striking_power"] == 61.0
    assert out.loc[0, "stamina_capacity"] == 100.0


def test_attach_physical_prefight_rejects_missing_or_extra_keys():
    core = pd.DataFrame(
        [{"fight_id": "f1", "fighter_id": "a", "standing_striking_tendency": 0.2}]
    )
    physical = pd.DataFrame([_physical_row(fight_id="other")])

    with pytest.raises(RuntimeError, match="core/physical prefight key mismatch"):
        attach_physical_prefight(core, physical)


def test_attach_physical_latest_adds_all_fields_one_to_one():
    core = pd.DataFrame(
        [{"fighter_id": "a", "fighter_name": "A", "standing_striking_tendency": 0.2}]
    )
    physical = pd.DataFrame(
        [{"fighter_id": "a", **{k: v for k, v in _physical_row().items() if k in PHYSICAL_COLUMNS}}]
    )

    out = attach_physical_latest(core, physical)

    assert len(out) == 1
    for column in PHYSICAL_COLUMNS:
        assert column in out.columns
    assert out.loc[0, "damage_durability"] == 63.0


def test_attach_physical_latest_rejects_fighter_mismatch():
    core = pd.DataFrame(
        [{"fighter_id": "a", "fighter_name": "A", "standing_striking_tendency": 0.2}]
    )
    physical = pd.DataFrame(
        [{"fighter_id": "b", **{k: v for k, v in _physical_row().items() if k in PHYSICAL_COLUMNS}}]
    )

    with pytest.raises(RuntimeError, match="core/physical latest fighter mismatch"):
        attach_physical_latest(core, physical)
