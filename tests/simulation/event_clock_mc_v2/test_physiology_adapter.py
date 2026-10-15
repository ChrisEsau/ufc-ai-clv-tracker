from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.fsr_v3.active_config import ActiveTraitConfig
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import (
    FROZEN_KD_POWER_BETA,
    fighter_mechanics_from_prefight,
    legacy_kdres_equivalent,
    legacy_power_equivalent,
)


def runtime():
    return SimpleNamespace(
        standing_accuracy=0.61,
        takedown_completion=0.42,
        ground_accuracy=0.57,
    )


def row(**updates):
    values = {
        "event_date": "2020-01-01",
        "fight_id": "fight-1",
        "fighter_id": "fighter-1",
        "striking_power_v3": 0.20,
        "damage_durability": 63.0,
        "knockdown_resistance_v3": 0.15,
        "stamina_capacity": 100.0,
        "stamina_depletion_resistance": 58.0,
    }
    values.update(updates)
    return values


def test_canonical_transforms_preserve_frozen_linear_predictors():
    power = np.array([-0.2, 0.0, 0.3])
    transformed = legacy_power_equivalent(power)
    np.testing.assert_allclose((transformed - 50.0) * FROZEN_KD_POWER_BETA, power)

    resistance = np.array([-0.1, 0.0, 0.25])
    beta = ActiveTraitConfig().frozen_event_clock_kdres_beta
    transformed = legacy_kdres_equivalent(resistance)
    np.testing.assert_allclose(-(transformed - 50.0) * beta, resistance)


def test_prefight_adapter_keeps_fighter_specific_physiology_separate():
    first = fighter_mechanics_from_prefight(row(), runtime())
    second = fighter_mechanics_from_prefight(
        row(
            fighter_id="fighter-2",
            striking_power_v3=-0.1,
            damage_durability=41.0,
            knockdown_resistance_v3=-0.08,
            stamina_depletion_resistance=72.0,
        ),
        runtime(),
    )
    assert first.striking_power != second.striking_power
    assert first.damage_durability != second.damage_durability
    assert first.knockdown_resistance != second.knockdown_resistance
    assert first.stamina_depletion_resistance != second.stamina_depletion_resistance
    assert first.stamina_capacity == second.stamina_capacity == 100.0


@pytest.mark.parametrize("missing", ["event_date", "fight_id", "fighter_id", "striking_power_v3"])
def test_prefight_adapter_fails_loudly_without_exact_historical_key_or_trait(missing):
    values = row()
    del values[missing]
    with pytest.raises(ValueError, match="missing physiology columns"):
        fighter_mechanics_from_prefight(values, runtime())


def test_prefight_adapter_rejects_noncanonical_capacity_and_nonfinite_values():
    with pytest.raises(ValueError, match="must remain 100"):
        fighter_mechanics_from_prefight(row(stamina_capacity=90), runtime())
    with pytest.raises(ValueError, match="non-finite"):
        fighter_mechanics_from_prefight(row(damage_durability=np.nan), runtime())


def test_finite_negative_synthetic_coordinates_are_valid():
    mechanics = fighter_mechanics_from_prefight(
        row(striking_power_v3=-2.0, knockdown_resistance_v3=-2.0), runtime()
    )
    assert mechanics.striking_power < 0
    assert mechanics.knockdown_resistance < 0


@pytest.mark.parametrize("field", ["striking_power", "knockdown_resistance"])
def test_nonfinite_synthetic_coordinates_still_fail(field):
    from dataclasses import replace

    mechanics = fighter_mechanics_from_prefight(row(), runtime())
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        replace(mechanics, **{field: float("inf")})
