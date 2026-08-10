from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_static_mc_ko_tko_v3_stamina as stamina


def _fsr28_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fight_id": ["f1"],
            "fighter_id": ["a"],
            "fatigue_accumulation_resistance": [62.0],
            "fatigue_performance_resilience": [58.0],
            "recovery_ability": [64.0],
        }
    )


def test_fsr32_copies_dynamic_cardio_into_explicit_stamina_contract() -> None:
    out = fsr32.build_fsr_32_database(_fsr28_fixture())
    row = out.iloc[0]
    assert row[fsr32.STAMINA_CAPACITY] == fsr32.DEFAULT_STAMINA_CAPACITY
    assert row[fsr32.STAMINA_DEPLETION_RESISTANCE] == 62.0
    assert row[fsr32.STAMINA_PERFORMANCE_RESILIENCE] == 58.0
    assert row[fsr32.STAMINA_RECOVERY_ABILITY] == 64.0


def test_fsr32_rejects_missing_source_trait() -> None:
    frame = _fsr28_fixture().drop(columns=["recovery_ability"])
    with pytest.raises(RuntimeError, match="missing required stamina-source"):
        fsr32.build_fsr_32_database(frame)


def test_strict_stamina_reader_rejects_missing_and_nonfinite_values() -> None:
    profile = pd.Series({fsr32.STAMINA_CAPACITY: 100.0})
    with pytest.raises(ValueError, match="missing required fighter parameter"):
        stamina._strict_profile_float(profile, fsr32.STAMINA_RECOVERY_ABILITY)

    profile[fsr32.STAMINA_RECOVERY_ABILITY] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        stamina._strict_profile_float(profile, fsr32.STAMINA_RECOVERY_ABILITY)


def test_power_curve_is_designed_to_decay_more_than_output() -> None:
    # Contract-level guard on the intended architecture: the minimum retained
    # power is below the minimum retained output for both resilience extremes.
    assert stamina.POWER_FLOOR_LOW_RESILIENCE < stamina.OUTPUT_FLOOR_LOW_RESILIENCE
    assert stamina.POWER_FLOOR_HIGH_RESILIENCE < stamina.OUTPUT_FLOOR_HIGH_RESILIENCE
    assert stamina.POWER_EXPONENT_LOW_RESILIENCE > stamina.OUTPUT_EXPONENT_LOW_RESILIENCE
    assert stamina.POWER_EXPONENT_HIGH_RESILIENCE > stamina.OUTPUT_EXPONENT_HIGH_RESILIENCE


def test_fresh_power_translation_is_stronger_than_damage_v1() -> None:
    from scripts.experimental import fsr_static_mc_damage_v1 as damage

    assert stamina.STAMINA_POWER_TAIL_RATING_SCALE < damage.POWER_TAIL_RATING_SCALE
    assert stamina.STAMINA_TAIL_MAGNITUDE_POWER_SCALE < damage.TAIL_MAGNITUDE_POWER_SCALE
