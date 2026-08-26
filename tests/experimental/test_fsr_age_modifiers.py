from __future__ import annotations

import pandas as pd
import pytest

from scripts.experimental import fsr_age_modifiers as age


def test_legacy_kd_and_durability_curve_is_preserved_from_yaml() -> None:
    for trait in ("knockdown_resistance", "damage_durability"):
        assert age.trait_age_modifier(trait, 25.0) == pytest.approx(0.0)
        assert age.trait_age_modifier(trait, 30.0) == pytest.approx(0.0)
        assert age.trait_age_modifier(trait, 35.0) == pytest.approx(-10.0)
        assert age.trait_age_modifier(trait, 40.0) == pytest.approx(-20.0)


def test_uncalibrated_candidates_are_inert() -> None:
    assert age.trait_age_modifier("striking_power", 40.0) == pytest.approx(0.0)
    assert age.trait_age_modifier("control_imposition", 40.0) == pytest.approx(0.0)
    assert age.trait_age_modifier("distance_striking_precision", 40.0) == pytest.approx(0.0)


def test_apply_age_modifiers_does_not_mutate_stored_profile() -> None:
    stored = pd.Series(
        {
            "knockdown_resistance": 60.0,
            "damage_durability": 55.0,
            "striking_power": 70.0,
        }
    )
    before = stored.copy(deep=True)

    effective, applied = age.apply_age_modifiers(stored, 35.0)

    pd.testing.assert_series_equal(stored, before)
    assert effective["knockdown_resistance"] == pytest.approx(50.0)
    assert effective["damage_durability"] == pytest.approx(45.0)
    assert effective["striking_power"] == pytest.approx(70.0)
    assert applied == {
        "knockdown_resistance": pytest.approx(-10.0),
        "damage_durability": pytest.approx(-10.0),
    }


def test_enabled_calibrated_traits_are_explicit() -> None:
    assert age.enabled_calibrated_traits() == (
        "knockdown_resistance",
        "damage_durability",
    )
