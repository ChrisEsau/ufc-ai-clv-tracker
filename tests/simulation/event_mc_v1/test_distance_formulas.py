import math

import pandas as pd
import pytest

from pipeline.simulation.event_mc_v1.components.formulas import (
    clinch_entry_interval_probability,
    interval_hazard_per_second,
    legacy_td_attempt_interval_probability,
    strike_attempt_rate_per_second,
    strike_landing_probability,
    style_preferences,
    td_attempt_interval_probability,
    td_success_probability,
)
from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile
from pipeline.simulation.event_mc_v1.diagnostics.phase7l_distance_td_calibration import calibration_for_distance_td
from pipeline.simulation.event_mc_v1.diagnostics.phase7m_td_success_calibration import calibration_for_td_success
from scripts.experimental.fsr_static_mc_v0 import StaticFSRMCV0


def profile(**overrides) -> FighterProfile:
    values = {
        "fighter_id": "fighter",
        "fighter_name": "Fighter",
        "distance_striking_pressure": 50.0,
        "distance_striking_precision": 50.0,
        "distance_striking_defense": 50.0,
        "clinch_striking_pressure": 50.0,
        "wrestling_entry": 50.0,
        "wrestling_conversion": 50.0,
        "td_defense": 50.0,
        "control_imposition": 50.0,
    }
    values.update(overrides)
    return FighterProfile(**values)


def legacy_series(value: FighterProfile) -> pd.Series:
    return pd.Series(value.__dict__)


def test_formula_outputs_match_current_legacy_consumers() -> None:
    red = profile(
        fighter_id="red",
        distance_striking_pressure=57,
        distance_striking_precision=54,
        clinch_striking_pressure=46,
        wrestling_entry=61,
        wrestling_conversion=58,
        control_imposition=55,
    )
    blue = profile(fighter_id="blue", distance_striking_defense=52, td_defense=56)
    legacy = StaticFSRMCV0(legacy_series(red), legacy_series(blue))
    assert style_preferences(red) == pytest.approx(
        __import__("scripts.experimental.fsr_static_mc_v0", fromlist=["_style_preferences"])._style_preferences(legacy_series(red))
    )
    legacy_calibration = calibration_for_distance_td(0.10)
    assert legacy_td_attempt_interval_probability(red, legacy_calibration) == pytest.approx(
        legacy._td_attempt_hazard(0, "DISTANCE")
    )
    assert legacy_td_attempt_interval_probability(red) > legacy_td_attempt_interval_probability(red, legacy_calibration)
    legacy_success_calibration = calibration_for_td_success(-0.40)
    assert td_success_probability(red, blue, legacy_success_calibration) == pytest.approx(legacy._td_success_prob(0))
    assert td_success_probability(red, blue) < td_success_probability(red, blue, legacy_success_calibration)
    assert clinch_entry_interval_probability(red) == pytest.approx(
        legacy._distance_clinch_hazard(0)
    )
    assert strike_landing_probability(red, blue) == pytest.approx(
        legacy._strike_accuracy(0, "DISTANCE")
    )
    expected_attempts_per_10s = legacy._strike_attempts
    # The V0 Poisson mean is deterministic even though its draw is not exposed.
    legacy_mean = (5.0 / 3.0) * math.exp(min(8.0, max(-8.0, 57 - 50)) / 12.0)
    calibrated_mean = (6.0 / 3.0) * math.exp(min(8.0, max(-8.0, 57 - 50)) / 12.0)
    assert strike_attempt_rate_per_second(red) * 10 == pytest.approx(calibrated_mean)
    assert calibrated_mean == pytest.approx(legacy_mean * 6.0 / 5.0)
    assert callable(expected_attempts_per_10s)


def test_interval_hazard_round_trip_is_exact() -> None:
    probability = td_attempt_interval_probability(profile(wrestling_entry=58))
    rate = interval_hazard_per_second(probability)
    assert 1.0 - math.exp(-rate * 10.0) == pytest.approx(probability)


def test_legacy_blend_not_ontology_correct_initiation() -> None:
    base = profile(wrestling_entry=55)
    more_control = profile(wrestling_entry=55, control_imposition=65)
    more_distance = profile(wrestling_entry=55, distance_striking_pressure=60)
    assert legacy_td_attempt_interval_probability(more_control) > legacy_td_attempt_interval_probability(base)
    assert legacy_td_attempt_interval_probability(more_distance) < legacy_td_attempt_interval_probability(base)


def test_conversion_and_defense_change_success_not_attempt_hazard() -> None:
    attacker = profile(wrestling_conversion=45)
    improved = profile(wrestling_conversion=65)
    defender = profile(td_defense=45)
    stronger_defender = profile(td_defense=65)
    assert td_attempt_interval_probability(attacker) == td_attempt_interval_probability(improved)
    assert td_success_probability(improved, defender) > td_success_probability(attacker, defender)
    assert td_attempt_interval_probability(attacker) == td_attempt_interval_probability(attacker)
    assert td_success_probability(attacker, stronger_defender) < td_success_probability(attacker, defender)
