import math

import pytest

from pipeline.simulation.event_mc_v1.components.formulas import (
    DISTANCE_TD_ATTEMPT_BASE_10S,
    clinch_entry_interval_probability,
    interval_hazard_per_second,
    legacy_td_attempt_interval_probability,
    strike_attempt_rate_per_second,
    strike_landing_probability,
    td_attempt_interval_probability,
    td_attempt_rate_per_second,
    td_success_probability,
)
from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile


def profile(**changes) -> FighterProfile:
    values = dict(
        fighter_id="fighter",
        fighter_name="Fighter",
        distance_striking_pressure=50.0,
        distance_striking_precision=50.0,
        distance_striking_defense=50.0,
        clinch_striking_pressure=50.0,
        wrestling_entry=50.0,
        wrestling_conversion=50.0,
        td_defense=50.0,
        control_imposition=50.0,
    )
    values.update(changes)
    return FighterProfile(**values)


def test_entry_monotonically_controls_intrinsic_probability_and_rate() -> None:
    probabilities = [
        td_attempt_interval_probability(profile(wrestling_entry=value))
        for value in (40, 50, 60)
    ]
    rates = [interval_hazard_per_second(value) for value in probabilities]
    assert probabilities == sorted(probabilities)
    assert rates == sorted(rates)
    assert len(set(probabilities)) == 3


@pytest.mark.parametrize(
    ("trait", "value"),
    [
        ("control_imposition", 80),
        ("distance_striking_pressure", 80),
        ("clinch_striking_pressure", 80),
    ],
)
def test_non_entry_traits_do_not_change_intrinsic_initiation(trait, value) -> None:
    assert td_attempt_interval_probability(profile(**{trait: value})) == (
        td_attempt_interval_probability(profile())
    )


def test_conversion_changes_success_but_not_initiation() -> None:
    low = profile(wrestling_conversion=35)
    high = profile(wrestling_conversion=65)
    defender = profile()
    assert td_attempt_interval_probability(low) == td_attempt_interval_probability(high)
    assert td_success_probability(low, defender) < td_success_probability(high, defender)


def test_opponent_defense_changes_success_but_not_initiation() -> None:
    attacker = profile()
    low = profile(td_defense=35)
    high = profile(td_defense=65)
    assert td_attempt_interval_probability(attacker) == td_attempt_interval_probability(attacker)
    assert td_success_probability(attacker, low) > td_success_probability(attacker, high)


def test_neutral_entry_uses_existing_base_and_hazard_round_trip() -> None:
    probability = td_attempt_interval_probability(profile(wrestling_entry=50))
    assert probability == DISTANCE_TD_ATTEMPT_BASE_10S
    hazard = interval_hazard_per_second(probability)
    assert 1.0 - math.exp(-hazard * 10) == pytest.approx(probability)
    assert td_attempt_rate_per_second(profile()) == pytest.approx(hazard)
    assert td_attempt_rate_per_second(
        profile(), context_multiplier=1.0
    ) == pytest.approx(hazard)


def test_legacy_comparator_remains_blended() -> None:
    neutral = profile()
    assert legacy_td_attempt_interval_probability(
        profile(control_imposition=65)
    ) > legacy_td_attempt_interval_probability(neutral)
    assert legacy_td_attempt_interval_probability(
        profile(distance_striking_pressure=65)
    ) < legacy_td_attempt_interval_probability(neutral)


def test_strike_and_clinch_formulas_are_unchanged_by_unrelated_td_traits() -> None:
    baseline = profile()
    changed = profile(wrestling_conversion=70, td_defense=30, control_imposition=65)
    defender = profile()
    assert strike_attempt_rate_per_second(changed) == strike_attempt_rate_per_second(baseline)
    assert strike_landing_probability(changed, defender) == strike_landing_probability(baseline, defender)
    # control and conversion/defense are absent from the Phase 2A clinch consumer.
    assert clinch_entry_interval_probability(changed) == clinch_entry_interval_probability(baseline)
