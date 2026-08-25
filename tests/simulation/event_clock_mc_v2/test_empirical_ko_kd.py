from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import EventClockShadowKOKDModel
from pipeline.simulation.event_mc_v1.components.profiles import (
    FighterProfile,
    MatchupProfiles,
    Side as V1Side,
)
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import (
    FightPhysiology,
    FightState,
    FighterPhysiology,
    Side,
)
from pipeline.simulation.event_clock_mc_v2.engine.causal_engine import EngineRNGs
from pipeline.simulation.event_clock_mc_v2.mechanics.config import FighterMechanics
from pipeline.simulation.event_clock_mc_v2.mechanics.ko_kd_empirical import (
    kd_probability,
    ko_probability,
    resolve_landed_strike,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.physiology import (
    apply_action_consequence,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import StrikeConsequence
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import (
    age_years_on_date,
    fighter_mechanics_from_prefight,
)


def fighter(*, power=50.0, kdres=50.0, age=30.0, durability=50.0):
    return FighterMechanics(
        1, 1, 1, 0, 0.4, 0.3, power, durability, kdres, 100, 50, age
    )


def state(*, time=0.0, red_stamina=1.0, blue_kds=0, trauma=0.0, acute=0.0):
    return FightState(
        fight_time_seconds=time,
        physiology=FightPhysiology(
            FighterPhysiology(stamina=red_stamina),
            FighterPhysiology(
                cumulative_trauma=trauma,
                acute_vulnerability=acute,
                knockdowns_suffered=blue_kds,
            ),
        ),
    )


class ScriptRNG:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.values.pop(0)


def probabilities(a=None, d=None, s=None, prior=0):
    a = a or fighter()
    d = d or fighter()
    s = s or state()
    kwargs = dict(
        prior_defender_kds=prior,
        elapsed_seconds=s.fight_time_seconds,
        attacker_stamina=s.physiology.red.stamina,
    )
    return ko_probability(a, d, **kwargs), kd_probability(a, d, **kwargs)


def test_ko_is_first_and_mutually_exclusive():
    rng = ScriptRNG([0.0, 0.0])
    result = resolve_landed_strike(
        state=state(),
        attacker_side=Side.RED,
        attacker=fighter(),
        defender=fighter(),
        rng=rng,
    )
    assert result.ko_tko and not result.knockdown and result.kd_probability == 0.0
    assert rng.calls == 1


def test_kd_is_rolled_only_after_ko_survival_and_increments_future_state():
    rng = ScriptRNG([1.0, 0.0])
    result = resolve_landed_strike(
        state=state(),
        attacker_side=Side.RED,
        attacker=fighter(),
        defender=fighter(),
        rng=rng,
    )
    assert not result.ko_tko and result.knockdown and rng.calls == 2
    consequence = StrikeConsequence(
        True,
        knockdown=result.knockdown,
        knockdown_probability=result.kd_probability,
        ko_probability=result.ko_probability,
        prior_defender_kds=result.prior_defender_kds,
    )
    updated = apply_action_consequence(
        state(), Side.RED, ActionFamily.STAND_ATTACK, consequence, fighter()
    )
    assert updated.physiology.blue.knockdowns_suffered == 1
    assert result.prior_defender_kds == 0


def test_prior_kd_changes_future_not_current_ko():
    fresh = probabilities(prior=0)
    hurt = probabilities(prior=1)
    assert hurt[0] > fresh[0] and hurt[1] > fresh[1]
    # The current strike's KD draw cannot feed the already-computed KO value.
    result = resolve_landed_strike(
        state=state(),
        attacker_side=Side.RED,
        attacker=fighter(),
        defender=fighter(),
        rng=ScriptRNG([1.0, 0.0]),
    )
    assert result.ko_probability == fresh[0] and result.prior_defender_kds == 0


def test_prior_kd_state_uses_defender_direction():
    directional = FightState(
        physiology=FightPhysiology(
            FighterPhysiology(knockdowns_suffered=3),
            FighterPhysiology(knockdowns_suffered=1),
        )
    )
    red = resolve_landed_strike(
        state=directional,
        attacker_side=Side.RED,
        attacker=fighter(),
        defender=fighter(),
        rng=ScriptRNG([1.0, 1.0]),
    )
    blue = resolve_landed_strike(
        state=directional,
        attacker_side=Side.BLUE,
        attacker=fighter(),
        defender=fighter(),
        rng=ScriptRNG([1.0, 1.0]),
    )
    assert red.prior_defender_kds == 1
    assert blue.prior_defender_kds == 3


def test_validated_predictor_directions_and_zero_terms():
    neutral = probabilities()
    high_power = probabilities(a=fighter(power=60))
    old_attacker = probabilities(a=fighter(age=40))
    old_defender = probabilities(d=fighter(age=40))
    high_res = probabilities(d=fighter(kdres=60))
    late = probabilities(s=state(time=600))
    tired = probabilities(s=state(red_stamina=0.05))
    assert high_power[0] > neutral[0] and high_power[1] > neutral[1]
    assert old_attacker[0] < neutral[0] and old_attacker[1] < neutral[1]
    assert old_defender[0] > neutral[0] and old_defender[1] > neutral[1]
    assert high_res[0] == neutral[0] and high_res[1] < neutral[1]
    assert late[0] == neutral[0] and late[1] < neutral[1]
    assert tired == neutral


def test_durability_trauma_and_acute_do_not_enter_hazards():
    neutral = probabilities()
    assert probabilities(d=fighter(durability=500)) == neutral
    assert probabilities(s=state(trauma=10000, acute=100)) == neutral


def _v1_profile(name, power, kdres, age):
    return FighterProfile(
        name,
        name,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        striking_power=power,
        knockdown_resistance=kdres,
        age_years=age,
    )


@pytest.mark.parametrize(
    "power,kdres,a_age,d_age,prior,seconds",
    [
        (50, 50, 30, 30, 0, 0),
        (60, 50, 30, 30, 0, 0),
        (40, 50, 30, 30, 0, 0),
        (50, 65, 30, 30, 0, 0),
        (50, 50, 30, 42, 0, 0),
        (50, 50, 23, 30, 0, 0),
        (50, 50, 30, 30, 1, 0),
        (50, 50, 30, 30, 2, 0),
        (50, 50, 30, 30, 0, 900),
    ],
)
def test_formula_parity_with_read_only_v1(power, kdres, a_age, d_age, prior, seconds):
    a = fighter(power=power, age=a_age)
    d = fighter(kdres=kdres, age=d_age)
    v1 = EventClockShadowKOKDModel(
        MatchupProfiles(
            _v1_profile("a", power, 50, a_age), _v1_profile("d", 50, kdres, d_age)
        )
    )
    v1_state = SimpleNamespace(
        fight_time_seconds=seconds, red_stamina=1.0, blue_stamina=1.0
    )
    assert ko_probability(
        a, d, prior_defender_kds=prior, elapsed_seconds=seconds, attacker_stamina=1
    ) == pytest.approx(
        v1.ko_probability(
            state=v1_state, attacker=V1Side.RED, prior_defender_kds=prior
        ),
        abs=1e-15,
    )
    assert kd_probability(
        a, d, prior_defender_kds=prior, elapsed_seconds=seconds, attacker_stamina=1
    ) == pytest.approx(
        v1.kd_probability(
            state=v1_state, attacker=V1Side.RED, prior_defender_kds=prior
        ),
        abs=1e-15,
    )


def test_sixth_stream_preserves_first_five_streams():
    old = [
        np.random.default_rng(s).random(5) for s in np.random.SeedSequence(123).spawn(5)
    ]
    new = EngineRNGs.from_seed(123)
    current = [
        new.red_timing,
        new.blue_timing,
        new.red_selection,
        new.blue_selection,
        new.mechanics,
    ]
    for expected, rng in zip(old, current):
        assert np.array_equal(expected, rng.random(5))


def test_exact_fight_date_age_is_separate_from_power():
    age = age_years_on_date("1990-06-15", "2020-06-15")
    assert age == pytest.approx(
        (np.datetime64("2020-06-15") - np.datetime64("1990-06-15")).astype(int)
        / 365.2425
    )
    mechanics = fighter(power=63, age=age)
    assert mechanics.striking_power == 63 and mechanics.age_years == age
    assert mechanics.striking_power != 63 - 1.15 * (age - 30)


def test_prefight_power_and_resistance_are_not_age_translated():
    row = {
        "event_date": "2020-06-15",
        "fight_id": "bout",
        "fighter_id": "fighter",
        "striking_power_v3": 0.2,
        "damage_durability": 55.0,
        "knockdown_resistance_v3": 0.1,
        "stamina_capacity": 100.0,
        "stamina_depletion_resistance": 60.0,
    }
    runtime = SimpleNamespace(
        standing_accuracy=0.5,
        takedown_completion=0.4,
        ground_accuracy=0.6,
    )
    young = fighter_mechanics_from_prefight(row, runtime, age_years=22.0)
    old = fighter_mechanics_from_prefight(row, runtime, age_years=42.0)
    assert young.striking_power == old.striking_power
    assert young.knockdown_resistance == old.knockdown_resistance
    assert (young.age_years, old.age_years) == (22.0, 42.0)
