import numpy as np
import pytest

from pipeline.simulation.event_mc_v1.components.actions import ActionAttempt
from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile, MatchupProfiles, Side
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifiers
from pipeline.simulation.event_mc_v1.physiology import ImpactTraumaKnockdownModel, PhysiologyTimeAdvanceModel
from pipeline.simulation.event_mc_v1.state import FightState


def fighter(name, **changes):
    values = dict(fighter_id=name, fighter_name=name, distance_striking_pressure=50,
                  distance_striking_precision=50, distance_striking_defense=50,
                  clinch_striking_pressure=50, wrestling_entry=50,
                  wrestling_conversion=50, td_defense=50, control_imposition=50)
    values.update(changes)
    return FighterProfile(**values)


def model(**blue):
    return ImpactTraumaKnockdownModel(MatchupProfiles(fighter("red"), fighter("blue", **blue)))


def resolve(subject, state=None, seed=1, power=1.0, landed=True):
    payload = ActionAttempt(Side.RED, "strike", DynamicModifiers(1, power), landed)
    return subject.resolve(state or FightState(), payload, 10, np.random.default_rng(seed), np.random.default_rng(seed + 1))


def test_miss_creates_no_physiology_and_landed_strike_creates_exactly_one_trauma_increment():
    subject = model()
    delta, events = resolve(subject, landed=False)
    assert delta == delta.__class__() and events == ()
    delta, events = resolve(subject)
    assert delta.blue_cumulative_trauma > 0
    assert len(events) == 1


def test_power_and_pre_action_stamina_enter_once_through_impact():
    subject = model()
    full = resolve(subject, seed=7, power=1.0)[1][0].payload
    tired = resolve(subject, seed=7, power=0.5)[1][0].payload
    assert tired.impact == pytest.approx(full.impact * 0.5)
    assert tired.current_resistance > full.current_resistance  # only lower trauma differs


def test_durability_reduces_trauma_without_changing_same_seed_impact():
    low = resolve(model(damage_durability=30), seed=8)[1][0].payload
    high = resolve(model(damage_durability=70), seed=8)[1][0].payload
    assert low.impact == high.impact
    assert high.primary_trauma < low.primary_trauma


def test_resistance_trauma_and_acute_move_kd_probability_in_required_directions():
    base = resolve(model(), FightState(), seed=9)[1][0].payload
    resistant = resolve(model(knockdown_resistance=70), FightState(), seed=9)[1][0].payload
    traumatized = resolve(model(), FightState(blue_cumulative_trauma=80), seed=9)[1][0].payload
    acute = resolve(model(), FightState(blue_acute_vulnerability=1), seed=9)[1][0].payload
    assert resistant.knockdown_probability < base.knockdown_probability
    assert traumatized.knockdown_probability > base.knockdown_probability
    assert acute.knockdown_probability > base.knockdown_probability


def test_kd_adds_acute_once_is_nonterminal_and_trauma_never_recovers():
    subject = model(knockdown_resistance=10)
    class AlwaysKnockdown:
        def random(self):
            return 0.0

    payload = ActionAttempt(Side.RED, "strike", DynamicModifiers(1, 1), True)
    delta, events = subject.resolve(FightState(), payload, 10, np.random.default_rng(2), AlwaysKnockdown())
    assert events[0].payload.knockdown
    assert delta.blue_acute_vulnerability == pytest.approx(0.5)
    assert delta.finished is None
    state = FightState(blue_cumulative_trauma=12, blue_acute_vulnerability=1)
    advanced = PhysiologyTimeAdvanceModel().advance(state, None, 30)
    assert advanced.blue_acute_vulnerability == pytest.approx(0.5)
    assert advanced.blue_cumulative_trauma is None


def test_impact_and_kd_streams_are_deterministic_but_stochastic_across_seeds():
    subject = model()
    one = resolve(subject, seed=11)[1][0].payload
    repeat = resolve(subject, seed=11)[1][0].payload
    other = resolve(subject, seed=12)[1][0].payload
    assert one == repeat
    assert one.impact != other.impact


def test_attacker_age_translates_power_once_before_impact_without_clipping():
    young = ImpactTraumaKnockdownModel(
        MatchupProfiles(
            fighter("red", striking_power=70.0, age_years=30.0),
            fighter("blue"),
        )
    )
    old = ImpactTraumaKnockdownModel(
        MatchupProfiles(
            fighter("red", striking_power=70.0, age_years=40.0),
            fighter("blue"),
        )
    )

    young_out = resolve(young, seed=21)[1][0].payload
    old_out = resolve(old, seed=21)[1][0].payload

    rating_delta = -1.15 * (40.0 - 30.0)
    scale = young.calibration.section("damage")["power_rating_scale"]
    expected_ratio = np.exp(rating_delta / scale)

    assert old_out.impact == pytest.approx(
        young_out.impact * expected_ratio
    )
    assert old_out.primary_trauma == pytest.approx(
        young_out.primary_trauma * expected_ratio
    )
    assert old_out.impact < young_out.impact
