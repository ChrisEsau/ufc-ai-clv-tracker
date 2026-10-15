import pytest

from pipeline.simulation.event_mc_v1.components.action_rates import FightFlowRateProvider
from pipeline.simulation.event_mc_v1.components.actions import DistanceCandidate
from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile, MatchupProfiles, Side
from pipeline.simulation.event_mc_v1.config import FightConfig
from pipeline.simulation.event_mc_v1.contracts import FightContext
from pipeline.simulation.event_mc_v1.engine import SimulationEngine
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifierProvider
from pipeline.simulation.event_mc_v1.rng import RNGManager
from pipeline.simulation.event_mc_v1.stamina import StaminaModel, StaminaTimeAdvanceModel
from pipeline.simulation.event_mc_v1.state import FightState, Phase


def fighter(name, **changes):
    values = dict(fighter_id=name, fighter_name=name, distance_striking_pressure=50,
                  distance_striking_precision=50, distance_striking_defense=50,
                  clinch_striking_pressure=50, wrestling_entry=50,
                  wrestling_conversion=50, td_defense=50, control_imposition=50,
                  stamina_capacity=100, stamina_depletion_resistance=50,
                  stamina_performance_resilience=50)
    values.update(changes)
    return FighterProfile(**values)


def matchup(**red):
    return MatchupProfiles(fighter("red", **red), fighter("blue"))


def test_modifiers_are_full_at_fresh_and_monotonic_with_resilience():
    provider = DynamicModifierProvider()
    fresh = provider.modifiers(fighter("red"), FightState(), Side.RED)
    tired = provider.modifiers(fighter("red"), FightState(red_stamina=0.3), Side.RED)
    resilient = provider.modifiers(fighter("red", stamina_performance_resilience=80), FightState(red_stamina=0.3), Side.RED)
    assert fresh.output_multiplier == fresh.power_multiplier == 1
    assert 0 < tired.power_multiplier < tired.output_multiplier < 1
    assert resilient.output_multiplier > tired.output_multiplier
    assert resilient.power_multiplier > tired.power_multiplier


def test_action_uses_pre_action_modifiers_then_costs_only_actor():
    profiles = matchup()
    model = StaminaModel(profiles)
    candidate = DistanceCandidate(Side.RED, "strike", profiles, model, DynamicModifierProvider())
    state = FightState(red_stamina=0.5, blue_stamina=0.7)
    before = DynamicModifierProvider().modifiers(profiles.red, state, Side.RED)
    resolution = candidate.resolve(state, FightContext(FightConfig(), 0, 1), RNGManager(2).stream(20))
    assert resolution.payload.dynamic_modifiers == before
    SimulationEngine._apply_delta(state, resolution.delta)
    assert state.red_stamina < 0.5
    assert state.blue_stamina == 0.7
    assert state.fight_time_seconds == 0


def test_cost_resistance_capacity_and_clamping():
    low = StaminaModel(matchup(stamina_capacity=50, stamina_depletion_resistance=30))
    high = StaminaModel(matchup(stamina_capacity=150, stamina_depletion_resistance=70))
    assert low.action_delta(FightState(), Side.RED, "takedown").red_stamina < high.action_delta(FightState(), Side.RED, "takedown").red_stamina
    depleted = FightState(red_stamina=0.001)
    SimulationEngine._apply_delta(depleted, low.action_delta(depleted, Side.RED, "takedown"))
    assert depleted.red_stamina == 0


def test_round_recovery_once_and_never_above_cap():
    profiles = matchup()
    stamina = StaminaModel(profiles)
    result = SimulationEngine(FightConfig(2, 1), FightFlowRateProvider(profiles), StaminaTimeAdvanceModel(stamina), RNGManager(3), round_recovery_model=stamina).run(FightState(red_stamina=0.5, blue_stamina=1))
    assert result.state.red_stamina == pytest.approx(0.7)
    assert result.state.blue_stamina == 1


def test_positional_cost_uses_exact_dt_for_both_fighters():
    model = StaminaModel(matchup())
    state = FightState(phase=Phase.GROUND, ground_controller="red")
    delta = model.positional_delta(state, 12.5)
    assert delta.red_stamina == pytest.approx(1 - 0.025 * 12.5 / 100)
    assert delta.blue_stamina == pytest.approx(1 - 0.035 * 12.5 / 100)


def test_full_stamina_provider_exactly_recovers_phase3_rates_and_exits_stay_passive():
    profiles = matchup()
    context = FightContext(FightConfig(), 0, 1)
    legacy = FightFlowRateProvider(profiles)
    active = FightFlowRateProvider(profiles, StaminaModel(profiles), DynamicModifierProvider())
    for state in (FightState(), FightState(phase=Phase.CLINCH, clinch_controller="red"), FightState(phase=Phase.GROUND, ground_controller="red")):
        old = {x.candidate.candidate_id: x.rate_per_second for x in legacy.candidates(state, context)}
        new = {x.candidate.candidate_id: x.rate_per_second for x in active.candidates(state, context)}
        assert new == old
    tired = FightState(phase=Phase.GROUND, ground_controller="red", red_stamina=0.2, blue_stamina=0.2)
    old = {x.candidate.candidate_id: x.rate_per_second for x in legacy.candidates(tired, context)}
    new = {x.candidate.candidate_id: x.rate_per_second for x in active.candidates(tired, context)}
    assert new["blue_ground_escape"] == old["blue_ground_escape"]
    assert new["blue_ground_reversal"] == old["blue_ground_reversal"]
    assert new["red_ground_strike"] < old["red_ground_strike"]
