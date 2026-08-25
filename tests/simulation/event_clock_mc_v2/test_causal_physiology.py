from dataclasses import replace
import numpy as np
from pipeline.simulation.event_clock_mc_v2.brain.capabilities import BrainCapabilities
from pipeline.simulation.event_clock_mc_v2.brain.memory import decision_context
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.causal.events import (
    ActionEvent,
    ActionFamily,
)
from pipeline.simulation.event_clock_mc_v2.causal.state import (
    FightPhysiology,
    FightState,
    FighterPhysiology,
    Phase,
    Side,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.config import (
    FighterMechanics,
    MechanicsInputs,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.physiology import (
    advance_physiology,
    apply_action_consequence,
    recover_round,
    resolve_landed_strike,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import (
    FinishMethod,
    StrikeConsequence,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action

BASE = FighterMechanics(1, 0.5, 1, 0, 0.4, 0.3)


def inputs(red=BASE, blue=BASE):
    return MechanicsInputs(red, blue)


def event():
    return ActionEvent(1, Side.RED, ActionFamily.STAND_ATTACK, Phase.STANDING)


def test_landed_strike_updates_trauma_but_miss_does_not():
    state = FightState(fight_time_seconds=1)
    landed = resolve_action(
        event(), state, inputs(), np.random.default_rng(3)
    ).consequence
    updated = apply_action_consequence(
        state, Side.RED, ActionFamily.STAND_ATTACK, landed, BASE
    )
    assert updated.physiology.blue.cumulative_trauma == landed.trauma_increment > 0
    miss = StrikeConsequence(False)
    unchanged = apply_action_consequence(
        state, Side.RED, ActionFamily.STAND_ATTACK, miss, BASE
    )
    assert unchanged.physiology.blue.cumulative_trauma == 0


def test_power_and_durability_move_trauma_in_expected_direction():
    low = FighterMechanics(1, 0.5, 1, 0, 0.4, 0.3, striking_power=30)
    high = replace(low, striking_power=80)
    durable = replace(BASE, damage_durability=80)
    fragile = replace(BASE, damage_durability=30)
    a = resolve_landed_strike(
        event(),
        FightState(fight_time_seconds=1),
        inputs(low, BASE),
        True,
        np.random.default_rng(7),
    )
    b = resolve_landed_strike(
        event(),
        FightState(fight_time_seconds=1),
        inputs(high, BASE),
        True,
        np.random.default_rng(7),
    )
    assert b.impact > a.impact
    c = resolve_landed_strike(
        event(),
        FightState(fight_time_seconds=1),
        inputs(BASE, durable),
        True,
        np.random.default_rng(8),
    )
    d = resolve_landed_strike(
        event(),
        FightState(fight_time_seconds=1),
        inputs(BASE, fragile),
        True,
        np.random.default_rng(8),
    )
    assert d.trauma_increment > c.trauma_increment


def test_sufficient_impact_can_knock_down_and_finish():
    power = replace(BASE, striking_power=1000)
    weak = replace(BASE, damage_durability=1, knockdown_resistance=1)
    consequence = resolve_landed_strike(
        event(),
        FightState(fight_time_seconds=1),
        inputs(power, weak),
        True,
        np.random.default_rng(2),
    )
    assert consequence.knockdown and consequence.acute_increment > 0
    assert (
        consequence.termination is not None
        and consequence.termination.finish_method is FinishMethod.KO_TKO
    )
    updated = apply_action_consequence(
        FightState(fight_time_seconds=1),
        Side.RED,
        ActionFamily.STAND_ATTACK,
        consequence,
        power,
    )
    assert updated.physiology.blue.knockdowns_suffered == 1


def test_stamina_cost_recovery_positional_cost_and_brain_visibility():
    state = FightState(
        fight_time_seconds=1, phase=Phase.GROUND, ground_controller=Side.RED
    )
    after = apply_action_consequence(
        state, Side.RED, ActionFamily.TAKEDOWN_ENTRY, None, BASE
    )
    assert after.physiology.red.stamina < 1
    advanced = advance_physiology(after, 11)
    assert advanced.physiology.blue.stamina < after.physiology.blue.stamina
    recovered = recover_round(advanced)
    assert recovered.physiology.red.stamina > advanced.physiology.red.stamina
    context = decision_context(recovered, Side.RED, BrainDecisionContext(), 900)
    assert context.fatigue == 1 - recovered.physiology.red.stamina


def test_acute_vulnerability_decays_on_authoritative_clock():
    physiology = FightPhysiology(
        FighterPhysiology(acute_vulnerability=1), FighterPhysiology()
    )
    state = FightState(physiology=physiology)
    advanced = advance_physiology(state, 30)
    assert advanced.physiology.red.acute_vulnerability == 0.5


def test_ko_termination_stops_causal_engine_immediately():
    from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
    from pipeline.simulation.event_clock_mc_v2.engine import (
        EngineFunctions,
        EngineInputs,
        FighterEngineInputs,
        run_causal_path,
    )

    power = replace(BASE, striking_power=1000)
    weak = replace(BASE, damage_durability=1, knockdown_resistance=1)
    cap = BrainCapabilities(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.4, 0.3)
    red = FighterEngineInputs(cap, BrainTimingContext(), BrainDecisionContext(), power)
    blue = FighterEngineInputs(cap, BrainTimingContext(), BrainDecisionContext(), weak)

    class Timing:
        def __call__(self, state, context, rng, config):
            return 1 if context is red.timing_context else 10

    def choose(state, actor, capabilities, context, rng, config):
        return ActionFamily.STAND_ATTACK

    result = run_causal_path(
        EngineInputs(red, blue),
        seed=2,
        horizon_seconds=20,
        functions=EngineFunctions(Timing(), choose, resolve_action),
    )
    assert len(result.events) == 1 and result.final_state.finished
    assert result.termination.finish_method is FinishMethod.KO_TKO
    assert (
        result.final_state.winner is Side.RED and result.reported_through_seconds == 1
    )
    assert result.final_pending_actions == ()
