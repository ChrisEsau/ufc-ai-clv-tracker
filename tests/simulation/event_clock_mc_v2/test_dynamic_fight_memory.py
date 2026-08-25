from dataclasses import FrozenInstanceError, replace
import numpy as np
import pytest

from pipeline.simulation.event_clock_mc_v2.brain.capabilities import BrainCapabilities
from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import BrainIntentPriors, action_probabilities_with_intent_priors
from pipeline.simulation.event_clock_mc_v2.brain.memory import FightMemoryConfig, decision_context, decay_memory, update_memory
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext, action_probabilities
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionEvent, ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import FightMemory, FighterMemory, FightState, Phase, Side
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import ActionOutcome, ActionResolution, TransitionKind, TransitionRequest

CAP=BrainCapabilities(.5,.5,.5,.5,.5,.5,.3,.4,.3)
PRIORS=BrainIntentPriors(80,4,.06,3,.3)

def event(action, outcome, *, time=1, actor=Side.RED, transition=None):
    source=Phase.STANDING if action in {ActionFamily.TAKEDOWN_ENTRY,ActionFamily.STAND_ATTACK} else Phase.GROUND
    return ActionResolution(ActionEvent(time,actor,action,source),outcome,transition)

def td_probability(state):
    rows=action_probabilities_with_intent_priors(state,Side.RED,CAP,decision_context(state,Side.RED,BrainDecisionContext(),900),PRIORS)
    return next(x.probability for x in rows if x.action_family is ActionFamily.TAKEDOWN_ENTRY)

def test_memory_is_frozen_bounded_and_owned_by_fight_state():
    state=FightState()
    assert state.memory == FightMemory()
    with pytest.raises(FrozenInstanceError): state.memory.red.td_success_recent=.5
    with pytest.raises(ValueError): FighterMemory(td_failure_recent=1.1)

def test_td_success_raises_and_repeated_failures_reduce_intent():
    transition=TransitionRequest(TransitionKind.DIRECT_TAKEDOWN,Phase.STANDING,Phase.GROUND,Side.RED)
    success=update_memory(FightMemory(),event(ActionFamily.TAKEDOWN_ENTRY,ActionOutcome.SUCCESS,transition=transition))
    success_state=FightState(fight_time_seconds=1,memory=success)
    assert td_probability(success_state)>td_probability(FightState(fight_time_seconds=1,memory=decay_memory(FightMemory(),1)))
    memory=FightMemory()
    for time in (1,2,3): memory=update_memory(memory,event(ActionFamily.TAKEDOWN_ENTRY,ActionOutcome.STUFFED,time=time))
    assert td_probability(FightState(fight_time_seconds=3,memory=memory))<td_probability(FightState(fight_time_seconds=3,memory=decay_memory(FightMemory(),3)))

def test_td_defender_memory_and_decay_return_toward_baseline():
    memory=update_memory(FightMemory(),event(ActionFamily.TAKEDOWN_ENTRY,ActionOutcome.STUFFED))
    assert memory.blue.td_defense_success_recent>0
    later=decay_memory(memory,121,FightMemoryConfig(half_life_seconds=60))
    assert later.red.td_failure_recent == pytest.approx(memory.red.td_failure_recent*.25)
    assert later.blue.td_defense_success_recent == pytest.approx(memory.blue.td_defense_success_recent*.25)

def test_striking_results_create_symmetric_recent_edge_and_decay():
    landed=update_memory(FightMemory(),event(ActionFamily.STAND_ATTACK,ActionOutcome.LANDED))
    assert landed.red.striking_edge>0>landed.blue.striking_edge
    assert abs(decay_memory(landed,61).red.striking_edge)<abs(landed.red.striking_edge)
    positive=action_probabilities(FightState(),Side.RED,CAP,BrainDecisionContext(striking_edge=.7))
    negative=action_probabilities(FightState(),Side.RED,CAP,BrainDecisionContext(striking_edge=-.7))
    p=lambda rows,a:next(x.probability for x in rows if x.action_family is a)
    assert p(positive,ActionFamily.STAND_ATTACK)>p(negative,ActionFamily.STAND_ATTACK)

def test_position_context_is_derived_from_authoritative_controller():
    top=FightState(phase=Phase.GROUND,ground_controller=Side.RED)
    assert decision_context(top,Side.RED,BrainDecisionContext(),900).dominant_top_position==1
    assert decision_context(top,Side.BLUE,BrainDecisionContext(),900).bad_bottom_position==1
    bottom_neutral=action_probabilities(top,Side.BLUE,CAP,BrainDecisionContext())
    bottom_bad=action_probabilities(top,Side.BLUE,CAP,decision_context(top,Side.BLUE,BrainDecisionContext(),900))
    p=lambda rows,a:next(x.probability for x in rows if x.action_family is a)
    assert p(bottom_bad,ActionFamily.ESCAPE_STAND)>p(bottom_neutral,ActionFamily.ESCAPE_STAND)

def test_dominant_top_and_late_score_directional_behavior():
    top=FightState(phase=Phase.GROUND,ground_controller=Side.RED)
    neutral=action_probabilities(top,Side.RED,CAP,BrainDecisionContext())
    dominant=action_probabilities(top,Side.RED,CAP,decision_context(top,Side.RED,BrainDecisionContext(),900))
    p=lambda rows,a:next(x.probability for x in rows if x.action_family is a)
    assert p(dominant,ActionFamily.DISENGAGE)<p(neutral,ActionFamily.DISENGAGE)
    behind=action_probabilities(FightState(),Side.RED,CAP,BrainDecisionContext(score_state=-1,late_urgency=1))
    ahead=action_probabilities(FightState(),Side.RED,CAP,BrainDecisionContext(score_state=1,late_urgency=1))
    assert p(behind,ActionFamily.STAND_ATTACK)>p(ahead,ActionFamily.STAND_ATTACK)

def test_memory_updates_are_deterministic_and_do_not_use_rng():
    resolution=event(ActionFamily.STAND_ATTACK,ActionOutcome.LANDED)
    assert update_memory(FightMemory(),resolution)==update_memory(FightMemory(),resolution)

def test_engine_path_memory_trace_is_causal_and_seed_deterministic():
    from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
    from pipeline.simulation.event_clock_mc_v2.engine import EngineInputs,FighterEngineInputs,run_causal_path
    from pipeline.simulation.event_clock_mc_v2.mechanics.config import FighterMechanics
    mechanics=FighterMechanics(.55,.35,.5,0,.4,.3)
    fighter=FighterEngineInputs(CAP,BrainTimingContext(),BrainDecisionContext(),mechanics)
    first=run_causal_path(EngineInputs(fighter,fighter),seed=991,horizon_seconds=30)
    repeat=run_causal_path(EngineInputs(fighter,fighter),seed=991,horizon_seconds=30)
    assert first==repeat
    assert all(record.pre_decision_context.td_success_recent <= record.resulting_actor_memory.td_success_recent or record.selected_action is not ActionFamily.TAKEDOWN_ENTRY for record in first.events)
    assert first.final_state.memory.updated_at_seconds <= first.reported_through_seconds
