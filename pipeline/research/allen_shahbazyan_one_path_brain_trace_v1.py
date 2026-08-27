"""Research-only one-path Brain trace for Brendan Allen vs Edmen Shahbazyan.
RESET_RANGE is excluded from the standing Brain chooser and standing event clock.
IMPROVE_POSITION and ADVANCE_POSITION are excluded from non-standing Brain choices.
Production mechanics are unchanged.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import numpy as np
from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import action_probabilities_with_intent_priors
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics.leavitt_brito_intent_rate_shadow import IntentRateBrain, _standing_rates as _standing_rates_original
FIGHT_ID='419fff06f338f5c6'; PATH_ID=0
REMOVED_GROUND_ACTIONS={ActionFamily.IMPROVE_POSITION,ActionFamily.ADVANCE_POSITION}

def _enum(v): return None if v is None else getattr(v,'value',str(v))

def _standing_rates_no_reset(state,actor,capabilities,context,priors,config):
    rates,pres=_standing_rates_original(state,actor,capabilities,context,priors,config)
    rates=dict(rates)
    rates.pop(ActionFamily.RESET_RANGE,None)
    return rates,pres

class TraceBrain(IntentRateBrain):
    def __init__(self, inputs, priors, horizon):
        super().__init__(inputs, priors, horizon); self.decisions=[]
    def action_chooser(self,state,actor,capabilities,context,rng,config):
        if state.phase is Phase.STANDING:
            rates,pres=_standing_rates_no_reset(state,actor,capabilities,context,self.priors[actor],config)
            acts=tuple(rates); w=np.asarray([rates[a] for a in acts],dtype=float); probs=w/w.sum()
            rows=[{'action':a.value,'rate_15m':float(rates[a]),'probability':float(p)} for a,p in zip(acts,probs,strict=True)]
            i=int(rng.choice(len(acts),p=probs)); selected=acts[i]; extra={'dynamic_pressure':float(pres)}
        else:
            raw=action_probabilities_with_intent_priors(state,actor,capabilities,context,self.priors[actor],config)
            dist=[r for r in raw if r.action_family not in REMOVED_GROUND_ACTIONS]
            if not dist:
                raise RuntimeError('all non-standing Brain actions were filtered')
            weights=np.asarray([r.probability for r in dist],dtype=float); probs=weights/weights.sum()
            rows=[{'action':r.action_family.value,'utility':float(r.utility),'probability':float(p)} for r,p in zip(dist,probs,strict=True)]
            i=int(rng.choice(len(dist),p=probs)); selected=dist[i].action_family; extra={}
        self.decisions.append({'decision_index':len(self.decisions),'timestamp_before_action':float(state.fight_time_seconds),'round':int(state.round_number),'phase':state.phase.value,'ground_controller':_enum(state.ground_controller),'clinch_controller':_enum(state.clinch_controller),'actor':actor.value,'context':asdict(context),'brain_options':rows,'selected_action':selected.value,**extra})
        return selected

def mechanic_probability(event,inputs):
    m=inputs.fighter(event.actor).mechanics; a=event.selected_action
    if a in {ActionFamily.STAND_ATTACK,ActionFamily.STAND_COUNTER}: return {'landing_probability':m.standing_strike_landing_probability}
    if a in {ActionFamily.TAKEDOWN_ENTRY,ActionFamily.CLINCH_TAKEDOWN}: return {'completion_probability':m.takedown_completion_probability}
    if a in {ActionFamily.GROUND_STRIKE,ActionFamily.BOTTOM_STRIKE}: return {'landing_probability':m.ground_strike_landing_probability}
    if a is ActionFamily.ESCAPE_STAND: return {'escape_success_probability':m.ground_escape_probability}
    if a is ActionFamily.REVERSAL: return {'reversal_success_probability':m.ground_reversal_probability}
    if a is ActionFamily.DISENGAGE: return {'ground_exit_probability':1.0,'automatic':True}
    if a is ActionFamily.SUBMISSION_ATTACK: return {'submission_conversion_probability':event.submission_probability}
    if a is ActionFamily.CLINCH_ENTRY: return {'completion_probability':inputs.mechanics_placeholders.clinch_entry_success_probability}
    if a is ActionFamily.BREAK_CLINCH: return {'completion_probability':inputs.mechanics_placeholders.break_clinch_success_probability}
    if a is ActionFamily.CLINCH_STRIKE: return {'landing_probability':inputs.mechanics_placeholders.clinch_strike_landing_probability}
    return {}

def main():
    pressure_mod.FIGHT_ID=FIGHT_ID; pressure_mod.PATHS=1
    intent_mod._standing_rates=_standing_rates_no_reset
    fight,inputs,priors,horizon,cfg=pressure_mod.build_setup(); brain=TraceBrain(inputs,priors,horizon)
    funcs=EngineFunctions(timing_sampler=brain.timing_sampler,action_chooser=brain.action_chooser)
    seed=derive_path_seed(SEED_SET_VERSION,FIGHT_ID,PATH_ID)
    out=run_causal_path(inputs,seed=seed,horizon_seconds=horizon,config=cfg,functions=funcs)
    if len(brain.decisions)!=len(out.events): raise RuntimeError(f'decision/event mismatch: {len(brain.decisions)} != {len(out.events)}')
    names={Side.RED:str(fight.r_name),Side.BLUE:str(fight.b_name)}; trace=[]
    for d,e in zip(brain.decisions,out.events,strict=True):
        trace.append({**d,'actor_name':names[e.actor],'event_timestamp':float(e.timestamp_seconds),'source_phase':e.source_phase.value,'outcome':e.outcome.value,'transition_kind':_enum(e.transition_kind),'resulting_phase':e.resulting_phase.value,'resulting_controller':_enum(e.resulting_controller),'mechanics':mechanic_probability(e,inputs),'impact':float(e.impact),'ko_probability':float(e.ko_probability),'kd_probability':float(e.kd_probability),'knockdown':bool(e.knockdown),'ko_tko':bool(e.ko_tko),'submission_attempt':bool(e.submission_attempt),'submission_probability':float(e.submission_probability),'submission_success':bool(e.submission_success)})
    payload={'study':'Allen-Shahbazyan one-path Brain trace without reset/improve/advance','production_changed':False,'reset_range_removed':True,'improve_position_removed':True,'advance_position_removed':True,'fight_id':FIGHT_ID,'path_id':PATH_ID,'seed_set':SEED_SET_VERSION,'seed':seed,'red':str(fight.r_name),'blue':str(fight.b_name),'brain_priors':{s.value:{'fighter':names[s],'standing_attempt_rate_15m':priors[s].standing_attempt_rate_15m,'takedown_attempt_rate_15m':priors[s].takedown_attempt_rate_15m,'clinch_entry_to_standing_ratio':priors[s].clinch_entry_to_standing_ratio,'ground_strike_odds_multiplier':priors[s].ground_strike_odds_multiplier,'submission_odds_multiplier':priors[s].submission_odds_multiplier,'ground_escape_probability':inputs.fighter(s).mechanics.ground_escape_probability,'ground_reversal_probability':inputs.fighter(s).mechanics.ground_reversal_probability} for s in Side},'termination':None if out.termination is None else {'winner':names[out.termination.winner],'winner_side':out.termination.winner.value,'method':out.termination.finish_method.value,'reported_through_seconds':out.reported_through_seconds},'timeline_segments':[{'start':s.start_time,'end':s.end_time,'duration':s.duration,'phase':s.phase.value,'controller':_enum(s.controller),'controller_name':None if s.controller is None else names[s.controller],'entry_reason':s.entry_reason,'exit_reason':s.exit_reason} for s in out.timeline_segments],'events':trace}
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
