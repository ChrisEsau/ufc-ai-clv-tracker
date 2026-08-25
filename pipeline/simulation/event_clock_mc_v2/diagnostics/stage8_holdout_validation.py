"""Fresh Stage 8 holdout validation of intent priors and ground structure."""
from __future__ import annotations
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import BrainIntentPriors
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineConfig, EngineFunctions, EngineInputs, FighterEngineInputs, run_causal_path
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import historical_fighter_rows, load_latest_profiles, load_prefight_snapshots
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from .stage6_real_causal_path import _capabilities, _mechanics
from .stage8_intent_prior_shadow import IntentPriorChooser
from .stage8_structural_population import MASTER, ROUND_STATS, actual_side_totals, elapsed_seconds, pick_col, side_rows

STANDING={ActionFamily.STAND_ATTACK,ActionFamily.STAND_COUNTER}; GROUND={ActionFamily.GROUND_STRIKE,ActionFamily.BOTTOM_STRIKE}; TD={ActionFamily.TAKEDOWN_ENTRY,ActionFamily.CLINCH_TAKEDOWN}
def rate(n,seconds): return float(n*900/seconds/2) if seconds else 0.0
def quantiles(xs):
 a=np.asarray(xs,float); return {"mean":float(a.mean()) if a.size else 0.,"median":float(np.median(a)) if a.size else 0.,"p90":float(np.quantile(a,.9)) if a.size else 0.}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--fights',type=int,default=100);p.add_argument('--paths-per-fight',type=int,default=20);p.add_argument('--exclude-discovery-fights',type=int,default=25);p.add_argument('--clinch-ratio',type=float,default=.06);p.add_argument('--ground-strike-multiplier',type=float,default=3.0);p.add_argument('--submission-multiplier',type=float,default=.3);p.add_argument('--seed-base',type=int,default=20260826);p.add_argument('--output',type=Path,default=Path('data/diagnostics/event_clock_mc_v2/stage8_holdout_validation.json')); a=p.parse_args()
 master=pd.read_parquet(MASTER).drop_duplicates('fight_id').copy();master['fight_id']=master['fight_id'].astype(str);dc=pick_col(master,'date','event_date');master['_event_date']=pd.to_datetime(master[dc],errors='coerce').dt.normalize();master=master.dropna(subset=['_event_date']).sort_values(['_event_date','fight_id'],ascending=[False,False])
 rounds=pd.read_parquet(ROUND_STATS); fc=pick_col(rounds,'fight_id','bout_id'); available=set(rounds[fc].astype(str)); snapshots=load_prefight_snapshots();reference=CapabilityReference.from_latest(load_latest_profiles())
 complete=[]
 for _,fight in master.iterrows():
  fid=str(fight['fight_id'])
  if fid not in available: continue
  try:
   rid,bid=str(fight['r_id']),str(fight['b_id']); rf,bf=historical_fighter_rows(snapshots,event_date=fight['_event_date'],fight_id=fid,fighter_ids=(rid,bid));side_rows(rounds,fid,rid,'red');side_rows(rounds,fid,bid,'blue')
  except Exception: continue
  complete.append((fight,rf,bf))
  if len(complete)>=a.exclude_discovery_fights+a.fights: break
 if len(complete)<a.exclude_discovery_fights+a.fights: raise RuntimeError(f'insufficient complete fights: {len(complete)}')
 discovery=[str(x[0]['fight_id']) for x in complete[:a.exclude_discovery_fights]]; chosen=complete[a.exclude_discovery_fights:]
 counts=Counter(); actual=defaultdict(float); phases=defaultdict(float); durations=defaultdict(list); group=defaultdict(Counter); seconds=0.;actual_seconds=0.;illegal=mismatch=0
 for fi,(fight,rf,bf) in enumerate(chosen):
  horizon=elapsed_seconds(fight); rid,bid=str(fight['r_id']),str(fight['b_id']);rc,rr=_capabilities(rf,bf,reference);bc,br=_capabilities(bf,rf,reference)
  inp=EngineInputs(FighterEngineInputs(rc,BrainTimingContext(activity_rate_ratio=max(.2,rr.standing_rate_15m/120.0)),BrainDecisionContext(),_mechanics(rr)),FighterEngineInputs(bc,BrainTimingContext(activity_rate_ratio=max(.2,br.standing_rate_15m/120.0)),BrainDecisionContext(),_mechanics(br)))
  chooser=IntentPriorChooser({Side.RED:BrainIntentPriors(rr.standing_rate_15m,rr.takedown_rate_15m,a.clinch_ratio,a.ground_strike_multiplier,a.submission_multiplier),Side.BLUE:BrainIntentPriors(br.standing_rate_15m,br.takedown_rate_15m,a.clinch_ratio,a.ground_strike_multiplier,a.submission_multiplier)})
  funcs=EngineFunctions(action_chooser=chooser); cfg=EngineConfig(number_of_rounds=max(1,math.ceil(horizon/300)))
  for side,fid in [('red',rid),('blue',bid)]:
   vals=actual_side_totals(side_rows(rounds,str(fight['fight_id']),fid,side)); actual['standing']+=vals['distance_att'];actual['clinch']+=vals['clinch_att'];actual['ground']+=vals['ground_att'];actual['td']+=vals['td_att'];actual['td_landed']+=vals['td_land'];actual['sub']+=vals['sub_att'];actual_seconds+=horizon
  caps={Side.RED:(rc,rr),Side.BLUE:(bc,br)}
  for pi in range(a.paths_per_fight):
   out=run_causal_path(inp,seed=a.seed_base+fi*10000+pi,horizon_seconds=horizon,config=cfg,functions=funcs);seconds+=out.reported_through_seconds
   exposure=0.
   for s in out.timeline_segments: phases[s.phase.value]+=s.duration;durations[s.phase.value].append(s.duration);exposure+=s.duration
   mismatch+=not np.isclose(exposure,out.reported_through_seconds,atol=1e-9)
   for e in out.events:
    ac=e.selected_action
    if e.source_phase is Phase.GROUND and ac in STANDING: illegal+=1
    if ac in STANDING: counts['standing']+=1
    elif ac is ActionFamily.CLINCH_STRIKE: counts['clinch']+=1
    elif ac in GROUND: counts['ground']+=1; counts['bottom_strike' if ac is ActionFamily.BOTTOM_STRIKE else 'top_strike']+=1
    if ac in TD:
     counts['td']+=1;counts['direct_td' if ac is ActionFamily.TAKEDOWN_ENTRY else 'clinch_td']+=1
     if e.resulting_phase is Phase.GROUND: counts['td_landed']+=1;counts['ground_entries']+=1
    if ac is ActionFamily.SUBMISSION_ATTACK: counts['sub']+=1
    if ac is ActionFamily.CONTROL: counts['control']+=1
    if ac is ActionFamily.ESCAPE_STAND: counts['escape_attempt']+=1;counts['escape_success']+=e.resulting_phase is Phase.STANDING
    if ac is ActionFamily.REVERSAL: counts['reversal_attempt']+=1;counts['reversal_success']+=e.transition_kind is not None
    cap,rt=caps[e.actor]; bucket='low' if cap.ground_top<.33 else 'mid' if cap.ground_top<.67 else 'high';group[bucket][ac.value]+=1
 if illegal or mismatch: raise AssertionError({'illegal':illegal,'timeline_mismatch':mismatch})
 sim={k:rate(counts[k],seconds) for k in ('standing','clinch','ground','td','td_landed','sub','direct_td','clinch_td','ground_entries')}; act={k:float(actual[k]*900/actual_seconds) for k in ('standing','clinch','ground','td','td_landed','sub')}
 ground_minutes=phases['ground']/60; payload={'diagnostic':'Stage 8 fresh holdout','discovery_fight_ids':discovery,'holdout_fight_ids':[str(x[0]['fight_id']) for x in chosen],'fights':len(chosen),'paths_per_fight':a.paths_per_fight,'total_paths':len(chosen)*a.paths_per_fight,'clinch_ratio':a.clinch_ratio,'ground_strike_multiplier':a.ground_strike_multiplier,'submission_multiplier':a.submission_multiplier,'invariants':{'illegal_cross_phase_actions':illegal,'timeline_exposure_mismatches':mismatch},'actual_per15_per_fighter':act,'sim_per15_per_fighter':sim,'phase_share':{k:v/sum(phases.values()) for k,v in phases.items()},'segment_seconds':{k:quantiles(v) for k,v in durations.items()},'ground_per_minute':{'top_control':counts['control']/ground_minutes,'bottom_escape_attempts':counts['escape_attempt']/ground_minutes,'ground_strikes':counts['top_strike']/ground_minutes,'bottom_strikes':counts['bottom_strike']/ground_minutes,'submissions':counts['sub']/ground_minutes},'success_rates':{'escape':counts['escape_success']/max(1,counts['escape_attempt']),'reversal':counts['reversal_success']/max(1,counts['reversal_attempt'])},'ground_action_distribution_by_capability':{k:dict(v) for k,v in group.items()}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2));print('WROTE',a.output)
if __name__=='__main__':main()
