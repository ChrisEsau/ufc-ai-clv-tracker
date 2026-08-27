"""Simulation-based OOS calibration of research takedown-attempt timing.

Research only; production is untouched.

Frozen architecture:
- standing strike clock = 1.0x raw matchup FSR
- RESET_RANGE removed
- clinch-entry rate = validated PIT clean-round proxy with frozen fitted scale 2.349514563106796
- inside-clinch timing unchanged
- takedown completion mechanics unchanged

Study questions:
1) Does matchup-effective FSR TD rate work as the absolute standing TD attempt clock?
2) Does the existing live td_factor improve or distort realized TD-attempt volume?

We therefore calibrate a global multiplier on raw FSR TD rate with live TD context OFF,
and separately report the current live-context condition at 1.0x. Target is UFCStats
fight-level TD attempts (both fighters). Pre-2025 train; 2025-26 untouched holdout.
"""
from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.research.clinch_entry_rate_simulation_oos_calibration import (
    build_proxy_table, pit_matchup_equiv, choose_fights, elapsed,
)
from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import BrainIntentPriors
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext, action_utilities
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineConfig, EngineFunctions, EngineInputs, FighterEngineInputs, run_causal_path
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import historical_fighter_rows, load_latest_profiles, load_prefight_snapshots
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage6_real_causal_path import _capabilities, _mechanics
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_structural_population import MASTER, ROUND_STATS, actual_side_totals, pick_col, side_rows
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

OUTDIR = Path('data/research/td_timing_simulation_oos')
CUTOFF = pd.Timestamp('2025-01-01')
TRAIN_FIGHTS = 40
HOLDOUT_FIGHTS = 40
PATHS = 8
SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)
CLINCH_SCALE = 2.349514563106796
SEED_BASE = 2026082704
EPS = 1e-12
ORIGINAL_STANDING_RATES = intent_mod._standing_rates


def rate_function(td_scale, clinch_rates, use_td_context):
    def _rates(state, actor, capabilities, context, priors, config):
        # Call original only to recover the current live TD-context factor when requested.
        original, _ = ORIGINAL_STANDING_RATES(state, actor, capabilities, context, priors, config)
        rates = dict(original)
        rates[ActionFamily.STAND_ATTACK] = max(float(priors.standing_attempt_rate_15m), EPS)
        if use_td_context:
            base = max(float(priors.takedown_attempt_rate_15m), EPS)
            context_factor = float(original[ActionFamily.TAKEDOWN_ENTRY]) / base
        else:
            context_factor = 1.0
        rates[ActionFamily.TAKEDOWN_ENTRY] = max(float(priors.takedown_attempt_rate_15m) * float(td_scale) * context_factor, EPS)
        rates[ActionFamily.CLINCH_ENTRY] = max(float(clinch_rates[actor]), EPS)
        rates.pop(ActionFamily.RESET_RANGE, None)
        return rates, 0.0
    return _rates


def simulate_set(fights, td_scale, use_td_context, rounds_df, reference, clean, global_equiv, split):
    rows=[]; actual_total=sim_total=success_total=0.0
    for fi,(mr,red_fsr,blue_fsr) in enumerate(fights):
        fid=str(mr.fight_id); h=elapsed(mr)
        red_cap,red_runtime=_capabilities(red_fsr,blue_fsr,reference)
        blue_cap,blue_runtime=_capabilities(blue_fsr,red_fsr,reference)
        rn=str(mr.get('r_name',mr.r_id)); bn=str(mr.get('b_name',mr.b_id))
        red_eq,_,_=pit_matchup_equiv(clean,global_equiv,mr._event_date,rn,bn)
        blue_eq,_,_=pit_matchup_equiv(clean,global_equiv,mr._event_date,bn,rn)
        clinch_rates={Side.RED:3.0*red_eq*CLINCH_SCALE,Side.BLUE:3.0*blue_eq*CLINCH_SCALE}
        intent_mod._standing_rates=rate_function(td_scale,clinch_rates,use_td_context)
        priors={
            Side.RED:BrainIntentPriors(red_runtime.standing_rate_15m,red_runtime.takedown_rate_15m,0.06,3.0,0.3),
            Side.BLUE:BrainIntentPriors(blue_runtime.standing_rate_15m,blue_runtime.takedown_rate_15m,0.06,3.0,0.3),
        }
        inputs=EngineInputs(
            red=FighterEngineInputs(red_cap,BrainTimingContext(),BrainDecisionContext(),_mechanics(red_runtime)),
            blue=FighterEngineInputs(blue_cap,BrainTimingContext(),BrainDecisionContext(),_mechanics(blue_runtime)),
        )
        cfg=EngineConfig(number_of_rounds=max(1,int(math.ceil(h/300.0))))
        ar=actual_side_totals(side_rows(rounds_df,fid,str(mr.r_id),'red'))['td_att']
        ab=actual_side_totals(side_rows(rounds_df,fid,str(mr.b_id),'blue'))['td_att']
        actual=float(ar+ab)
        pa=[]; ps=[]
        for pi in range(PATHS):
            brain=intent_mod.IntentRateBrain(inputs,priors,h)
            funcs=EngineFunctions(timing_sampler=brain.timing_sampler,action_chooser=brain.action_chooser)
            result=run_causal_path(inputs,seed=SEED_BASE+fi*1000+pi,horizon_seconds=h,config=cfg,functions=funcs)
            attempts=0; successes=0
            for ev in result.events:
                if ev.selected_action in (ActionFamily.TAKEDOWN_ENTRY,ActionFamily.CLINCH_TAKEDOWN):
                    attempts += 1
                    if ev.resulting_phase.value == 'ground' and ev.resulting_controller is ev.actor:
                        successes += 1
            pa.append(float(attempts)); ps.append(float(successes))
        sim=float(np.mean(pa)); succ=float(np.mean(ps))
        actual_total+=actual; sim_total+=sim; success_total+=succ
        rows.append({'split':split,'td_scale':float(td_scale),'use_live_td_context':bool(use_td_context),'fight_id':fid,
                     'event_date':str(pd.Timestamp(mr._event_date).date()),'red_name':rn,'blue_name':bn,
                     'actual_td_attempts_both':actual,'sim_td_attempts_both_mean':sim,'sim_td_success_both_mean':succ,
                     'red_fsr_td_rate_15m':float(red_runtime.takedown_rate_15m),'blue_fsr_td_rate_15m':float(blue_runtime.takedown_rate_15m),'paths':PATHS})
    return rows,{'actual_td_attempts':actual_total,'sim_td_attempts':sim_total,
                 'E_over_O':sim_total/actual_total if actual_total>0 else None,
                 'sim_td_successes':success_total,'sim_completion_rate':success_total/sim_total if sim_total>0 else None}


def main():
    master=pd.read_parquet(MASTER).drop_duplicates('fight_id').copy(); master['fight_id']=master.fight_id.astype(str)
    dc=pick_col(master,'date','event_date'); master['_event_date']=pd.to_datetime(master[dc],errors='coerce').dt.normalize(); master=master.dropna(subset=['_event_date'])
    rounds=pd.read_parquet(ROUND_STATS).copy(); clean,global_equiv,_,_=build_proxy_table(rounds)
    snaps=load_prefight_snapshots(); reference=CapabilityReference.from_latest(load_latest_profiles())
    train=choose_fights(master,rounds,snaps,before_cutoff=True,n=TRAIN_FIGHTS); hold=choose_fights(master,rounds,snaps,before_cutoff=False,n=HOLDOUT_FIGHTS)
    all_rows=[]; grid=[]
    for s in SCALES:
        rs,sm=simulate_set(train,s,False,rounds,reference,clean,global_equiv,'train_raw_grid'); all_rows+=rs; grid.append({'scale':s,**sm}); print('TRAIN_RAW',s,sm,flush=True)
    g=sorted(grid,key=lambda x:x['scale']); target=g[0]['actual_td_attempts']; fitted=min(g,key=lambda x:abs(x['sim_td_attempts']-target))['scale']
    for a,b in zip(g[:-1],g[1:]):
        if (a['sim_td_attempts']-target)*(b['sim_td_attempts']-target)<=0 and b['sim_td_attempts']!=a['sim_td_attempts']:
            fitted=a['scale']+(target-a['sim_td_attempts'])*(b['scale']-a['scale'])/(b['sim_td_attempts']-a['sim_td_attempts']); break
    fitted=float(fitted)
    tr,trsm=simulate_set(train,fitted,False,rounds,reference,clean,global_equiv,'train_raw_fitted'); ho,hosm=simulate_set(hold,fitted,False,rounds,reference,clean,global_equiv,'holdout_raw_fitted')
    ctxtr,ctxtrsm=simulate_set(train,1.0,True,rounds,reference,clean,global_equiv,'train_context_1x'); ctxho,ctxhosm=simulate_set(hold,1.0,True,rounds,reference,clean,global_equiv,'holdout_context_1x')
    all_rows += tr+ho+ctxtr+ctxho; intent_mod._standing_rates=ORIGINAL_STANDING_RATES
    result={'study':'simulation OOS calibration of takedown attempt timing','production_changed':False,'standing_clock_scale':1.0,
            'clinch_entry_scale':CLINCH_SCALE,'inside_clinch_timing_scale':1.0,'td_completion_mechanics_changed':False,
            'target':'UFCStats fight-level takedown attempts, both fighters','candidate_raw_fsr_grid':grid,'fitted_raw_fsr_td_scale':fitted,
            'train_raw_fitted':trsm,'holdout_raw_fitted':hosm,'train_current_live_td_context_1x':ctxtrsm,'holdout_current_live_td_context_1x':ctxhosm,
            'cutoff':str(CUTOFF.date()),'train_fights':TRAIN_FIGHTS,'holdout_fights':HOLDOUT_FIGHTS,'paths_per_fight':PATHS}
    OUTDIR.mkdir(parents=True,exist_ok=True); (OUTDIR/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); pd.DataFrame(all_rows).to_csv(OUTDIR/'fight_level_results.csv',index=False); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
