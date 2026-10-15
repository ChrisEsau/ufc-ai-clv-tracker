"""Leakage-safe Stage-1 FSR prior-strength attribution for bantamweight.

Rebuilds the validated standing/takedown rate and paired-effectiveness families
chronologically under three prior-strength settings: current (1.0), half (0.5),
and quarter (0.25). No production publication is changed.

Each variant is evaluated against later realized fight outputs at the runtime
input boundary: standing attempt rate, standing accuracy, takedown attempt rate,
and takedown completion. Market favorite probability is used only as an external
separation diagnostic, never as a fitting target.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.replay.rate_families import (
    standing_spec, takedown_spec, build_rate_fighter_fights, replay_tendency, replay_suppression,
)
from pipeline.fsr_v3.replay.paired_effectiveness import (
    standing_effectiveness_spec, takedown_effectiveness_spec, replay_effectiveness_family,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import build_two_way_market, MARKET_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.bantamweight_fsr_population_inverse_market_audit import (
    ROUND_STATS, pick_col, fighter_rows, round_totals,
)

OUT=Path('data/diagnostics/event_clock_mc_v2/bantamweight_fsr_shrinkage_attribution')
DIVISION='bantamweight'; EPS=1e-9
KEYS=['event_date','fight_id','fighter_id']

def logit(p):
    p=float(np.clip(p,EPS,1-EPS)); return math.log(p/(1-p))
def sigmoid(x): return 1.0/(1.0+math.exp(-float(np.clip(x,-40,40))))

def scaled_config(strength: float) -> FSRV3Config:
    c=FSRV3Config(); s=float(strength)
    return replace(
        c,
        standing_tendency_prior_seconds=c.standing_tendency_prior_seconds*s,
        takedown_tendency_prior_seconds=c.takedown_tendency_prior_seconds*s,
        standing_suppression_prior_shape=c.standing_suppression_prior_shape*s,
        takedown_suppression_prior_shape=c.takedown_suppression_prior_shape*s,
        # Normal prior precision is proportional to 1/sigma^2.
        standing_effectiveness_sigma_offense=c.standing_effectiveness_sigma_offense/math.sqrt(s),
        standing_effectiveness_sigma_defense=c.standing_effectiveness_sigma_defense/math.sqrt(s),
        takedown_effectiveness_sigma_offense=c.takedown_effectiveness_sigma_offense/math.sqrt(s),
        takedown_effectiveness_sigma_defense=c.takedown_effectiveness_sigma_defense/math.sqrt(s),
    )

def hist_trait(frame, trait):
    x=frame[frame['trait'].eq(trait)][KEYS+['pre_rating']].copy()
    return x.rename(columns={'pre_rating':trait})

def build_variant(paired: pd.DataFrame, strength: float) -> pd.DataFrame:
    cfg=scaled_config(strength)
    ss=standing_spec(cfg); ts=takedown_spec(cfg)
    sff=build_rate_fighter_fights(ss,paired_rounds=paired)
    tff=build_rate_fighter_fights(ts,paired_rounds=paired)
    sth=replay_tendency(sff,ss); ssh=replay_suppression(sth,ss)
    tth=replay_tendency(tff,ts); tsh=replay_suppression(tth,ts)
    seh=replay_effectiveness_family(standing_effectiveness_spec(cfg),paired_rounds=paired)
    teh=replay_effectiveness_family(takedown_effectiveness_spec(cfg),paired_rounds=paired)
    pieces=[
        sth[KEYS+['pre_rating']].rename(columns={'pre_rating':'standing_striking_tendency'}),
        ssh[KEYS+['pre_rating']].rename(columns={'pre_rating':'standing_striking_suppression'}),
        hist_trait(seh,'standing_striking_offense'), hist_trait(seh,'standing_striking_defense'),
        tth[KEYS+['pre_rating']].rename(columns={'pre_rating':'takedown_tendency'}),
        tsh[KEYS+['pre_rating']].rename(columns={'pre_rating':'takedown_suppression'}),
        hist_trait(teh,'takedown_offense'), hist_trait(teh,'takedown_defense'),
    ]
    out=pieces[0]
    for p in pieces[1:]: out=out.merge(p,on=KEYS,how='outer',validate='one_to_one')
    return out

def actual_by_fight(cohort):
    rs=pd.read_parquet(ROUND_STATS).copy(); fcol=pick_col(rs,'fight_id','bout_id'); rs[fcol]=rs[fcol].astype(str)
    rows=[]
    for _,mr in cohort.iterrows():
        fr=rs[rs[fcol].eq(str(mr['fight_id']))]; rr,br=fighter_rows(fr,mr)
        if rr.empty or br.empty: continue
        exposure=max(float(mr['match_time_sec']),1.0)
        for side,g,fid in [('red',rr,str(mr['r_id'])),('blue',br,str(mr['b_id']))]:
            a=round_totals(g)
            rows.append({'fight_id':str(mr['fight_id']),'event_date':mr['event_date'],'fighter_id':fid,'side':side,
                'actual_standing_rate_15m':a['standing_att']*900.0/exposure,
                'actual_standing_accuracy':a['standing_land']/a['standing_att'] if a['standing_att']>0 else np.nan,
                'actual_td_rate_15m':a['td_att']*900.0/exposure,
                'actual_td_completion':a['td_land']/a['td_att'] if a['td_att']>0 else np.nan})
    return pd.DataFrame(rows)

def evaluate_variant(name,variant,cohort,actual,base,market):
    keep=set(cohort['fight_id'].astype(str)); v=variant[variant['fight_id'].isin(keep)].copy()
    b=base[base['fight_id'].isin(keep)][KEYS+['standing_accuracy_baseline','takedown_completion_baseline']].copy()
    v=v.merge(b,on=KEYS,how='inner',validate='one_to_one').merge(actual,on=KEYS,how='inner',validate='one_to_one')
    rows=[]; fights=[]
    for fid,g in v.groupby('fight_id'):
        if len(g)!=2: continue
        a,brow=g.iloc[0],g.iloc[1]; directional=[]
        for me,op in [(a,brow),(brow,a)]:
            pr_st=float(me['standing_striking_tendency'])*float(op['standing_striking_suppression'])
            pr_sa=sigmoid(logit(float(me['standing_accuracy_baseline']))+float(me['standing_striking_offense'])-float(op['standing_striking_defense']))
            pr_td=float(me['takedown_tendency'])*float(op['takedown_suppression'])
            pr_tc=sigmoid(logit(float(me['takedown_completion_baseline']))+float(me['takedown_offense'])-float(op['takedown_defense']))
            vals={'standing_rate_15m':pr_st,'standing_accuracy':pr_sa,'td_rate_15m':pr_td,'td_completion':pr_tc}
            directional.append(vals)
            for metric,pred in vals.items():
                actual_col='actual_'+metric
                av=float(me[actual_col]) if pd.notna(me[actual_col]) else np.nan
                rows.append({'variant':name,'fight_id':fid,'fighter_id':me['fighter_id'],'metric':metric,'predicted':pred,'actual':av,
                             'abs_error':abs(pred-av) if np.isfinite(av) else np.nan})
        # Scale-free magnitude of matchup separation at runtime boundary.
        sep=(abs(math.log(max(directional[0]['standing_rate_15m'],EPS)/max(directional[1]['standing_rate_15m'],EPS)))
             +abs(logit(directional[0]['standing_accuracy'])-logit(directional[1]['standing_accuracy']))
             +abs(math.log(max(directional[0]['td_rate_15m'],EPS)/max(directional[1]['td_rate_15m'],EPS)))
             +abs(logit(directional[0]['td_completion'])-logit(directional[1]['td_completion'])))
        fights.append({'variant':name,'fight_id':fid,'runtime_separation':sep})
    detail=pd.DataFrame(rows); fd=pd.DataFrame(fights)
    summary=[]
    for metric,g in detail.groupby('metric'):
        z=g.dropna(subset=['actual'])
        summary.append({'variant':name,'metric':metric,'n':len(z),'mae':float(z['abs_error'].mean()),
                        'corr_pred_actual':float(z['predicted'].corr(z['actual'])) if len(z)>2 else np.nan,
                        'pred_sd':float(z['predicted'].std()),'actual_sd':float(z['actual'].std())})
    priced=fd.merge(market[['fight_id','market_favorite_fair_p']],on='fight_id',how='inner')
    msum={'variant':name,'n_priced':len(priced),'mean_runtime_separation':float(priced['runtime_separation'].mean()) if len(priced) else np.nan,
          'corr_separation_market':float(priced['runtime_separation'].corr(priced['market_favorite_fair_p'])) if len(priced)>2 else np.nan}
    return detail,pd.DataFrame(summary),fd,msum

def main():
    master=pd.read_parquet(MASTER_PATH).drop_duplicates('fight_id').copy(); master['fight_id']=master['fight_id'].astype(str)
    master['event_date']=pd.to_datetime(master['date'],errors='coerce').dt.normalize()
    cohort=master[master['division'].astype(str).str.strip().str.lower().eq(DIVISION)].copy()
    base=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy(); base['fight_id']=base['fight_id'].astype(str); base['fighter_id']=base['fighter_id'].astype(str); base['event_date']=pd.to_datetime(base['event_date']).dt.normalize()
    valid=set(base.groupby('fight_id').size().loc[lambda s:s==2].index.astype(str)); cohort=cohort[cohort['fight_id'].isin(valid)&cohort['match_time_sec'].notna()].copy()
    actual=actual_by_fight(cohort); paired=build_paired_rounds()
    market=build_two_way_market(MARKET_PATH).copy(); market['fight_id']=market['fight_id'].astype(str); market=market[market['fight_id'].isin(set(cohort['fight_id']))]
    all_detail=[]; all_summary=[]; all_fights=[]; market_rows=[]
    for name,strength in [('current_1.00',1.0),('prior_0.50',0.5),('prior_0.25',0.25)]:
        print(f'building {name}...')
        variant=build_variant(paired,strength)
        d,s,f,m=evaluate_variant(name,variant,cohort,actual,base,market)
        all_detail.append(d); all_summary.append(s); all_fights.append(f); market_rows.append(m)
    detail=pd.concat(all_detail,ignore_index=True); summary=pd.concat(all_summary,ignore_index=True); fights=pd.concat(all_fights,ignore_index=True); msum=pd.DataFrame(market_rows)
    OUT.mkdir(parents=True,exist_ok=True); detail.to_csv(OUT/'runtime_output_fit_detail.csv',index=False); summary.to_csv(OUT/'runtime_output_fit_summary.csv',index=False); fights.to_csv(OUT/'fight_runtime_separation.csv',index=False); msum.to_csv(OUT/'market_separation_summary.csv',index=False)
    print('\nBANTAMWEIGHT FSR PRIOR-STRENGTH ATTRIBUTION — LEAKAGE SAFE')
    print(f'fights={cohort.fight_id.nunique()} priced={market.fight_id.nunique()}')
    print('\nRUNTIME OUTPUT FIT'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nMARKET SEPARATION'); print(msum.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nInterpretation rule: weaker priors matter only if they improve realized-output fit AND increase market-strength separation without merely exploding prediction SD.')
if __name__=='__main__': main()
