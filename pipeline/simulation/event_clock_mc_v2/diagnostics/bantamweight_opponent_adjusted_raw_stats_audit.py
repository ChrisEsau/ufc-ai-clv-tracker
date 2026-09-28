"""Leakage-safe one-pass opponent-adjusted raw round-stat audit for bantamweight.

Measurement only. Production FSR and Event Clock mechanics are untouched.

For each completed historical fight, measure a fighter's realized technical
output relative to what that opponent had allowed before the fight. Also measure
how much the defender allowed relative to what that attacker had produced before
the fight. Same-event updates are delayed.

Compare:
- simple prior raw averages;
- opponent-adjusted residual states;
- current FSR V3 matchup deltas.

No market information is used to construct or fit the raw states.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import build_two_way_market, MARKET_PATH

OUT = Path('data/diagnostics/event_clock_mc_v2/bantamweight_opponent_adjusted_raw_stats_audit')
DIVISION = 'bantamweight'
KEYS = ['event_date','fight_id','fighter_id','opponent_id']
METRICS = ['standing_rate','standing_accuracy','td_rate','td_completion']

FSR_TRAITS = [
    'standing_striking_tendency','standing_striking_suppression','standing_striking_offense','standing_striking_defense',
    'takedown_tendency','takedown_suppression','takedown_offense','takedown_defense',
    'ground_striking_tendency','ground_striking_suppression','ground_striking_offense',
    'escape_tendency','escape_suppression','submission_tendency','submission_suppression',
    'striking_power','durability','knockdown_resistance',
]


def safe_ratio(a,b):
    return np.nan if b is None or not np.isfinite(b) or b <= 0 else float(a)/float(b)


def build_fighter_fights() -> pd.DataFrame:
    p = build_paired_rounds().copy()
    p['event_date'] = pd.to_datetime(p['event_date']).dt.normalize()
    for c in ['fight_id','fighter_id','opponent_id']:
        p[c] = p[c].astype(str)
    req = ['distance_attempted','distance_landed','standing_exposure_seconds','td_attempted','td_landed','td_tendency_exposure_seconds']
    miss = [c for c in req if c not in p.columns]
    if miss:
        raise ValueError(f'paired rounds missing required raw columns: {miss}')
    x = p.groupby(KEYS, as_index=False).agg(
        distance_attempted=('distance_attempted','sum'),
        distance_landed=('distance_landed','sum'),
        standing_exposure_seconds=('standing_exposure_seconds','sum'),
        td_attempted=('td_attempted','sum'),
        td_landed=('td_landed','sum'),
        td_exposure_seconds=('td_tendency_exposure_seconds','sum'),
    )
    x['standing_rate'] = np.where(x['standing_exposure_seconds']>0, x['distance_attempted']*900.0/x['standing_exposure_seconds'], np.nan)
    x['standing_accuracy'] = np.where(x['distance_attempted']>0, x['distance_landed']/x['distance_attempted'], np.nan)
    x['td_rate'] = np.where(x['td_exposure_seconds']>0, x['td_attempted']*900.0/x['td_exposure_seconds'], np.nan)
    x['td_completion'] = np.where(x['td_attempted']>0, x['td_landed']/x['td_attempted'], np.nan)
    return x.sort_values(KEYS).reset_index(drop=True)


def mean_state(state: dict[str, dict[str,list[float]]], fighter: str, metric: str) -> float:
    vals = state.get(fighter, {}).get(metric, [])
    return float(np.mean(vals)) if vals else np.nan


def append_state(state: dict[str, dict[str,list[float]]], fighter: str, metric: str, value: float):
    if not np.isfinite(value):
        return
    state.setdefault(fighter, {}).setdefault(metric, []).append(float(value))


def population_means(history: dict[str,list[float]]) -> dict[str,float]:
    return {m:(float(np.mean(history[m])) if history.get(m) else np.nan) for m in METRICS}


def replay_states(ff: pd.DataFrame) -> pd.DataFrame:
    # Produced = fighter's own raw historical outputs.
    produced: dict[str,dict[str,list[float]]] = {}
    # Allowed = realized opponent output while facing this defender.
    allowed: dict[str,dict[str,list[float]]] = {}
    # Opponent-adjusted residuals accumulated by fighter.
    offense_resid: dict[str,dict[str,list[float]]] = {}
    defense_resid: dict[str,dict[str,list[float]]] = {}
    population_history = {m:[] for m in METRICS}
    rows=[]

    for event_date, batch in ff.groupby('event_date', sort=True):
        pop = population_means(population_history)
        pending=[]
        for r in batch.to_dict('records'):
            fighter=str(r['fighter_id']); opp=str(r['opponent_id'])
            rec={'event_date':event_date,'fight_id':str(r['fight_id']),'fighter_id':fighter,'opponent_id':opp}
            for m in METRICS:
                actual=float(r[m]) if pd.notna(r[m]) else np.nan
                own_prior=mean_state(produced,fighter,m)
                opp_allowed_prior=mean_state(allowed,opp,m)
                opp_produced_prior=mean_state(produced,opp,m)
                # Population fallback prevents cold-start opponent baselines from dropping rows,
                # while residual history itself remains strictly prior-only.
                baseline_allowed=opp_allowed_prior if np.isfinite(opp_allowed_prior) else pop[m]
                baseline_produced=opp_produced_prior if np.isfinite(opp_produced_prior) else pop[m]
                off_r=actual-baseline_allowed if np.isfinite(actual) and np.isfinite(baseline_allowed) else np.nan
                # Positive defensive residual means this defender allowed MORE than the attacker's norm (bad defense).
                def_r=actual-baseline_produced if np.isfinite(actual) and np.isfinite(baseline_produced) else np.nan
                rec[f'raw_{m}']=own_prior
                rec[f'oppadj_off_{m}']=mean_state(offense_resid,fighter,m)
                rec[f'oppadj_def_allowed_{m}']=mean_state(defense_resid,fighter,m)
                rec[f'current_{m}']=actual
                pending.append((fighter,opp,m,actual,off_r,def_r))
            rec['prior_fights']=len(produced.get(fighter,{}).get('standing_rate',[]))
            rows.append(rec)

        # Same-event delayed update.
        for fighter,opp,m,actual,off_r,def_r in pending:
            if not np.isfinite(actual):
                continue
            append_state(produced,fighter,m,actual)
            append_state(allowed,opp,m,actual)
            append_state(offense_resid,fighter,m,off_r)
            append_state(defense_resid,opp,m,def_r)
            population_history[m].append(actual)

    return pd.DataFrame(rows).sort_values(['event_date','fight_id','fighter_id']).reset_index(drop=True)


def winner_red(r):
    w=str(r.get('winner',''))
    if w in (str(r.get('r_id','')),str(r.get('r_name',''))): return 1.0
    if w in (str(r.get('b_id','')),str(r.get('b_name',''))): return 0.0
    return np.nan


def build_fight_frame(master, states, market, fsr):
    smap={(str(r.fight_id),str(r.fighter_id)):r for r in states.itertuples(index=False)}
    mmap=market.set_index('fight_id') if len(market) else None
    fsr=fsr.copy(); fsr['fight_id']=fsr['fight_id'].astype(str); fsr['fighter_id']=fsr['fighter_id'].astype(str)
    usable_fsr=[c for c in FSR_TRAITS if c in fsr.columns]
    rows=[]
    for _,r in master.iterrows():
        fid=str(r['fight_id']); rk=(fid,str(r['r_id'])); bk=(fid,str(r['b_id']))
        if rk not in smap or bk not in smap: continue
        rr,bb=smap[rk],smap[bk]
        rec={'fight_id':fid,'event_date':r['event_date'],'red':r['r_name'],'blue':r['b_name'],'y_red':winner_red(r)}
        for m in METRICS:
            rav=getattr(rr,f'raw_{m}'); bav=getattr(bb,f'raw_{m}')
            rec[f'raw_{m}_delta']=float(rav)-float(bav) if pd.notna(rav) and pd.notna(bav) else np.nan
            ro=getattr(rr,f'oppadj_off_{m}'); bo=getattr(bb,f'oppadj_off_{m}')
            rd=getattr(rr,f'oppadj_def_allowed_{m}'); bd=getattr(bb,f'oppadj_def_allowed_{m}')
            rec[f'oa_off_{m}_delta']=float(ro)-float(bo) if pd.notna(ro) and pd.notna(bo) else np.nan
            rec[f'oa_def_{m}_delta']=float(rd)-float(bd) if pd.notna(rd) and pd.notna(bd) else np.nan
            # Directional matchup edge: red offense quality + blue defensive weakness
            # minus blue offense quality + red defensive weakness.
            rec[f'oa_matchup_{m}']=(float(ro)+float(bd)-float(bo)-float(rd)) if all(pd.notna(v) for v in [ro,bo,rd,bd]) else np.nan
        rec['prior_fights_red']=float(rr.prior_fights); rec['prior_fights_blue']=float(bb.prior_fights)
        fg=fsr[fsr['fight_id'].eq(fid)]
        fr=fg[fg['fighter_id'].eq(str(r['r_id']))]; fb=fg[fg['fighter_id'].eq(str(r['b_id']))]
        if len(fr) and len(fb):
            fr=fr.iloc[0]; fb=fb.iloc[0]
            for c in usable_fsr:
                if pd.notna(fr[c]) and pd.notna(fb[c]): rec['fsr_'+c]=float(fr[c])-float(fb[c])
        if mmap is not None and fid in mmap.index:
            mr=mmap.loc[fid]; mr=mr.iloc[0] if isinstance(mr,pd.DataFrame) else mr
            rec['market_favorite_fair_p']=float(mr['market_favorite_fair_p'])
            rec['red_is_market_favorite']=float(str(mr['favorite_id'])==str(r['r_id']))
        rows.append(rec)
    return pd.DataFrame(rows)


def score(train,test,cols,name):
    tr=train.dropna(subset=cols+['y_red']).copy(); te=test.dropna(subset=cols+['y_red']).copy()
    if len(tr)<50 or len(te)<30:
        raise ValueError(f'{name} insufficient complete cases train={len(tr)} test={len(te)}')
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=3000))
    model.fit(tr[cols],tr['y_red'].astype(int)); p=model.predict_proba(te[cols])[:,1]
    y=te['y_red'].astype(int).to_numpy()
    out={'arm':name,'n_train':len(tr),'n_test':len(te),'accuracy':accuracy_score(y,p>=.5),'auc':roc_auc_score(y,p),
         'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-6,1-1e-6))}
    d=te[['fight_id','event_date','red','blue','y_red']].copy(); d['arm']=name; d['p_red']=p
    for c in ['market_favorite_fair_p','red_is_market_favorite']:
        if c in te.columns: d[c]=te[c].values
    z=d.dropna(subset=['market_favorite_fair_p','red_is_market_favorite']).copy()
    if len(z)>2:
        z['model_fav_p']=np.where(z['red_is_market_favorite']>.5,z['p_red'],1-z['p_red'])
        out['market_corr']=float(z['model_fav_p'].corr(z['market_favorite_fair_p']))
        out['market_mae_pp']=float(100*(z['model_fav_p']-z['market_favorite_fair_p']).abs().mean())
    else:
        out['market_corr']=np.nan; out['market_mae_pp']=np.nan
    return out,d


def bucket_summary(detail):
    x=detail.dropna(subset=['market_favorite_fair_p','red_is_market_favorite']).copy()
    x['model_fav_p']=np.where(x['red_is_market_favorite']>.5,x['p_red'],1-x['p_red'])
    x['favorite_won']=np.where(x['red_is_market_favorite']>.5,x['y_red'],1-x['y_red'])
    x['bucket']=pd.cut(x['market_favorite_fair_p'],[.5,.6,.7,.8,.9,1.001],labels=['50-60','60-70','70-80','80-90','90+'],right=False)
    return x.groupby(['arm','bucket'],observed=True).agg(n=('fight_id','size'),market_mean=('market_favorite_fair_p','mean'),model_mean=('model_fav_p','mean'),actual_fav_win=('favorite_won','mean')).reset_index()


def main():
    master=pd.read_parquet(MASTER_PATH).drop_duplicates('fight_id').copy(); master['fight_id']=master['fight_id'].astype(str)
    master['event_date']=pd.to_datetime(master['date'],errors='coerce').dt.normalize()
    master=master[master['division'].astype(str).str.strip().str.lower().eq(DIVISION)].sort_values(['event_date','fight_id']).copy()
    ff=build_fighter_fights(); states=replay_states(ff)
    market=build_two_way_market(MARKET_PATH).copy(); market['fight_id']=market['fight_id'].astype(str)
    fsr=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    frame=build_fight_frame(master,states,market,fsr).dropna(subset=['y_red']).sort_values(['event_date','fight_id']).reset_index(drop=True)

    raw_cols=[f'raw_{m}_delta' for m in METRICS]
    oa_cols=[c for c in frame.columns if c.startswith('oa_')]
    matchup_cols=[f'oa_matchup_{m}' for m in METRICS]
    fsr_cols=[c for c in frame.columns if c.startswith('fsr_')]

    # Chronological split is defined once on the full fight timeline.
    cut=max(1,int(len(frame)*.70)); train=frame.iloc[:cut].copy(); test=frame.iloc[cut:].copy()

    arms=[
        ('raw_simple',raw_cols),
        ('oppadj_matchup_only',matchup_cols),
        ('oppadj_full',oa_cols),
        ('raw_plus_oppadj',raw_cols+oa_cols),
        ('fsr_only',fsr_cols),
    ]
    summaries=[]; details=[]
    for name,cols in arms:
        print(f'scoring {name} with {len(cols)} features...')
        s,d=score(train,test,cols,name); summaries.append(s); details.append(d)
    summary=pd.DataFrame(summaries); detail=pd.concat(details,ignore_index=True); buckets=bucket_summary(detail)

    # Strict apples-to-apples subset where every arm has all features.
    all_cols=sorted(set(raw_cols+oa_cols+fsr_cols))
    common=frame.dropna(subset=all_cols+['y_red']).copy().sort_values(['event_date','fight_id']).reset_index(drop=True)
    ccut=max(1,int(len(common)*.70)); ctr=common.iloc[:ccut]; cte=common.iloc[ccut:]
    common_rows=[]
    for name,cols in arms:
        s,_=score(ctr,cte,cols,name); s['comparison']='common_complete_case'; common_rows.append(s)
    common_summary=pd.DataFrame(common_rows)

    OUT.mkdir(parents=True,exist_ok=True)
    states.to_csv(OUT/'fighter_fight_opponent_adjusted_states.csv',index=False)
    frame.to_csv(OUT/'fight_features.csv',index=False)
    summary.to_csv(OUT/'heldout_summary.csv',index=False)
    common_summary.to_csv(OUT/'common_complete_case_summary.csv',index=False)
    detail.to_csv(OUT/'heldout_predictions.csv',index=False)
    buckets.to_csv(OUT/'market_bucket_summary.csv',index=False)

    print('\nBANTAMWEIGHT OPPONENT-ADJUSTED RAW STATS AUDIT — LEAKAGE SAFE')
    print(f'fights={len(frame)} train={len(train)} test={len(test)} common_fights={len(common)} common_test={len(cte)}')
    print('\nHELDOUT'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nCOMMON COMPLETE CASE'); print(common_summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nMARKET BUCKETS'); print(buckets.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nInterpretation: opponent adjustment is supported only if it improves held-out winner proper scores and/or strong-favorite market gradient versus simple raw averages on the common comparison set. Market is never a fitting target.')

if __name__=='__main__':
    main()
