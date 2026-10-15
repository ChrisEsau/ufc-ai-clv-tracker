"""Leakage-safe raw round-stat predictive audit for bantamweight.

Bypasses FSR trait construction. Uses only prior UFC observations derived from
ufc_round_stats.parquet via build_paired_rounds().

Questions:
1) Do cumulative prior raw standing/TD statistics predict the next fight's raw
   standing/TD outputs?
2) Do raw-stat matchup deltas predict held-out winners and historical market
   strength better than the current FSR representation?

Measurement only; production FSR/MC are untouched.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import build_two_way_market, MARKET_PATH

OUT = Path('data/diagnostics/event_clock_mc_v2/bantamweight_raw_round_stats_predictive_audit')
DIVISION = 'bantamweight'
EPS = 1e-9
KEYS = ['event_date','fight_id','fighter_id']

FSR_TRAITS = [
    'standing_striking_tendency','standing_striking_suppression','standing_striking_offense','standing_striking_defense',
    'takedown_tendency','takedown_suppression','takedown_offense','takedown_defense',
    'ground_striking_tendency','ground_striking_suppression','ground_striking_offense',
    'escape_tendency','escape_suppression','submission_tendency','submission_suppression',
    'striking_power','durability','knockdown_resistance',
]


def safe_div(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)


def build_fighter_fights() -> pd.DataFrame:
    p=build_paired_rounds().copy()
    p['event_date']=pd.to_datetime(p['event_date']).dt.normalize()
    p['fight_id']=p['fight_id'].astype(str); p['fighter_id']=p['fighter_id'].astype(str)
    required=['distance_attempted','distance_landed','standing_exposure_seconds','td_attempted','td_landed','td_tendency_exposure_seconds']
    missing=[c for c in required if c not in p.columns]
    if missing: raise ValueError(f'paired rounds missing required raw columns: {missing}')
    agg=(p.groupby(KEYS,as_index=False).agg(
        distance_attempted=('distance_attempted','sum'), distance_landed=('distance_landed','sum'),
        standing_exposure_seconds=('standing_exposure_seconds','sum'),
        td_attempted=('td_attempted','sum'), td_landed=('td_landed','sum'),
        td_exposure_seconds=('td_tendency_exposure_seconds','sum'),
    ))
    # Optional raw fields, if standardized source exposes them.
    optional={
        'ground_attempted':'sum','ground_landed':'sum','control_seconds':'sum',
        'knockdowns':'sum','submission_attempts':'sum',
    }
    for c,how in optional.items():
        if c in p.columns:
            x=p.groupby(KEYS,as_index=False)[c].agg(how)
            agg=agg.merge(x,on=KEYS,how='left',validate='one_to_one')
    return agg.sort_values(KEYS).reset_index(drop=True)


def add_prior_raw_states(ff: pd.DataFrame) -> pd.DataFrame:
    x=ff.sort_values(['fighter_id','event_date','fight_id']).copy()
    sum_cols=['distance_attempted','distance_landed','standing_exposure_seconds','td_attempted','td_landed','td_exposure_seconds']
    for c in ['ground_attempted','ground_landed','control_seconds','knockdowns','submission_attempts']:
        if c in x.columns: sum_cols.append(c)
    for c in sum_cols:
        x['prior_'+c]=x.groupby('fighter_id')[c].transform(lambda s:s.cumsum().shift(1)).fillna(0.0)
    x['prior_fights']=x.groupby('fighter_id').cumcount().astype(float)
    x['raw_standing_rate_15m']=safe_div(x['prior_distance_attempted']*900.0,x['prior_standing_exposure_seconds'])
    x['raw_standing_accuracy']=safe_div(x['prior_distance_landed'],x['prior_distance_attempted'])
    x['raw_td_rate_15m']=safe_div(x['prior_td_attempted']*900.0,x['prior_td_exposure_seconds'])
    x['raw_td_completion']=safe_div(x['prior_td_landed'],x['prior_td_attempted'])
    # Current-fight realized outputs for next-fight fit scoring.
    x['actual_standing_rate_15m']=safe_div(x['distance_attempted']*900.0,x['standing_exposure_seconds'])
    x['actual_standing_accuracy']=safe_div(x['distance_landed'],x['distance_attempted'])
    x['actual_td_rate_15m']=safe_div(x['td_attempted']*900.0,x['td_exposure_seconds'])
    x['actual_td_completion']=safe_div(x['td_landed'],x['td_attempted'])
    if 'ground_attempted' in x.columns:
        x['raw_ground_attempts_per_fight']=safe_div(x['prior_ground_attempted'],np.maximum(x['prior_fights'],1.0))
    if 'ground_landed' in x.columns and 'ground_attempted' in x.columns:
        x['raw_ground_accuracy']=safe_div(x['prior_ground_landed'],x['prior_ground_attempted'])
    if 'control_seconds' in x.columns:
        x['raw_control_seconds_per_fight']=safe_div(x['prior_control_seconds'],np.maximum(x['prior_fights'],1.0))
    if 'knockdowns' in x.columns:
        x['raw_kd_per_fight']=safe_div(x['prior_knockdowns'],np.maximum(x['prior_fights'],1.0))
    if 'submission_attempts' in x.columns:
        x['raw_sub_attempts_per_fight']=safe_div(x['prior_submission_attempts'],np.maximum(x['prior_fights'],1.0))
    return x


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
    raw_cols=[c for c in states.columns if c.startswith('raw_')]
    rows=[]
    for _,r in master.iterrows():
        fid=str(r['fight_id']); rk=(fid,str(r['r_id'])); bk=(fid,str(r['b_id']))
        if rk not in smap or bk not in smap: continue
        rr,bb=smap[rk],smap[bk]
        rec={'fight_id':fid,'event_date':r['event_date'],'red':r['r_name'],'blue':r['b_name'],'y_red':winner_red(r)}
        for c in raw_cols:
            av=getattr(rr,c); bv=getattr(bb,c)
            rec['raw_'+c[4:]+'_delta']=float(av)-float(bv) if pd.notna(av) and pd.notna(bv) else np.nan
        rec['raw_prior_fights_delta']=float(rr.prior_fights)-float(bb.prior_fights)
        # current FSR comparator on exactly the same fight rows
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


def output_fit(states):
    rows=[]
    pairs=[('raw_standing_rate_15m','actual_standing_rate_15m'),('raw_standing_accuracy','actual_standing_accuracy'),
           ('raw_td_rate_15m','actual_td_rate_15m'),('raw_td_completion','actual_td_completion')]
    for pred,act in pairs:
        z=states[['event_date',pred,act]].dropna().copy()
        if len(z)<10: continue
        cut=int(len(z)*.70); te=z.iloc[cut:]
        # direct raw cumulative predictor
        rows.append({'model':'raw_prior_state','metric':act,'n_test':len(te),'mae':mean_absolute_error(te[act],te[pred]),
                     'corr':te[pred].corr(te[act]),'pred_sd':te[pred].std(),'actual_sd':te[act].std()})
    return pd.DataFrame(rows)


def score(train,test,cols,name,kind='logit'):
    tr=train.dropna(subset=cols+['y_red']).copy(); te=test.dropna(subset=cols+['y_red']).copy()
    ytr=tr['y_red'].astype(int); yte=te['y_red'].astype(int).to_numpy()
    if kind=='gb':
        model=HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=200,l2_regularization=2.0,random_state=20260824)
    else:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=3000))
    model.fit(tr[cols],ytr); p=model.predict_proba(te[cols])[:,1]
    out={'arm':name,'n_train':len(tr),'n_test':len(te),'accuracy':accuracy_score(yte,p>=.5),'auc':roc_auc_score(yte,p),
         'brier':brier_score_loss(yte,p),'log_loss':log_loss(yte,np.clip(p,1e-6,1-1e-6))}
    d=te[['fight_id','event_date','red','blue','y_red']].copy(); d['arm']=name; d['p_red']=p
    for c in ['market_favorite_fair_p','red_is_market_favorite']:
        if c in te: d[c]=te[c].values
    z=d.dropna(subset=['market_favorite_fair_p','red_is_market_favorite']) if 'market_favorite_fair_p' in d else pd.DataFrame()
    if len(z)>2:
        fav=np.where(z['red_is_market_favorite']>.5,z['p_red'],1-z['p_red'])
        out['market_corr']=float(pd.Series(fav,index=z.index).corr(z['market_favorite_fair_p']))
        out['market_mae_pp']=float(100*np.mean(np.abs(fav-z['market_favorite_fair_p'].to_numpy())))
    else: out['market_corr']=np.nan; out['market_mae_pp']=np.nan
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
    ff=build_fighter_fights(); states=add_prior_raw_states(ff)
    market=build_two_way_market(MARKET_PATH).copy(); market['fight_id']=market['fight_id'].astype(str)
    fsr=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    frame=build_fight_frame(master,states,market,fsr).dropna(subset=['y_red']).sort_values(['event_date','fight_id']).reset_index(drop=True)
    raw_cols=[c for c in frame.columns if c.startswith('raw_') and c.endswith('_delta')]
    fsr_cols=[c for c in frame.columns if c.startswith('fsr_')]
    # Use only fights with at least one prior fight for each side: raw-history question is undefined otherwise.
    pf='raw_prior_fights_delta'
    # Individual raw features already become NaN on no-history; complete-case handles cold starts.
    cut=max(1,int(len(frame)*.70)); train=frame.iloc[:cut]; test=frame.iloc[cut:]
    arms=[('raw_logit',raw_cols,'logit'),('raw_gb',raw_cols,'gb'),('fsr_logit',fsr_cols,'logit')]
    summaries=[]; details=[]
    for name,cols,kind in arms:
        print(f'scoring {name} with {len(cols)} features...')
        s,d=score(train,test,cols,name,kind); summaries.append(s); details.append(d)
    summary=pd.DataFrame(summaries); detail=pd.concat(details,ignore_index=True); buckets=bucket_summary(detail)
    ofit=output_fit(states)
    OUT.mkdir(parents=True,exist_ok=True)
    states.to_csv(OUT/'fighter_fight_raw_prior_states.csv',index=False); frame.to_csv(OUT/'fight_features.csv',index=False)
    ofit.to_csv(OUT/'next_fight_raw_output_fit.csv',index=False); summary.to_csv(OUT/'heldout_summary.csv',index=False)
    detail.to_csv(OUT/'heldout_predictions.csv',index=False); buckets.to_csv(OUT/'market_bucket_summary.csv',index=False)
    print('\nBANTAMWEIGHT RAW ROUND-STATS PREDICTIVE AUDIT — LEAKAGE SAFE')
    print(f'fights={len(frame)} train={len(train)} test={len(test)} raw_features={len(raw_cols)} fsr_features={len(fsr_cols)}')
    print('\nNEXT-FIGHT RAW OUTPUT FIT'); print(ofit.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nHELDOUT WINNER / MARKET'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nMARKET BUCKETS'); print(buckets.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nInterpretation: if raw prior stats materially outperform FSR or recover the strong-favorite gradient, FSR transformation is discarding/compressing useful source signal. If raw stats are equally weak, the missing signal is not present in simple UFCStats aggregates alone.')

if __name__=='__main__': main()
