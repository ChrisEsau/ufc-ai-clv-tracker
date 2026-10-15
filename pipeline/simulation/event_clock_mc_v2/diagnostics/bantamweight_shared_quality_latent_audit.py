"""Leakage-safe Stage-3 shared fighter-quality audit for bantamweight.

Measurement only. Production FSR and Event Clock mechanics are untouched.

Builds two prefight shared-quality signals chronologically:
1) outcome Elo from prior UFC fight results;
2) performance quality from prior realized fight-flow dominance, updated only
   after each event and centered within each fight.

Then asks whether adding those signals to current FSR matchup deltas improves
held-out winner prediction and whether predicted matchup strength better tracks
historical offered/legacy-consensus market favorite probability.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import build_two_way_market, MARKET_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.bantamweight_fsr_population_inverse_market_audit import ROUND_STATS, pick_col, fighter_rows, round_totals

OUT = Path('data/diagnostics/event_clock_mc_v2/bantamweight_shared_quality_latent_audit')
DIVISION = 'bantamweight'
ELO_K = 24.0
PERF_ALPHA = 0.35
EPS = 1e-9

# Use the validated simulator-facing means, excluding uncertainty columns/baselines.
FSR_TRAITS = [
    'standing_striking_tendency','standing_striking_suppression','standing_striking_offense','standing_striking_defense',
    'takedown_tendency','takedown_suppression','takedown_offense','takedown_defense',
    'ground_striking_tendency','ground_striking_suppression','ground_striking_offense',
    'escape_tendency','escape_suppression','submission_tendency','submission_suppression',
    'striking_power','durability','knockdown_resistance',
]


def elo_expected(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((b-a)/400.0))


def safe_share(a: float, b: float) -> float:
    den = abs(a) + abs(b)
    return 0.0 if den <= 0 else (a-b)/den


def build_actual_lookup(master: pd.DataFrame) -> dict[str, dict[str, dict[str,float]]]:
    rs = pd.read_parquet(ROUND_STATS).copy()
    fcol = pick_col(rs,'fight_id','bout_id'); rs[fcol] = rs[fcol].astype(str)
    out = {}
    for _, mr in master.iterrows():
        fid = str(mr['fight_id']); fr = rs[rs[fcol].eq(fid)]
        rr, br = fighter_rows(fr, mr)
        if rr.empty or br.empty: continue
        out[fid] = {'red':round_totals(rr), 'blue':round_totals(br)}
    return out


def realized_performance_margin(a: dict, b: dict) -> float:
    """Bounded, winner-independent realized fight-flow margin in [-1,1] approx.

    Components are intentionally simple and symmetric. This is a diagnostic
    latent, not a proposed production formula.
    """
    sig = safe_share(a['sig_land'], b['sig_land'])
    td = safe_share(a['td_land'], b['td_land'])
    ground = safe_share(a['ground_land'], b['ground_land'])
    # attempts add style/initiative information without dominating effectiveness.
    td_att = safe_share(a['td_att'], b['td_att'])
    return float(0.60*sig + 0.20*td + 0.10*ground + 0.10*td_att)


def build_quality_history(master: pd.DataFrame, actual: dict) -> pd.DataFrame:
    elo: dict[str,float] = {}
    perf: dict[str,float] = {}
    rows=[]
    for event_date, batch in master.sort_values(['event_date','fight_id']).groupby('event_date', sort=True):
        pending=[]
        for _, r in batch.iterrows():
            rid,bid=str(r['r_id']),str(r['b_id']); fid=str(r['fight_id'])
            re,be=elo.get(rid,1500.0),elo.get(bid,1500.0)
            rp,bp=perf.get(rid,0.0),perf.get(bid,0.0)
            rows.append({'fight_id':fid,'event_date':event_date,'red_elo':re,'blue_elo':be,'elo_edge':re-be,
                         'red_perf_quality':rp,'blue_perf_quality':bp,'perf_edge':rp-bp})
            winner=str(r.get('winner',''))
            if winner == str(r.get('r_name','')) or winner == rid:
                yred=1.0
            elif winner == str(r.get('b_name','')) or winner == bid:
                yred=0.0
            else:
                yred=0.5
            pm = np.nan
            if fid in actual:
                pm=realized_performance_margin(actual[fid]['red'],actual[fid]['blue'])
            pending.append((rid,bid,re,be,rp,bp,yred,pm))
        # Same-event delayed updates.
        for rid,bid,re,be,rp,bp,yred,pm in pending:
            er=elo_expected(re,be)
            elo[rid]=re + ELO_K*(yred-er)
            elo[bid]=be + ELO_K*((1.0-yred)-(1.0-er))
            if np.isfinite(pm):
                perf[rid]=(1-PERF_ALPHA)*rp + PERF_ALPHA*float(pm)
                perf[bid]=(1-PERF_ALPHA)*bp - PERF_ALPHA*float(pm)
    return pd.DataFrame(rows)


def build_fight_frame(master: pd.DataFrame, fsr: pd.DataFrame, quality: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    fsr=fsr.copy(); fsr['fight_id']=fsr['fight_id'].astype(str); fsr['fighter_id']=fsr['fighter_id'].astype(str)
    usable=[c for c in FSR_TRAITS if c in fsr.columns]
    rows=[]
    q=quality.set_index('fight_id')
    m=market.set_index('fight_id') if len(market) else pd.DataFrame()
    for _,r in master.iterrows():
        fid=str(r['fight_id']); g=fsr[fsr['fight_id'].eq(fid)]
        if len(g)!=2 or fid not in q.index: continue
        rr=g[g['fighter_id'].eq(str(r['r_id']))]; bb=g[g['fighter_id'].eq(str(r['b_id']))]
        if rr.empty or bb.empty: continue
        rr=rr.iloc[0]; bb=bb.iloc[0]
        rec={'fight_id':fid,'event_date':r['event_date'],'red':r['r_name'],'blue':r['b_name']}
        winner=str(r.get('winner',''))
        rec['y_red']=1 if winner in (str(r['r_id']),str(r['r_name'])) else 0 if winner in (str(r['b_id']),str(r['b_name'])) else np.nan
        for c in usable:
            av=float(rr[c]) if pd.notna(rr[c]) else np.nan; bv=float(bb[c]) if pd.notna(bb[c]) else np.nan
            # log-ratio for positive rate/multiplier traits, simple delta otherwise.
            if c in ('standing_striking_tendency','standing_striking_suppression','takedown_tendency','takedown_suppression','ground_striking_tendency','ground_striking_suppression','escape_tendency','escape_suppression','submission_tendency','submission_suppression') and av>0 and bv>0:
                rec['fsr_'+c]=math.log(av/bv)
            else:
                rec['fsr_'+c]=av-bv
        qr=q.loc[fid]
        rec['elo_edge']=float(qr['elo_edge']); rec['perf_edge']=float(qr['perf_edge'])
        if len(market) and fid in m.index:
            mr=m.loc[fid]
            if isinstance(mr,pd.DataFrame): mr=mr.iloc[0]
            rec['market_favorite_fair_p']=float(mr['market_favorite_fair_p'])
            rec['market_favorite_id']=str(mr['favorite_id'])
            rec['red_is_market_favorite']=float(str(mr['favorite_id'])==str(r['r_id']))
        rows.append(rec)
    return pd.DataFrame(rows)


def score_arm(train, test, feature_cols, name):
    tr=train.dropna(subset=feature_cols+['y_red']).copy(); te=test.dropna(subset=feature_cols+['y_red']).copy()
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000))
    model.fit(tr[feature_cols],tr['y_red'].astype(int))
    p=model.predict_proba(te[feature_cols])[:,1]; y=te['y_red'].astype(int).to_numpy()
    pred=(p>=.5).astype(int)
    out={'arm':name,'n_train':len(tr),'n_test':len(te),'accuracy':accuracy_score(y,pred),'auc':roc_auc_score(y,p),
         'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-6,1-1e-6))}
    detail=te[['fight_id','event_date','red','blue','y_red']].copy(); detail['arm']=name; detail['p_red']=p
    for c in ['market_favorite_fair_p','market_favorite_id','red_is_market_favorite']:
        if c in te.columns: detail[c]=te[c].values
    if 'market_favorite_fair_p' in detail.columns:
        z=detail.dropna(subset=['market_favorite_fair_p','red_is_market_favorite']).copy()
        z['model_fav_p']=np.where(z['red_is_market_favorite']>0.5,z['p_red'],1-z['p_red'])
        if len(z)>2:
            out['market_corr']=float(z['model_fav_p'].corr(z['market_favorite_fair_p']))
            out['market_mae_pp']=float(100*(z['model_fav_p']-z['market_favorite_fair_p']).abs().mean())
        else:
            out['market_corr']=np.nan; out['market_mae_pp']=np.nan
    return out,detail


def bucket_summary(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty or 'market_favorite_fair_p' not in details.columns: return pd.DataFrame()
    x=details.dropna(subset=['market_favorite_fair_p','red_is_market_favorite']).copy()
    x['model_fav_p']=np.where(x['red_is_market_favorite']>0.5,x['p_red'],1-x['p_red'])
    x['favorite_won']=np.where(x['red_is_market_favorite']>0.5,x['y_red'],1-x['y_red'])
    bins=[.5,.6,.7,.8,.9,1.0001]; labels=['50-60','60-70','70-80','80-90','90+']
    x['market_bucket']=pd.cut(x['market_favorite_fair_p'],bins=bins,labels=labels,right=False)
    return x.groupby(['arm','market_bucket'],observed=True).agg(n=('fight_id','size'),market_mean=('market_favorite_fair_p','mean'),model_mean=('model_fav_p','mean'),actual_fav_win=('favorite_won','mean')).reset_index()


def main():
    master=pd.read_parquet(MASTER_PATH).drop_duplicates('fight_id').copy(); master['fight_id']=master['fight_id'].astype(str)
    master['event_date']=pd.to_datetime(master['date'],errors='coerce').dt.normalize()
    master=master[master['division'].astype(str).str.strip().str.lower().eq(DIVISION)].copy()
    master=master.sort_values(['event_date','fight_id']).reset_index(drop=True)
    fsr=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    market=build_two_way_market(MARKET_PATH).copy(); market['fight_id']=market['fight_id'].astype(str)
    actual=build_actual_lookup(master); quality=build_quality_history(master,actual)
    frame=build_fight_frame(master,fsr,quality,market)
    frame=frame.dropna(subset=['y_red']).sort_values(['event_date','fight_id']).reset_index(drop=True)
    fsr_cols=[c for c in frame.columns if c.startswith('fsr_')]
    cut=max(1,int(len(frame)*0.70)); train=frame.iloc[:cut].copy(); test=frame.iloc[cut:].copy()
    arms={
        'fsr_only':fsr_cols,
        'fsr_plus_elo':fsr_cols+['elo_edge'],
        'fsr_plus_performance':fsr_cols+['perf_edge'],
        'fsr_plus_both':fsr_cols+['elo_edge','perf_edge'],
        'elo_only':['elo_edge'],
        'performance_only':['perf_edge'],
    }
    summaries=[]; details=[]
    for name,cols in arms.items():
        print(f'scoring {name}...')
        s,d=score_arm(train,test,cols,name); summaries.append(s); details.append(d)
    summary=pd.DataFrame(summaries); detail=pd.concat(details,ignore_index=True); buckets=bucket_summary(detail)
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/'fight_features.csv',index=False); quality.to_csv(OUT/'quality_history.csv',index=False)
    summary.to_csv(OUT/'heldout_summary.csv',index=False); detail.to_csv(OUT/'heldout_predictions.csv',index=False); buckets.to_csv(OUT/'market_bucket_summary.csv',index=False)
    print('\nBANTAMWEIGHT SHARED QUALITY LATENT AUDIT — LEAKAGE SAFE')
    print(f'fights={len(frame)} train={len(train)} test={len(test)} cutoff={test.event_date.min().date() if len(test) else None}')
    print('\nHELDOUT'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nMARKET BUCKETS'); print(buckets.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nInterpretation: a shared-quality latent is useful only if adding it to FSR improves held-out winner proper scores and restores the market-strength gradient; standalone market correlation is not sufficient.')

if __name__=='__main__': main()
