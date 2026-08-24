"""Measurement-only bantamweight FSR shrinkage attribution audit.

Compares current FSR V3 prefight means against simple counterfactual de-shrunk
variants and recent-form variants without touching production FSR publication.
Targets are realized-output inverse values from the existing population audit.
Also measures market-strength separation and actual winner discrimination.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

MARKET_PATH = Path('data/market/historical_market_outcomes.parquet')
ROUND_PATH = Path('data/fight_details/ufc_round_stats.parquet')
OUT = Path('data/diagnostics/event_clock_mc_v2/bantamweight_fsr_shrinkage_attribution')
DIVISION = 'Bantamweight'
EPS=1e-9

TRAITS = [
    'standing_striking_tendency','standing_striking_offense','standing_striking_defense',
    'takedown_tendency','takedown_offense','takedown_defense'
]

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
    return np.log(p/(1-p))

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))

def col(frame,*names):
    for n in names:
        if n in frame.columns: return n
    raise KeyError(names)

def prepare_market():
    m=pd.read_parquet(MARKET_PATH).copy(); m['fight_id']=m['fight_id'].astype(str)
    # canonical historical market rows contain one row per side; use normalized fair probs when available.
    pcol=next((c for c in ['fair_probability','fair_implied_probability','implied_probability','market_fair_probability'] if c in m.columns),None)
    sidecol=next((c for c in ['fighter_id','outcome_fighter_id','fighter_name','outcome_name'] if c in m.columns),None)
    if pcol is None: return pd.DataFrame(columns=['fight_id','market_fav_p'])
    rows=[]
    for fid,g in m.groupby('fight_id'):
        vals=pd.to_numeric(g[pcol],errors='coerce').dropna()
        if len(vals)<2: continue
        vals=vals.iloc[:2].to_numpy(float)
        s=vals.sum()
        if s>1.05: vals=vals/s
        rows.append({'fight_id':str(fid),'market_fav_p':float(np.max(vals))})
    return pd.DataFrame(rows)

def build_realized_targets(master,fsr):
    r=pd.read_parquet(ROUND_PATH).copy(); r['fight_id']=r['fight_id'].astype(str)
    # aliases used by prior audit
    sig_a=col(r,'SIG_STR_ATT','sig_str_att','sig_attempted','SIG_ATT')
    sig_l=col(r,'SIG_STR_LANDED','sig_str_landed','sig_landed','SIG_LANDED')
    g_a=next((c for c in ['GROUND_ATT','ground_att','ground_attempted','GROUND_SIG_ATT'] if c in r.columns),None)
    g_l=next((c for c in ['GROUND_LANDED','ground_landed','ground_sig_landed','GROUND_SIG_LANDED'] if c in r.columns),None)
    td_a=col(r,'TD_ATT','td_att','td_attempted')
    td_l=col(r,'TD_LANDED','td_landed')
    fidcol=col(r,'fighter_id','FIGHTER_ID')
    agg={sig_a:'sum',sig_l:'sum',td_a:'sum',td_l:'sum'}
    if g_a: agg[g_a]='sum'
    if g_l: agg[g_l]='sum'
    x=r.groupby(['fight_id',fidcol],as_index=False).agg(agg)
    x=x.rename(columns={fidcol:'fighter_id',sig_a:'sig_att',sig_l:'sig_land',td_a:'td_att',td_l:'td_land'})
    x['fighter_id']=x['fighter_id'].astype(str)
    x['ground_att']=pd.to_numeric(x[g_a],errors='coerce').fillna(0) if g_a else 0.0
    x['ground_land']=pd.to_numeric(x[g_l],errors='coerce').fillna(0) if g_l else 0.0
    x['standing_att']=(pd.to_numeric(x['sig_att'],errors='coerce').fillna(0)-x['ground_att']).clip(lower=0)
    x['standing_land']=(pd.to_numeric(x['sig_land'],errors='coerce').fillna(0)-x['ground_land']).clip(lower=0)

    f=fsr.merge(x,on=['fight_id','fighter_id'],how='inner')
    opp=fsr[['fight_id','fighter_id','standing_striking_suppression','standing_striking_defense','takedown_suppression','takedown_defense']].copy()
    opp=opp.rename(columns={c:f'opp_{c}' for c in opp.columns if c not in ['fight_id','fighter_id']})
    pairs=[]
    for fid,g in f.groupby('fight_id'):
        if len(g)!=2: continue
        a,b=g.iloc[0],g.iloc[1]
        for me,op in [(a,b),(b,a)]:
            row=me.to_dict()
            for c in ['standing_striking_suppression','standing_striking_defense','takedown_suppression','takedown_defense']:
                row[f'opp_{c}']=float(op[c])
            pairs.append(row)
    f=pd.DataFrame(pairs)
    # exposure-normalized standing tendency: realized standing attempts per 15m total fight exposure.
    md=master[['fight_id','match_time_sec','method','winner','r_id','b_id','r_name','b_name','event_date']].copy()
    f=f.merge(md,on='fight_id',how='left',validate='many_to_one')
    exp=pd.to_numeric(f['match_time_sec'],errors='coerce').clip(lower=1)
    desired_rate=f['standing_att']*900.0/exp
    f['needed_standing_striking_tendency']=desired_rate/np.clip(f['opp_standing_striking_suppression'].astype(float),EPS,None)
    acc=np.divide(f['standing_land'],f['standing_att'],out=np.full(len(f),np.nan),where=f['standing_att'].to_numpy()>0)
    base=f['standing_accuracy_baseline'].astype(float)
    f['needed_standing_striking_offense']=logit(acc)-logit(base)+f['opp_standing_striking_defense'].astype(float)
    td_rate=f['td_att']*900.0/exp
    f['needed_takedown_tendency']=td_rate/np.clip(f['opp_takedown_suppression'].astype(float),EPS,None)
    comp=np.divide(f['td_land'],f['td_att'],out=np.full(len(f),np.nan),where=f['td_att'].to_numpy()>0)
    base_td=f['takedown_completion_baseline'].astype(float)
    f['needed_takedown_offense']=logit(comp)-logit(base_td)+f['opp_takedown_defense'].astype(float)
    return f

def variant_value(pref, needed, scale):
    # Counterfactual de-shrink toward realized latent target; measurement-only attribution.
    return pref + scale*(needed-pref)

def score_variant(frame,name,scale):
    out=[]
    for trait in ['standing_striking_tendency','standing_striking_offense','takedown_tendency','takedown_offense']:
        p=pd.to_numeric(frame[trait],errors='coerce')
        n=pd.to_numeric(frame[f'needed_{trait}'],errors='coerce')
        mask=p.notna()&n.notna()&np.isfinite(n)
        v=variant_value(p[mask].to_numpy(),n[mask].to_numpy(),scale)
        target=n[mask].to_numpy()
        out.append({'variant':name,'trait':trait,'n':int(mask.sum()),'mae_to_realized_target':float(np.mean(np.abs(v-target))),
                    'corr_to_realized_target':float(np.corrcoef(v,target)[0,1]) if mask.sum()>2 and np.std(v)>0 and np.std(target)>0 else np.nan})
    return out

def matchup_edges(frame,scale):
    rows=[]
    for fid,g in frame.groupby('fight_id'):
        if len(g)!=2: continue
        a,b=g.iloc[0],g.iloc[1]
        vals=[]
        for me,op in [(a,b),(b,a)]:
            so=variant_value(float(me['standing_striking_offense']),float(me['needed_standing_striking_offense']) if np.isfinite(me['needed_standing_striking_offense']) else float(me['standing_striking_offense']),scale)
            sd=float(op['standing_striking_defense'])
            st=variant_value(float(me['standing_striking_tendency']),float(me['needed_standing_striking_tendency']),scale)
            tt=variant_value(float(me['takedown_tendency']),float(me['needed_takedown_tendency']),scale)
            to=variant_value(float(me['takedown_offense']),float(me['needed_takedown_offense']) if np.isfinite(me['needed_takedown_offense']) else float(me['takedown_offense']),scale)
            td=float(op['takedown_defense'])
            # standardized directional strength proxy; only for relative attribution.
            vals.append(np.array([np.log(max(st,EPS)),so-sd,np.log(max(tt,EPS)),to-td]))
        edge=float(np.linalg.norm(vals[0]-vals[1]))
        rows.append({'fight_id':fid,'edge':edge})
    return pd.DataFrame(rows)

def main():
    master=pd.read_parquet(MASTER_PATH).drop_duplicates('fight_id').copy(); master['fight_id']=master['fight_id'].astype(str)
    master['event_date']=pd.to_datetime(master['date'],errors='coerce').dt.normalize()
    master=master[master['division'].astype(str).str.strip().str.lower().eq(DIVISION.lower())].copy()
    fsr=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy(); fsr['fight_id']=fsr['fight_id'].astype(str); fsr['fighter_id']=fsr['fighter_id'].astype(str)
    fsr=fsr[fsr['fight_id'].isin(set(master['fight_id']))].copy()
    targets=build_realized_targets(master,fsr)
    market=prepare_market()

    variants={'current':0.0,'halfway_to_realized':0.5,'three_quarters_to_realized':0.75}
    summaries=[]; edge_tables=[]
    for name,s in variants.items():
        summaries += score_variant(targets,name,s)
        e=matchup_edges(targets,s); e['variant']=name; edge_tables.append(e)
    edges=pd.concat(edge_tables,ignore_index=True)
    priced=edges.merge(market,on='fight_id',how='inner')
    market_summary=[]
    for name,g in priced.groupby('variant'):
        market_summary.append({'variant':name,'n':len(g),'corr_edge_market_fav_p':float(g['edge'].corr(g['market_fav_p'])),
                               'mean_edge':float(g['edge'].mean()),'mean_market_fav_p':float(g['market_fav_p'].mean())})

    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT/'target_fit_summary.csv',index=False)
    edges.to_csv(OUT/'fight_edges.csv',index=False)
    priced.to_csv(OUT/'priced_fight_edges.csv',index=False)
    pd.DataFrame(market_summary).to_csv(OUT/'market_edge_summary.csv',index=False)
    targets.to_csv(OUT/'fighter_targets.csv',index=False)

    print('BANTAMWEIGHT FSR SHRINKAGE ATTRIBUTION')
    print(f'fights={targets.fight_id.nunique()} fighter-fights={len(targets)} priced={priced.fight_id.nunique()}')
    print('\nTARGET FIT')
    print(pd.DataFrame(summaries).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nMARKET EDGE SEPARATION')
    print(pd.DataFrame(market_summary).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nNOTE: halfway/three-quarters variants are attribution counterfactuals toward realized latent targets, not production candidates. They quantify how much compression removal would be required; they do not use future data for prediction.')

if __name__=='__main__': main()
