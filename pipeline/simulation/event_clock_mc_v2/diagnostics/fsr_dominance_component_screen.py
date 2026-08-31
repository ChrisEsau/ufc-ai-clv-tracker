"""Research-only matched-cohort screen of dominance components beyond FSR V3.

Splits prior-fight dominance into striking, damage/KD, wrestling/control,
submission pressure, finish dominance, and combined dominance. All information is
strictly pre-fight. No FSR, simulator, market, or raw data are modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, brier_score_loss, log_loss, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import build_two_way_market, choose_trait_columns, build_matchups, safe_logit
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_dominance_residual_audit import build_fight_dominance

ROUND_PATH=Path('data/fight_details/ufc_round_stats.parquet')
MASTER_PATH=Path('data/master/ufc_master.parquet')
MARKET_PATH=Path('data/market/historical_market_outcomes.parquet')

COMPONENTS={
    'striking':['margin_sig'],
    'damage':['margin_kd'],
    'wrestling_control':['margin_td','margin_ctrl'],
    'submission_pressure':['margin_sub'],
    'finish':['finish_win'],
    'combined':['margin_sig','margin_kd','margin_td','margin_ctrl','margin_sub','finish_win'],
}

def add_component_scores(d):
    d=d.copy()
    for name, cols in COMPONENTS.items():
        use=[c for c in cols if c in d.columns]
        if not use:
            continue
        if name=='finish':
            raw=d['finish_win'].astype(float)
        elif name=='combined':
            raw=(d.get('margin_sig',0.0)+2*d.get('margin_kd',0.0)+0.6*d.get('margin_td',0.0)+0.004*d.get('margin_ctrl',0.0)+0.8*d.get('margin_sub',0.0)+0.75*d.get('finish_win',0.0))
        elif name=='wrestling_control':
            raw=d.get('margin_td',0.0)+0.004*d.get('margin_ctrl',0.0)
        else:
            raw=sum((d[c].astype(float) for c in use), start=pd.Series(0.0,index=d.index))
        d[f'raw_{name}']=raw
        g=d.groupby(['division_key','era'])[f'raw_{name}']
        mu=g.transform('mean'); sd=g.transform('std').replace(0,np.nan)
        d[f'{name}_z']=((raw-mu)/sd).fillna(0.0)
    return d

def add_prefight_features(matchups,d):
    score_cols=[f'{k}_z' for k in COMPONENTS if f'{k}_z' in d.columns]
    by={fid:g.sort_values(['fight_date','fight_id']).reset_index(drop=True) for fid,g in d.groupby('fighter_id')}
    rows=[]
    for _,r in matchups.iterrows():
        rec=r.to_dict(); dt=pd.Timestamp(r['fight_date'])
        for label,fid in [('fav',str(r['favorite_id'])),('dog',str(r['underdog_id']))]:
            h=by.get(fid,pd.DataFrame())
            if not h.empty: h=h[h['fight_date']<dt]
            for c in score_cols:
                vals=h[c].to_numpy(float) if not h.empty else np.array([])
                rec[f'{label}_{c}_last1']=vals[-1] if len(vals) else np.nan
                rec[f'{label}_{c}_last3']=float(np.mean(vals[-3:])) if len(vals) else np.nan
                if len(vals):
                    w=np.array([0.5**i for i in range(len(vals)-1,-1,-1)],float); w/=w.sum()
                    rec[f'{label}_{c}_ewm']=float(np.sum(vals*w))
                else: rec[f'{label}_{c}_ewm']=np.nan
        for c in score_cols:
            for win in ('last1','last3','ewm'):
                a=rec[f'fav_{c}_{win}']; b=rec[f'dog_{c}_{win}']
                rec[f'delta_{c}_{win}']=a-b if pd.notna(a) and pd.notna(b) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows),score_cols

def fit_market(tr,te,features,label):
    ytr=safe_logit(tr['market_favorite_fair_p']); yte=safe_logit(te['market_favorite_fair_p'])
    m=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=10.0))]); m.fit(tr[features],ytr)
    pred=m.predict(te[features]); pp=1/(1+np.exp(-pred))
    return {'model':label,'feature_count':len(features),'test_r2_logit':r2_score(yte,pred),'test_rmse_logit':mean_squared_error(yte,pred)**0.5,'mean_abs_residual_pp':float(np.mean(np.abs(100*(te['market_favorite_fair_p'].to_numpy()-pp))))}

def fit_winner(tr,te,features,label):
    m=Pipeline([('scale',StandardScaler()),('lr',LogisticRegression(C=.25,max_iter=2000))]); m.fit(tr[features],tr['favorite_won'].astype(int))
    p=m.predict_proba(te[features])[:,1]; y=te['favorite_won'].astype(int).to_numpy()
    return {'model':label,'feature_count':len(features),'auc':roc_auc_score(y,p),'brier':brier_score_loss(y,p),'logloss':log_loss(y,p),'accuracy':accuracy_score(y,p>=.5)}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out-dir',type=Path,required=True); args=ap.parse_args()
    rounds=pd.read_parquet(ROUND_PATH); master=pd.read_parquet(MASTER_PATH); fsr=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    market=build_two_way_market(MARKET_PATH); traits=choose_trait_columns(fsr); matchups=build_matchups(market,fsr,master,traits)
    dom,usable,_=build_fight_dominance(rounds,master); dom=add_component_scores(dom)
    frame,score_cols=add_prefight_features(matchups,dom)
    fsr_features=[f'delta__{c}' for c in traits]
    component_sets={}
    for c in score_cols:
        component_sets[c[:-2]]=[f'delta_{c}_last1',f'delta_{c}_last3',f'delta_{c}_ewm']
    all_component_features=[x for v in component_sets.values() for x in v]
    required=fsr_features+all_component_features+['market_favorite_fair_p','favorite_won','fight_date']
    complete=frame.dropna(subset=required).sort_values(['fight_date','fight_id']).reset_index(drop=True)
    cut=int(len(complete)*.70); tr=complete.iloc[:cut].copy(); te=complete.iloc[cut:].copy()

    market_rows=[fit_market(tr,te,fsr_features,'fsr_only')]
    winner_rows=[fit_winner(tr,te,fsr_features,'fsr_only')]
    for name, feats in component_sets.items():
        market_rows.append(fit_market(tr,te,fsr_features+feats,f'fsr_plus_{name}'))
        winner_rows.append(fit_winner(tr,te,fsr_features+feats,f'fsr_plus_{name}'))
    market_rows.append(fit_market(tr,te,fsr_features+all_component_features,'fsr_plus_all_components'))
    winner_rows.append(fit_winner(tr,te,fsr_features+all_component_features,'fsr_plus_all_components'))
    mr=pd.DataFrame(market_rows); wr=pd.DataFrame(winner_rows)
    bmr=mr[mr.model=='fsr_only'].iloc[0]; bwr=wr[wr.model=='fsr_only'].iloc[0]
    mr['delta_r2_vs_fsr']=mr.test_r2_logit-bmr.test_r2_logit; mr['delta_rmse_vs_fsr']=mr.test_rmse_logit-bmr.test_rmse_logit
    wr['delta_auc_vs_fsr']=wr.auc-bwr.auc; wr['delta_brier_vs_fsr']=wr.brier-bwr.brier; wr['delta_logloss_vs_fsr']=wr.logloss-bwr.logloss
    mr=mr.sort_values('test_rmse_logit'); wr=wr.sort_values('brier')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    mr.to_csv(args.out_dir/'component_market_models.csv',index=False); wr.to_csv(args.out_dir/'component_winner_models.csv',index=False)
    pd.DataFrame([{'joined_fights':len(frame),'complete_fights':len(complete),'train_fights':len(tr),'test_fights':len(te),'cut_date':str(te.fight_date.min().date()),'usable_stats':','.join(usable)}]).to_csv(args.out_dir/'metadata.csv',index=False)
    print('FSR DOMINANCE COMPONENT MATCHED-COHORT SCREEN')
    print(f'joined={len(frame)} complete={len(complete)} train={len(tr)} test={len(te)} cut={te.fight_date.min().date()} stats={usable}')
    print('\nMARKET MODELS'); print(mr.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nACTUAL-WINNER MODELS'); print(wr.to_string(index=False,float_format=lambda x:f'{x:.5f}'))

if __name__=='__main__': main()
