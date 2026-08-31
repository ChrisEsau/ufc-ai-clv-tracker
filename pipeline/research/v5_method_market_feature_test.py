from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

OUT=Path('data/research/prop_mispricing'); OUT.mkdir(parents=True,exist_ok=True)
FEATURES=['reach_diff','recent_form_recent_avg_fight_time_diff','age_diff','ewm_sapm_diff','ewm_recent_sapm_diff','style_ko_finisher_score_diff','ewm_td_acc_diff','recent_finish_rate_diff','chin_risk_diff','recent_form_avg_opponent_elo_diff','recent_avg_fight_time_diff','aggression_index_diff','age_squared_diff','sapm_diff','ewm_kd_avg_diff','style_all_round_finisher_score_diff','recent_form_kd_absorbed_avg_diff','ewm_recent_splm_diff','elo_diff','ewm_elo_diff','ewm_recent_td_avg_diff','days_since_last_fight_diff','td_avg_diff','style_score_spread_diff','ko_dependency_diff','recent_form_avg_fight_time_diff','wrestling_mismatch_diff','win_pct_diff','recent_form_ko_rate_diff','recent_form_worst_loss_elo_diff','age_x_career_ko_losses_diff','ewm_str_def_diff','losses_diff','ewm_recent_win_pct_diff','avg_opponent_elo_diff','ewm_td_avg_diff','avg_fight_time_diff','ewm_days_since_last_fight_diff','pressure_striking_adv_diff','weight_diff','ctrl_against_per_min_diff','ewm_finish_loss_rate_diff','ewm_win_pct_diff','victory_concentration_index_diff','recent_form_td_acc_diff','sub_avg_diff','recent_form_best_win_elo_diff','ewm_best_win_elo_diff','style_primary_score_diff','recent_form_recent_finish_rate_diff','market_overround']
METHOD_FEATURES=['method_implied_red_win_p','method_vs_ml_red_gap','method_inside_distance_p','method_win_concentration_diff']
PARAMS={'max_depth':1,'eta':0.03,'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':10,'lambda':8.0,'alpha':1.0,'objective':'binary:logistic','eval_metric':'logloss','seed':42,'nthread':2}
FOLDS=[('2021','2020-12-31','2021-01-01','2021-12-31'),('2022','2021-12-31','2022-01-01','2022-12-31'),('2023','2022-12-31','2023-01-01','2023-12-31'),('2024','2023-12-31','2024-01-01','2024-12-31')]
METHOD_KEYS=['win_by_ko_tko_dq','win_by_submission','win_by_decision']

def clip(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def logit(p): p=clip(p); return np.log(p/(1-p))
def sigmoid(z): z=np.clip(np.asarray(z,float),-30,30); return 1/(1+np.exp(-z))
def met(y,p):
    y=np.asarray(y,int); p=clip(p)
    return {'n':int(len(y)),'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}

m=pd.read_parquet('data/market/historical_market_outcomes.parquet').copy()
m['date']=pd.to_datetime(m.date,errors='coerce')
m['implied_probability']=pd.to_numeric(m.implied_probability,errors='coerce')
m=m[(m.bookmaker=='legacy_consensus')&(m.result_status=='graded')&m.won.notna()].dropna(subset=['date','fight_id','implied_probability']).copy()
m['won']=m.won.astype(bool).astype(int)

# Moneyline anchor: exactly frozen V5 construction.
ml=m[m.market_key=='moneyline'].copy(); good=ml.groupby('fight_id').size(); good=good[good==2].index; ml=ml[ml.fight_id.isin(good)].copy()
ml['market_overround']=ml.groupby('fight_id').implied_probability.transform('sum'); ml['fair_market_p']=ml.implied_probability/ml.market_overround
red=ml[ml.outcome_side.astype(str).eq('red')].copy()

# Six-way method market. Require one red and one blue quote for each of KO/SUB/DEC.
mm=m[m.market_key.isin(METHOD_KEYS)].copy()
counts=mm.groupby(['fight_id','market_key','outcome_side']).size().unstack(fill_value=0)
# Keep fights with exactly one quote in each required key/side combination.
valid=[]
for fid,g in mm.groupby('fight_id'):
    ok=True
    for k in METHOD_KEYS:
        kg=g[g.market_key==k]
        if len(kg)!=2 or set(kg.outcome_side.astype(str))!={'red','blue'}:
            ok=False; break
    if ok: valid.append(fid)
mm=mm[mm.fight_id.isin(valid)].copy()
wide=mm.pivot_table(index='fight_id',columns=['outcome_side','market_key'],values='implied_probability',aggfunc='first')
rows=[]
for fid,r in wide.iterrows():
    try:
        raw={}
        for side in ['red','blue']:
            for k in METHOD_KEYS: raw[(side,k)]=float(r[(side,k)])
        total=sum(raw.values())
        if not np.isfinite(total) or total<=0: continue
        q={k:v/total for k,v in raw.items()}
        red_win=sum(q[('red',k)] for k in METHOD_KEYS); blue_win=sum(q[('blue',k)] for k in METHOD_KEYS)
        inside=sum(q[(side,k)] for side in ['red','blue'] for k in ['win_by_ko_tko_dq','win_by_submission'])
        red_sh=[q[('red',k)]/red_win for k in METHOD_KEYS] if red_win>0 else [1/3]*3
        blue_sh=[q[('blue',k)]/blue_win for k in METHOD_KEYS] if blue_win>0 else [1/3]*3
        conc=sum(x*x for x in red_sh)-sum(x*x for x in blue_sh)
        rows.append({'fight_id':fid,'method_implied_red_win_p':red_win,'method_inside_distance_p':inside,'method_win_concentration_diff':conc,'method_sixway_overround':total})
    except KeyError:
        pass
mf=pd.DataFrame(rows)
red=red.merge(mf,on='fight_id',how='left')
red['method_vs_ml_red_gap']=red['method_implied_red_win_p']-red['fair_market_p']

fv=pd.read_parquet('data/features/moneyline_feature_view.parquet')
basecols=[c for c in FEATURES if c!='market_overround']
df=red.merge(fv[['fight_id']+basecols],on='fight_id',how='inner').sort_values(['date','fight_id']).copy()
Xraw=df[FEATURES+METHOD_FEATURES].replace([np.inf,-np.inf],np.nan)

summary={'experiment':'frozen_v5_plus_sixway_method_market_v1','selection_objective':'2021-2024 chronological OOF log loss only; ROI not used','method_normalization':'normalize six legacy_consensus implied probabilities jointly across red/blue KO/TKO, submission, decision','method_features':METHOD_FEATURES,'coverage':{'all_rows':int(len(df)),'rows_with_complete_method':int(df[METHOD_FEATURES].notna().all(axis=1).sum())},'models':{}}
stores={}
for name,cols in {'v5':FEATURES,'v5_plus_method':FEATURES+METHOD_FEATURES}.items():
    parts=[]; folds=[]
    for fn,te,vs,ve in FOLDS:
        tr=df.date<=te; va=(df.date>=vs)&(df.date<=ve)
        validcols=[c for c in cols if Xraw.loc[tr,c].notna().any()]
        med=Xraw.loc[tr,validcols].median(numeric_only=True)
        Xtr=Xraw.loc[tr,validcols].fillna(med).fillna(0); Xva=Xraw.loc[va,validcols].fillna(med).fillna(0)
        ytr=df.loc[tr,'won'].astype(int).to_numpy(); yva=df.loc[va,'won'].astype(int).to_numpy()
        mtr=logit(df.loc[tr,'fair_market_p']); mva=logit(df.loc[va,'fair_market_p'])
        dtr=xgb.DMatrix(Xtr,label=ytr,base_margin=mtr,feature_names=validcols); dva=xgb.DMatrix(Xva,label=yva,base_margin=mva,feature_names=validcols)
        model=xgb.train(PARAMS,dtr,num_boost_round=300,verbose_eval=False); p=sigmoid(model.predict(dva,output_margin=True))
        mmtr=met(yva,sigmoid(mva)); mx=met(yva,p)
        folds.append({'fold':fn,'train_n':int(tr.sum()),'validation_n':int(va.sum()),'validation_method_complete_n':int(df.loc[va,METHOD_FEATURES].notna().all(axis=1).sum()),'market':mmtr,'model':mx,'delta_log_loss_vs_market':float(mx['log_loss']-mmtr['log_loss'])})
        parts.append(pd.DataFrame({'fight_id':df.loc[va,'fight_id'].to_numpy(),'date':df.loc[va,'date'].to_numpy(),'won':yva,'market_p':sigmoid(mva),'model_p':p,'method_complete':df.loc[va,METHOD_FEATURES].notna().all(axis=1).to_numpy()}))
    odf=pd.concat(parts,ignore_index=True); summary['models'][name]={'feature_count':len(cols),'folds':folds,'oof':met(odf.won,odf.model_p)}; stores[name]=odf
v=summary['models']['v5']['oof']['log_loss']; w=summary['models']['v5_plus_method']['oof']['log_loss']
summary['comparison']={'v5_plus_method_minus_v5_log_loss':float(w-v),'winner_by_oof_log_loss':'v5_plus_method' if w<v else 'v5'}
# Complete-method subset metric using exact same validation rows.
for name,odf in stores.items():
    z=odf[odf.method_complete].copy(); summary['models'][name]['oof_complete_method_subset']=met(z.won,z.model_p)
stores['v5_plus_method'].to_csv(OUT/'v5_plus_method_market_oof.csv',index=False)
json.dump(summary,open(OUT/'v5_plus_method_market_summary.json','w'),indent=2)
print(json.dumps(summary,indent=2))
