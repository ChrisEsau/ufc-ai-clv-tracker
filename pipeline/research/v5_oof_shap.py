import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

OUT=Path('data/research/prop_mispricing')
OUT.mkdir(parents=True,exist_ok=True)
FEATURES=['reach_diff','recent_form_recent_avg_fight_time_diff','age_diff','ewm_sapm_diff','ewm_recent_sapm_diff','style_ko_finisher_score_diff','ewm_td_acc_diff','recent_finish_rate_diff','chin_risk_diff','recent_form_avg_opponent_elo_diff','recent_avg_fight_time_diff','aggression_index_diff','age_squared_diff','sapm_diff','ewm_kd_avg_diff','style_all_round_finisher_score_diff','recent_form_kd_absorbed_avg_diff','ewm_recent_splm_diff','elo_diff','ewm_elo_diff','ewm_recent_td_avg_diff','days_since_last_fight_diff','td_avg_diff','style_score_spread_diff','ko_dependency_diff','recent_form_avg_fight_time_diff','wrestling_mismatch_diff','win_pct_diff','recent_form_ko_rate_diff','recent_form_worst_loss_elo_diff','age_x_career_ko_losses_diff','ewm_str_def_diff','losses_diff','ewm_recent_win_pct_diff','avg_opponent_elo_diff','ewm_td_avg_diff','avg_fight_time_diff','ewm_days_since_last_fight_diff','pressure_striking_adv_diff','weight_diff','ctrl_against_per_min_diff','ewm_finish_loss_rate_diff','ewm_win_pct_diff','victory_concentration_index_diff','recent_form_td_acc_diff','sub_avg_diff','recent_form_best_win_elo_diff','ewm_best_win_elo_diff','style_primary_score_diff','recent_form_recent_finish_rate_diff','market_overround']
params={'max_depth':1,'eta':0.03,'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':10,'lambda':8.0,'alpha':1.0,'objective':'binary:logistic','eval_metric':'logloss','seed':42,'nthread':2}
rounds=300

def clip_p(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def logit(p):
    p=clip_p(p); return np.log(p/(1-p))

market=pd.read_parquet('data/market/historical_market_outcomes.parquet').copy()
market=market[(market.bookmaker=='legacy_consensus')&(market.result_status=='graded')&market.won.notna()]
market['date']=pd.to_datetime(market['date'])
market['won']=market['won'].astype(bool).astype(int)
market['implied_probability']=pd.to_numeric(market['implied_probability'],errors='coerce')
ml=market[market.market_key=='moneyline'].copy()
good=ml.groupby('fight_id').size(); good=good[good==2].index; ml=ml[ml.fight_id.isin(good)]
ml['market_overround']=ml.groupby('fight_id').implied_probability.transform('sum')
ml['fair_market_p']=ml.implied_probability/ml.market_overround
red=ml[ml.outcome_side.astype(str).eq('red')].copy()
fv=pd.read_parquet('data/features/moneyline_feature_view.parquet')
df=red.merge(fv[['fight_id']+[f for f in FEATURES if f!='market_overround']],on='fight_id',how='inner').sort_values(['date','fight_id']).copy()
Xraw=df[FEATURES].replace([np.inf,-np.inf],np.nan)
folds=[('2021','2020-12-31','2021-01-01','2021-12-31'),('2022','2021-12-31','2022-01-01','2022-12-31'),('2023','2022-12-31','2023-01-01','2023-12-31'),('2024','2023-12-31','2024-01-01','2024-12-31')]
parts=[]
for fold,train_end,val_start,val_end in folds:
    tr=df.date<=train_end; va=(df.date>=val_start)&(df.date<=val_end)
    valid=[c for c in FEATURES if Xraw.loc[tr,c].notna().any()]
    med=Xraw.loc[tr,valid].median(numeric_only=True)
    Xtr=Xraw.loc[tr,valid].fillna(med).fillna(0.0); Xva=Xraw.loc[va,valid].fillna(med).fillna(0.0)
    ytr=df.loc[tr,'won'].astype(int).to_numpy(); mtr=logit(df.loc[tr,'fair_market_p'])
    mva=logit(df.loc[va,'fair_market_p'])
    dtr=xgb.DMatrix(Xtr,label=ytr,base_margin=mtr,feature_names=valid)
    dva=xgb.DMatrix(Xva,base_margin=mva,feature_names=valid)
    model=xgb.train(params,dtr,num_boost_round=rounds,verbose_eval=False)
    contrib=model.predict(dva,pred_contribs=True)
    # contrib columns are feature SHAP on margin scale plus bias. Base margin is separate market prior.
    shap=contrib[:,:-1]
    tmp=pd.DataFrame({'feature':valid,'mean_abs_shap':np.mean(np.abs(shap),axis=0),'mean_shap':np.mean(shap,axis=0),'fold':fold,'n':int(va.sum())})
    parts.append(tmp)
allf=pd.concat(parts,ignore_index=True)
# weight by fold n
agg=(allf.assign(wabs=allf.mean_abs_shap*allf.n,wmean=allf.mean_shap*allf.n)
     .groupby('feature',as_index=False).agg(n=('n','sum'),wabs=('wabs','sum'),wmean=('wmean','sum')))
agg['mean_abs_shap']=agg.wabs/agg.n; agg['mean_shap']=agg.wmean/agg.n
agg=agg.sort_values('mean_abs_shap',ascending=False).reset_index(drop=True)
agg['rank']=np.arange(1,len(agg)+1)
agg=agg[['rank','feature','mean_abs_shap','mean_shap','n']]
agg.to_csv(OUT/'v5_oof_shap_importance.csv',index=False)
allf.to_csv(OUT/'v5_oof_shap_by_fold.csv',index=False)
summary={'experiment':'frozen_v5_chronological_oof_shap','method':'XGBoost pred_contribs / TreeSHAP on each held-out chronological fold; market base_margin excluded from feature SHAP ranking','oof_n':int(sum(int(((df.date>=s)&(df.date<=e)).sum()) for _,_,s,e in folds)),'top_20':agg.head(20).to_dict('records'),'elo_features':agg[agg.feature.isin(['elo_diff','ewm_elo_diff','recent_form_avg_opponent_elo_diff','avg_opponent_elo_diff','recent_form_worst_loss_elo_diff','recent_form_best_win_elo_diff','ewm_best_win_elo_diff'])].to_dict('records')}
with open(OUT/'v5_oof_shap_summary.json','w') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
