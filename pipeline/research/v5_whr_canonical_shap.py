from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import whr

OUT=Path('data/research/prop_mispricing'); OUT.mkdir(parents=True,exist_ok=True)
FEATURES=['reach_diff','recent_form_recent_avg_fight_time_diff','age_diff','ewm_sapm_diff','ewm_recent_sapm_diff','style_ko_finisher_score_diff','ewm_td_acc_diff','recent_finish_rate_diff','chin_risk_diff','recent_form_avg_opponent_elo_diff','recent_avg_fight_time_diff','aggression_index_diff','age_squared_diff','sapm_diff','ewm_kd_avg_diff','style_all_round_finisher_score_diff','recent_form_kd_absorbed_avg_diff','ewm_recent_splm_diff','elo_diff','ewm_elo_diff','ewm_recent_td_avg_diff','days_since_last_fight_diff','td_avg_diff','style_score_spread_diff','ko_dependency_diff','recent_form_avg_fight_time_diff','wrestling_mismatch_diff','win_pct_diff','recent_form_ko_rate_diff','recent_form_worst_loss_elo_diff','age_x_career_ko_losses_diff','ewm_str_def_diff','losses_diff','ewm_recent_win_pct_diff','avg_opponent_elo_diff','ewm_td_avg_diff','avg_fight_time_diff','ewm_days_since_last_fight_diff','pressure_striking_adv_diff','weight_diff','ctrl_against_per_min_diff','ewm_finish_loss_rate_diff','ewm_win_pct_diff','victory_concentration_index_diff','recent_form_td_acc_diff','sub_avg_diff','recent_form_best_win_elo_diff','ewm_best_win_elo_diff','style_primary_score_diff','recent_form_recent_finish_rate_diff','market_overround']
PARAMS={'max_depth':1,'eta':0.03,'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':10,'lambda':8.0,'alpha':1.0,'objective':'binary:logistic','eval_metric':'logloss','seed':42,'nthread':2}
FOLDS=[('2021','2020-12-31','2021-01-01','2021-12-31'),('2022','2021-12-31','2022-01-01','2022-12-31'),('2023','2022-12-31','2023-01-01','2023-12-31'),('2024','2023-12-31','2024-01-01','2024-12-31')]
def clip(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def logit(p): p=clip(p); return np.log(p/(1-p))
def sigmoid(z): z=np.clip(np.asarray(z,float),-30,30); return 1/(1+np.exp(-z))
market=pd.read_parquet('data/market/historical_market_outcomes.parquet').copy(); market=market[(market.bookmaker=='legacy_consensus')&(market.result_status=='graded')&market.won.notna()].copy(); market['date']=pd.to_datetime(market.date,errors='coerce'); market['won']=market.won.astype(bool).astype(int); market['implied_probability']=pd.to_numeric(market.implied_probability,errors='coerce'); market=market.dropna(subset=['date','implied_probability','outcome_label','fight_id']); ml=market[market.market_key=='moneyline'].copy(); good=ml.groupby('fight_id').size(); good=good[good==2].index; ml=ml[ml.fight_id.isin(good)].copy(); ml['market_overround']=ml.groupby('fight_id').implied_probability.transform('sum'); ml['fair_market_p']=ml.implied_probability/ml.market_overround
red=ml[ml.outcome_side.astype(str).eq('red')].copy(); blue=ml[ml.outcome_side.astype(str).eq('blue')][['fight_id','outcome_label']].rename(columns={'outcome_label':'blue_name'}); red=red.merge(blue,on='fight_id',validate='one_to_one').rename(columns={'outcome_label':'red_name'}); red['red_name']=red.red_name.astype(str); red['blue_name']=red.blue_name.astype(str)
base=whr.Base(config={'w2':7.5625}); pmap={}; first=red.date.min().normalize(); known=set()
def last_rating(name):
    if name not in known: return 0.0
    try:
        a=base.ratings_for_player(name)
        return float(a[-1][1]) if len(a) else 0.0
    except Exception: return 0.0
for date,day in red.sort_values(['date','fight_id']).groupby('date',sort=True):
    for r in day.itertuples(index=False):
        a=last_rating(r.red_name); b=last_rating(r.blue_name); pmap[r.fight_id]=1.0/(1.0+10.0**((b-a)/400.0))
    step=int((pd.Timestamp(date).normalize()-first).days)
    for r in day.itertuples(index=False):
        base.create_game(r.red_name,r.blue_name,'B' if int(r.won)==1 else 'W',step,0.0); known.add(r.red_name); known.add(r.blue_name)
    base.iterate(50)
red['whr_p_red']=red.fight_id.map(pmap).fillna(.5); red['whr_logit_diff']=logit(red.whr_p_red)
fv=pd.read_parquet('data/features/moneyline_feature_view.parquet'); df=red.merge(fv[['fight_id']+[c for c in FEATURES if c!='market_overround']],on='fight_id',how='inner').sort_values(['date','fight_id']).copy(); cols=FEATURES+['whr_logit_diff']; Xraw=df[cols].replace([np.inf,-np.inf],np.nan)
rows=[]
for fn,te,vs,ve in FOLDS:
    tr=df.date<=te; va=(df.date>=vs)&(df.date<=ve); valid=[c for c in cols if Xraw.loc[tr,c].notna().any()]; med=Xraw.loc[tr,valid].median(numeric_only=True); Xtr=Xraw.loc[tr,valid].fillna(med).fillna(0); Xva=Xraw.loc[va,valid].fillna(med).fillna(0); ytr=df.loc[tr,'won'].astype(int).to_numpy(); mtr=logit(df.loc[tr,'fair_market_p']); mva=logit(df.loc[va,'fair_market_p']); dtr=xgb.DMatrix(Xtr,label=ytr,base_margin=mtr,feature_names=valid); dva=xgb.DMatrix(Xva,base_margin=mva,feature_names=valid); model=xgb.train(PARAMS,dtr,num_boost_round=300,verbose_eval=False); contrib=model.predict(dva,pred_contribs=True); # last column is bias
    for j,f in enumerate(valid):
        vals=contrib[:,j]
        for v in vals: rows.append((fn,f,float(v)))
sh=pd.DataFrame(rows,columns=['fold','feature','shap'])
agg=sh.groupby('feature').agg(mean_abs_shap=('shap',lambda s: float(np.mean(np.abs(s)))),mean_shap=('shap','mean'),n=('shap','size')).reset_index().sort_values('mean_abs_shap',ascending=False).reset_index(drop=True); agg['rank']=np.arange(1,len(agg)+1)
fold=sh.groupby(['fold','feature']).agg(mean_abs_shap=('shap',lambda s: float(np.mean(np.abs(s)))),mean_shap=('shap','mean'),n=('shap','size')).reset_index(); agg.to_csv(OUT/'v5_plus_whr_canonical_shap_global.csv',index=False); fold.to_csv(OUT/'v5_plus_whr_canonical_shap_by_fold.csv',index=False)
w=agg.loc[agg.feature.eq('whr_logit_diff')].iloc[0].to_dict(); summary={'experiment':'v5_plus_canonical_whr_oof_treeshap_v1','rows':int(sh.shape[0]),'oof_fights':int(sum((df.date.between(vs,ve)).sum() for _,_,vs,ve in FOLDS)),'whr':{k:(int(v) if k in ('rank','n') else float(v) if isinstance(v,(np.floating,float)) else v) for k,v in w.items()},'top20':agg.head(20).to_dict(orient='records')}; json.dump(summary,open(OUT/'v5_plus_whr_canonical_shap_summary.json','w'),indent=2); print(json.dumps(summary,indent=2))
