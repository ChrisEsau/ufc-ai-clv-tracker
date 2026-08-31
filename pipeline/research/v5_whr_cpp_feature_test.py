from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import whr

OUT=Path('data/research/prop_mispricing'); OUT.mkdir(parents=True,exist_ok=True)
FEATURES=['reach_diff','recent_form_recent_avg_fight_time_diff','age_diff','ewm_sapm_diff','ewm_recent_sapm_diff','style_ko_finisher_score_diff','ewm_td_acc_diff','recent_finish_rate_diff','chin_risk_diff','recent_form_avg_opponent_elo_diff','recent_avg_fight_time_diff','aggression_index_diff','age_squared_diff','sapm_diff','ewm_kd_avg_diff','style_all_round_finisher_score_diff','recent_form_kd_absorbed_avg_diff','ewm_recent_splm_diff','elo_diff','ewm_elo_diff','ewm_recent_td_avg_diff','days_since_last_fight_diff','td_avg_diff','style_score_spread_diff','ko_dependency_diff','recent_form_avg_fight_time_diff','wrestling_mismatch_diff','win_pct_diff','recent_form_ko_rate_diff','recent_form_worst_loss_elo_diff','age_x_career_ko_losses_diff','ewm_str_def_diff','losses_diff','ewm_recent_win_pct_diff','avg_opponent_elo_diff','ewm_td_avg_diff','avg_fight_time_diff','ewm_days_since_last_fight_diff','pressure_striking_adv_diff','weight_diff','ctrl_against_per_min_diff','ewm_finish_loss_rate_diff','ewm_win_pct_diff','victory_concentration_index_diff','recent_form_td_acc_diff','sub_avg_diff','recent_form_best_win_elo_diff','ewm_best_win_elo_diff','style_primary_score_diff','recent_form_recent_finish_rate_diff','market_overround']
PARAMS={'max_depth':1,'eta':0.03,'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':10,'lambda':8.0,'alpha':1.0,'objective':'binary:logistic','eval_metric':'logloss','seed':42,'nthread':2}
FOLDS=[('2021','2020-12-31','2021-01-01','2021-12-31'),('2022','2021-12-31','2022-01-01','2022-12-31'),('2023','2022-12-31','2023-01-01','2023-12-31'),('2024','2023-12-31','2024-01-01','2024-12-31')]
def clip(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def logit(p): p=clip(p); return np.log(p/(1-p))
def sigmoid(z): z=np.clip(np.asarray(z,float),-30,30); return 1/(1+np.exp(-z))
def met(y,p):
 y=np.asarray(y,int); p=clip(p); return {'n':int(len(y)),'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}
market=pd.read_parquet('data/market/historical_market_outcomes.parquet').copy(); market=market[(market.bookmaker=='legacy_consensus')&(market.result_status=='graded')&market.won.notna()].copy(); market['date']=pd.to_datetime(market.date,errors='coerce'); market['won']=market.won.astype(bool).astype(int); market['implied_probability']=pd.to_numeric(market.implied_probability,errors='coerce'); market=market.dropna(subset=['date','implied_probability','outcome_label','fight_id']); ml=market[market.market_key=='moneyline'].copy(); good=ml.groupby('fight_id').size(); good=good[good==2].index; ml=ml[ml.fight_id.isin(good)].copy(); ml['market_overround']=ml.groupby('fight_id').implied_probability.transform('sum'); ml['fair_market_p']=ml.implied_probability/ml.market_overround
red=ml[ml.outcome_side.astype(str).eq('red')].copy(); blue=ml[ml.outcome_side.astype(str).eq('blue')][['fight_id','outcome_label']].rename(columns={'outcome_label':'blue_name'}); red=red.merge(blue,on='fight_id',validate='one_to_one').rename(columns={'outcome_label':'red_name'}); red['red_name']=red.red_name.astype(str); red['blue_name']=red.blue_name.astype(str)
base=whr.Base(); pmap={}; first=red.date.min().normalize(); known=set()
def last_elo(name):
 if name not in known: return 0.0
 try:
  a=base.ratings_for_player(name)
  return float(a[-1][1]) if len(a) else 0.0
 except Exception: return 0.0
for date,day in red.sort_values(['date','fight_id']).groupby('date',sort=True):
 for r in day.itertuples(index=False):
  a=last_elo(r.red_name); b=last_elo(r.blue_name); pmap[r.fight_id]=1.0/(1.0+10.0**((b-a)/400.0))
 step=int((pd.Timestamp(date).normalize()-first).days)
 for r in day.itertuples(index=False):
  base.create_game(r.red_name,r.blue_name,'B' if int(r.won)==1 else 'W',step,0.0); known.add(r.red_name); known.add(r.blue_name)
 base.iterate(50)
red['whr_p_red']=red.fight_id.map(pmap).fillna(.5); red['whr_logit_diff']=logit(red.whr_p_red)
fv=pd.read_parquet('data/features/moneyline_feature_view.parquet'); df=red.merge(fv[['fight_id']+[c for c in FEATURES if c!='market_overround']],on='fight_id',how='inner').sort_values(['date','fight_id']).copy(); Xraw=df[FEATURES+['whr_logit_diff']].replace([np.inf,-np.inf],np.nan)
summary={'experiment':'frozen_v5_plus_leakage_safe_compiled_whr_v1','selection_objective':'2021-2024 chronological OOF log loss only; ROI not used','whr_protocol':'same-date blocked expanding history; 50 Newton iterations after each event date; default compiled WHR config; zero handicap','new_feature':'whr_logit_diff','models':{}}
stores={}
for name,cols in {'v5':FEATURES,'v5_plus_whr':FEATURES+['whr_logit_diff']}.items():
 parts=[]; folds=[]
 for fn,te,vs,ve in FOLDS:
  tr=df.date<=te; va=(df.date>=vs)&(df.date<=ve); valid=[c for c in cols if Xraw.loc[tr,c].notna().any()]; med=Xraw.loc[tr,valid].median(numeric_only=True); Xtr=Xraw.loc[tr,valid].fillna(med).fillna(0); Xva=Xraw.loc[va,valid].fillna(med).fillna(0); ytr=df.loc[tr,'won'].astype(int).to_numpy(); yva=df.loc[va,'won'].astype(int).to_numpy(); mtr=logit(df.loc[tr,'fair_market_p']); mva=logit(df.loc[va,'fair_market_p']); dtr=xgb.DMatrix(Xtr,label=ytr,base_margin=mtr,feature_names=valid); dva=xgb.DMatrix(Xva,label=yva,base_margin=mva,feature_names=valid); model=xgb.train(PARAMS,dtr,num_boost_round=300,verbose_eval=False); p=sigmoid(model.predict(dva,output_margin=True)); mm=met(yva,sigmoid(mva)); mx=met(yva,p); folds.append({'fold':fn,'train_n':int(tr.sum()),'validation_n':int(va.sum()),'market':mm,'model':mx,'delta_log_loss_vs_market':float(mx['log_loss']-mm['log_loss'])}); parts.append(pd.DataFrame({'fight_id':df.loc[va,'fight_id'].to_numpy(),'date':df.loc[va,'date'].to_numpy(),'won':yva,'market_p':sigmoid(mva),'model_p':p,'whr_p_red':df.loc[va,'whr_p_red'].to_numpy()}))
 odf=pd.concat(parts,ignore_index=True); summary['models'][name]={'feature_count':len(cols),'folds':folds,'oof':met(odf.won,odf.model_p)}; stores[name]=odf
v=summary['models']['v5']['oof']['log_loss']; w=summary['models']['v5_plus_whr']['oof']['log_loss']; summary['comparison']={'v5_plus_whr_minus_v5_log_loss':float(w-v),'winner_by_oof_log_loss':'v5_plus_whr' if w<v else 'v5'}; summary['standalone_whr_oof']=met(stores['v5_plus_whr'].won,stores['v5_plus_whr'].whr_p_red)
stores['v5_plus_whr'].to_csv(OUT/'v5_plus_whr_cpp_oof.csv',index=False); json.dump(summary,open(OUT/'v5_plus_whr_cpp_summary.json','w'),indent=2); print(json.dumps(summary,indent=2))
