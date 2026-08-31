from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

OUT = Path('data/research/prop_mispricing')
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
'reach_diff','recent_form_recent_avg_fight_time_diff','age_diff','ewm_sapm_diff','ewm_recent_sapm_diff',
'style_ko_finisher_score_diff','ewm_td_acc_diff','recent_finish_rate_diff','chin_risk_diff','recent_form_avg_opponent_elo_diff',
'recent_avg_fight_time_diff','aggression_index_diff','age_squared_diff','sapm_diff','ewm_kd_avg_diff',
'style_all_round_finisher_score_diff','recent_form_kd_absorbed_avg_diff','ewm_recent_splm_diff','elo_diff','ewm_elo_diff',
'ewm_recent_td_avg_diff','days_since_last_fight_diff','td_avg_diff','style_score_spread_diff','ko_dependency_diff',
'recent_form_avg_fight_time_diff','wrestling_mismatch_diff','win_pct_diff','recent_form_ko_rate_diff','recent_form_worst_loss_elo_diff',
'age_x_career_ko_losses_diff','ewm_str_def_diff','losses_diff','ewm_recent_win_pct_diff','avg_opponent_elo_diff',
'ewm_td_avg_diff','avg_fight_time_diff','ewm_days_since_last_fight_diff','pressure_striking_adv_diff','weight_diff',
'ctrl_against_per_min_diff','ewm_finish_loss_rate_diff','ewm_win_pct_diff','victory_concentration_index_diff','recent_form_td_acc_diff',
'sub_avg_diff','recent_form_best_win_elo_diff','ewm_best_win_elo_diff','style_primary_score_diff','recent_form_recent_finish_rate_diff'
]
FEATURE_COLS = FEATURES + ['market_overround']

market = pd.read_parquet('data/market/historical_market_outcomes.parquet').copy()
market = market[(market['bookmaker']=='legacy_consensus') & (market['result_status']=='graded') & market['won'].notna()].copy()
market['date'] = pd.to_datetime(market['date'], errors='coerce')
market['won'] = market['won'].astype(bool).astype(int)
market['implied_probability'] = pd.to_numeric(market['implied_probability'], errors='coerce')
market = market.dropna(subset=['date','implied_probability']).copy()
ml = market[market['market_key']=='moneyline'].copy()
good = ml.groupby('fight_id').size(); good = good[good==2].index
ml = ml[ml['fight_id'].isin(good)].copy()
ml['market_overround'] = ml.groupby('fight_id')['implied_probability'].transform('sum')
ml['fair_market_p'] = ml['implied_probability']/ml['market_overround']
red = ml[ml['outcome_side'].astype(str).eq('red')].copy()
fv = pd.read_parquet('data/features/moneyline_feature_view.parquet').copy()
missing = [c for c in FEATURES if c not in fv.columns]
if missing: raise RuntimeError(f'Missing frozen V5 features: {missing}')
df = red.merge(fv[['fight_id']+FEATURES], on='fight_id', how='inner').sort_values(['date','fight_id']).copy()
Xraw = df[FEATURE_COLS].replace([np.inf,-np.inf],np.nan)

def clip_p(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def logit(p):
    p=clip_p(p); return np.log(p/(1-p))
def sigmoid(z):
    z=np.clip(np.asarray(z,float),-30,30); return 1/(1+np.exp(-z))
def metrics(y,p):
    y=np.asarray(y,int); p=clip_p(p)
    return {'n':int(len(y)),'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),
            'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}

base_params = dict(eta=0.03,subsample=0.8,colsample_bytree=0.7,min_child_weight=10,**{'lambda':8.0,'alpha':1.0},
                   objective='binary:logistic',eval_metric='logloss',seed=42,nthread=2)
folds=[('2021','2020-12-31','2021-01-01','2021-12-31'),('2022','2021-12-31','2022-01-01','2022-12-31'),
       ('2023','2022-12-31','2023-01-01','2023-12-31'),('2024','2023-12-31','2024-01-01','2024-12-31')]

summary={'experiment':'frozen_v5_depth1_vs_depth2_only','selection_objective':'2021-2024 chronological OOF log loss only; ROI not used',
         'frozen_feature_count':len(FEATURE_COLS),'features':FEATURE_COLS,'models':{}}
pred_parts={}
for depth in [1,2]:
    params=dict(base_params,max_depth=depth)
    parts=[]; fold_out=[]
    for fold,train_end,val_start,val_end in folds:
        tr=df['date']<=train_end; va=(df['date']>=val_start)&(df['date']<=val_end)
        med=Xraw.loc[tr,FEATURE_COLS].median(numeric_only=True)
        Xtr=Xraw.loc[tr,FEATURE_COLS].fillna(med).fillna(0.0); Xva=Xraw.loc[va,FEATURE_COLS].fillna(med).fillna(0.0)
        ytr=df.loc[tr,'won'].to_numpy(dtype=int); yva=df.loc[va,'won'].to_numpy(dtype=int)
        mtr=logit(df.loc[tr,'fair_market_p']); mva=logit(df.loc[va,'fair_market_p'])
        dtr=xgb.DMatrix(Xtr,label=ytr,base_margin=mtr,feature_names=FEATURE_COLS)
        dva=xgb.DMatrix(Xva,label=yva,base_margin=mva,feature_names=FEATURE_COLS)
        model=xgb.train(params,dtr,num_boost_round=300,verbose_eval=False)
        full_margin=model.predict(dva,output_margin=True)
        p=sigmoid(full_margin)
        mm=metrics(yva,sigmoid(mva)); mx=metrics(yva,p)
        fold_out.append({'fold':fold,'train_n':int(tr.sum()),'validation_n':int(va.sum()),'market':mm,'model':mx,
                         'delta_log_loss_vs_market':mx['log_loss']-mm['log_loss'],'delta_brier_vs_market':mx['brier']-mm['brier']})
        parts.append(pd.DataFrame({'fight_id':df.loc[va,'fight_id'].to_numpy(),'date':df.loc[va,'date'].to_numpy(),
                                   'fold':fold,'won':yva,'market_p':sigmoid(mva),f'depth{depth}_p':p}))
    odf=pd.concat(parts,ignore_index=True)
    om=metrics(odf['won'],odf[f'depth{depth}_p'])
    marketm=metrics(odf['won'],odf['market_p'])
    summary['models'][f'depth_{depth}']={'params':params,'folds':fold_out,'oof':om,
                                         'delta_log_loss_vs_market':om['log_loss']-marketm['log_loss'],
                                         'delta_brier_vs_market':om['brier']-marketm['brier']}
    pred_parts[depth]=odf

merged=pred_parts[1].merge(pred_parts[2][['fight_id','depth2_p']],on='fight_id',how='inner',validate='one_to_one')
ll1=summary['models']['depth_1']['oof']['log_loss']; ll2=summary['models']['depth_2']['oof']['log_loss']
summary['comparison']={'depth2_minus_depth1_log_loss':ll2-ll1,
                       'depth2_minus_depth1_brier':summary['models']['depth_2']['oof']['brier']-summary['models']['depth_1']['oof']['brier'],
                       'winner_by_oof_log_loss':'depth_2' if ll2<ll1 else 'depth_1'}
with open(OUT/'v5_depth1_vs_depth2_summary.json','w') as f: json.dump(summary,f,indent=2)
merged.to_csv(OUT/'v5_depth1_vs_depth2_oof.csv',index=False)
print(json.dumps(summary,indent=2))
