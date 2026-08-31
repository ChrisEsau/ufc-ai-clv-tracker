from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, brier_score_loss

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/research/prop_mispricing'
OOF=OUT/'xgboost_v5_exact_reproduction_selected_oof.csv'
TEST=OUT/'xgboost_v5_exact_reproduction_test_predictions.csv'

EPS=1e-12
BINS=[0.0,0.3,0.4,0.5,0.6,0.7,0.8,1.0000001]
LABELS=['<30%','30-40%','40-50%','50-60%','60-70%','70-80%','80%+']

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return np.log(p/(1-p))

def sigmoid(z):
    return 1/(1+np.exp(-np.asarray(z,float)))

def load_oof():
    d=pd.read_csv(OOF)
    d['date']=pd.to_datetime(d['date'])
    d['year']=d['date'].dt.year
    d=d.rename(columns={'market_p':'market_p','model_p':'v5_p','won':'y'})
    d['market_logit']=logit(d.market_p)
    d['v5_logit']=logit(d.v5_p)
    d['resid']=d.v5_logit-d.market_logit
    d['bucket']=pd.cut(d.market_p,BINS,right=False,labels=LABELS)
    return d

def load_test():
    d=pd.read_csv(TEST)
    d=d[(d.market_key=='moneyline')&(d.bookmaker=='legacy_consensus')&(d.result_status=='graded')].copy()
    # one RED row per fight; this preserves the same orientation as OOF
    d=d[d.canonical_side=='red'].copy()
    d['date']=pd.to_datetime(d['date'])
    d=d[(d.date>='2025-01-01')&(d.date<='2026-03-28')].copy()
    d['market_p']=d['fair_market_p'].astype(float)
    d['v5_p']=d['model_p'].astype(float)
    d['y']=d['won'].astype(int)
    d['market_logit']=logit(d.market_p)
    d['v5_logit']=logit(d.v5_p)
    d['resid']=d.v5_logit-d.market_logit
    d['bucket']=pd.cut(d.market_p,BINS,right=False,labels=LABELS)
    return d

def fit_k(x):
    if len(x)<25 or np.max(np.abs(x.resid.values))<1e-10:
        return 1.0
    y=x.y.values
    ml=x.market_logit.values
    r=x.resid.values
    def f(k): return log_loss(y,sigmoid(ml+k*r),labels=[0,1])
    return float(minimize_scalar(f,bounds=(0.0,2.0),method='bounded',options={'xatol':1e-10}).x)

def fit_map(train):
    return {b:fit_k(train[train.bucket==b]) for b in LABELS}

def apply_map(d,kmap):
    ks=d.bucket.astype(str).map(kmap).fillna(1.0).astype(float).values
    return sigmoid(d.market_logit.values+ks*d.resid.values),ks

def metrics(d,p):
    return {'n':int(len(d)),'log_loss':float(log_loss(d.y,p,labels=[0,1])),'brier':float(brier_score_loss(d.y,p))}

oof=load_oof(); test=load_test()
rows=[]; pred_parts=[]
for year in [2022,2023,2024]:
    tr=oof[oof.year<year].copy(); te=oof[oof.year==year].copy()
    kmap=fit_map(tr)
    cp,ks=apply_map(te,kmap)
    pred_parts.append(pd.DataFrame({'fight_id':te.fight_id,'date':te.date,'year':year,'market_p':te.market_p,'v5_p':te.v5_p,'cal_p':cp,'bucket':te.bucket.astype(str),'k_applied':ks,'y':te.y}))
    row={'test_year':year,'train_n':int(len(tr)),'test_n':int(len(te)),'k_by_bucket':kmap,'market':metrics(te,te.market_p.values),'v5':metrics(te,te.v5_p.values),'conditional_cal':metrics(te,cp)}
    rows.append(row)

chrono=pd.concat(pred_parts,ignore_index=True)
final_map=fit_map(oof)
test_cp,test_ks=apply_map(test,final_map)
summary={
 'experiment':'frozen_v5_conditional_market_probability_residual_calibration_v1',
 'mapping':'calibrated_logit = market_logit + k(market_probability_bucket) * V5_logit_residual',
 'selection_objective':'binary log loss only; no ROI used',
 'market_probability_buckets':LABELS,
 'k_search_bounds':[0.0,2.0],
 'chronological_protocol':'fit bucket-specific k values on prior OOF years; test next year: 2021->2022, 2021-22->2023, 2021-23->2024',
 'folds':rows,
 'chronological_2022_2024_metrics':{
   'market':metrics(chrono,chrono.market_p.values),
   'v5':metrics(chrono,chrono.v5_p.values),
   'conditional_cal':metrics(chrono,chrono.cal_p.values),
 },
 'final_k_by_bucket_fit_on_all_2021_2024_oof':final_map,
 'evaluation_period':'2025-01-01 through 2026-03-28; previously examined in prior diagnostics',
 'evaluation_metrics':{
   'market':metrics(test,test.market_p.values),
   'v5':metrics(test,test.v5_p.values),
   'conditional_cal':metrics(test,test_cp),
 }
}
OUT.mkdir(parents=True,exist_ok=True)
with open(OUT/'v5_conditional_residual_calibration_summary.json','w') as f: json.dump(summary,f,indent=2)
chrono.to_csv(OUT/'v5_conditional_residual_calibration_chrono_oof.csv',index=False)
t=pd.DataFrame({'fight_id':test.fight_id,'date':test.date,'market_p':test.market_p,'v5_p':test.v5_p,'cal_p':test_cp,'bucket':test.bucket.astype(str),'k_applied':test_ks,'y':test.y})
t.to_csv(OUT/'v5_conditional_residual_calibration_2025_to_20260328.csv',index=False)
print(json.dumps(summary,indent=2))
