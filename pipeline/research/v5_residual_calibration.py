import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar

OOF=Path('data/research/prop_mispricing/xgboost_v5_exact_reproduction_selected_oof.csv')
TEST=Path('data/research/prop_mispricing/xgboost_v5_exact_reproduction_test_predictions.csv')
OUT=Path('data/research/prop_mispricing')
EPS=1e-12

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return np.log(p/(1-p))
def sigmoid(x): return 1/(1+np.exp(-np.asarray(x,float)))
def ll(y,p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); y=np.asarray(y,float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def brier(y,p): return float(np.mean((np.asarray(p,float)-np.asarray(y,float))**2))
def fit_k(df):
    m=logit(df.market_p.values); r=logit(df.model_p.values)-m; y=df.won.values
    f=lambda k: ll(y,sigmoid(m+k*r))
    z=minimize_scalar(f,bounds=(0,2),method='bounded',options={'xatol':1e-12})
    return float(z.x)
def metrics(df,pcol):
    return {'n':len(df),'log_loss':ll(df.won,df[pcol]),'brier':brier(df.won,df[pcol])}

oof=pd.read_csv(OOF); oof['date']=pd.to_datetime(oof.date); oof=oof.sort_values(['date','fight_id']).copy()
oof['market_logit']=logit(oof.market_p); oof['v5_logit']=logit(oof.model_p); oof['logit_residual']=oof.v5_logit-oof.market_logit; oof['prob_residual']=oof.model_p-oof.market_p

# Chronological calibration OOF: fit scalar residual strength only on earlier OOF years.
parts=[]; fold_rows=[]
for y in [2022,2023,2024]:
    tr=oof[oof.date.dt.year<y].copy(); va=oof[oof.date.dt.year==y].copy()
    k=fit_k(tr)
    va['calibrated_p']=sigmoid(va.market_logit+k*va.logit_residual)
    parts.append(va)
    fold_rows.append({'test_year':y,'train_n':len(tr),'test_n':len(va),'k':k,
      'market_ll':ll(va.won,va.market_p),'v5_ll':ll(va.won,va.model_p),'cal_ll':ll(va.won,va.calibrated_p),
      'market_brier':brier(va.won,va.market_p),'v5_brier':brier(va.won,va.model_p),'cal_brier':brier(va.won,va.calibrated_p)})
chrono=pd.concat(parts,ignore_index=True)

# Final scalar fit on all 2021-24 OOF, then apply once to 2025-Mar28 2026.
k_final=fit_k(oof)
test=pd.read_csv(TEST); test['date']=pd.to_datetime(test.date)
# one RED row per fight for probability metrics
test_red=test[(test.market_key=='moneyline')&(test.bookmaker=='legacy_consensus')&(test.canonical_side=='red')].copy()
test_red=test_red[(test_red.date>=pd.Timestamp('2025-01-01'))&(test_red.date<=pd.Timestamp('2026-03-28'))].copy()
test_red['market_p']=pd.to_numeric(test_red.fair_market_p)
test_red['model_p']=pd.to_numeric(test_red.model_p)
test_red['won']=pd.to_numeric(test_red.won)
test_red['market_logit']=logit(test_red.market_p); test_red['logit_residual']=logit(test_red.model_p)-test_red.market_logit
test_red['calibrated_p']=sigmoid(test_red.market_logit+k_final*test_red.logit_residual)

# Pooled chronological calibration OOF metrics
chrono_metrics={
 'market':metrics(chrono,'market_p'),
 'v5':metrics(chrono,'model_p'),
 'calibrated':metrics(chrono,'calibrated_p')}
test_metrics={
 'market':metrics(test_red,'market_p'),
 'v5':metrics(test_red,'model_p'),
 'calibrated':metrics(test_red,'calibrated_p')}

# Residual-size diagnostics on full 2021-24 OOF (measurement only).
absr=oof.prob_residual.abs()
bins=[0,.01,.02,.03,.04,.06,.08,np.inf]; labels=['0-1%','1-2%','2-3%','3-4%','4-6%','6-8%','8%+']
oof['abs_prob_residual_bucket']=pd.cut(absr,bins=bins,labels=labels,right=False)
diag=[]
for lab in labels:
    q=oof[oof.abs_prob_residual_bucket==lab]
    if len(q)==0: continue
    # Signed market error and model movement toward outcome, averaged in orientation of residual sign.
    s=np.sign(q.prob_residual.values); y=q.won.values
    market_signed_outcome=(y-q.market_p.values)*s
    model_move=q.prob_residual.values*s
    realized_needed=market_signed_outcome
    diag.append({'bucket':lab,'n':len(q),'avg_abs_prob_residual':float(np.mean(np.abs(q.prob_residual))),
                 'avg_abs_logit_residual':float(np.mean(np.abs(q.logit_residual))),
                 'avg_realized_directional_gap_vs_market':float(np.mean(realized_needed)),
                 'avg_v5_directional_move':float(np.mean(model_move)),
                 'direction_accuracy':float(np.mean(((y-q.market_p.values)*s)>0))})

summary={
 'experiment':'frozen_v5_scalar_logit_residual_calibration_v1',
 'mapping':'calibrated_logit = market_logit + k * (v5_logit - market_logit)',
 'selection_objective':'binary log loss only; no ROI used',
 'k_search_bounds':[0.0,2.0],
 'chronological_calibration_protocol':'fit k on prior OOF years, test next year: 2021->2022, 2021-22->2023, 2021-23->2024',
 'folds':fold_rows,
 'chronological_2022_2024_metrics':chrono_metrics,
 'final_k_fit_on_all_2021_2024_oof':k_final,
 'evaluation_period':'2025-01-01 through 2026-03-28; previously examined in prior diagnostics',
 'evaluation_metrics':test_metrics,
 'residual_bucket_diagnostics':diag}

pd.DataFrame(fold_rows).to_csv(OUT/'v5_residual_calibration_folds.csv',index=False)
chrono[['fight_id','date','won','market_p','model_p','calibrated_p','logit_residual','prob_residual']].to_csv(OUT/'v5_residual_calibration_chrono_oof.csv',index=False)
test_red[['fight_id','date','won','market_p','model_p','calibrated_p','logit_residual']].to_csv(OUT/'v5_residual_calibration_2025_to_20260328.csv',index=False)
pd.DataFrame(diag).to_csv(OUT/'v5_residual_calibration_buckets.csv',index=False)
with open(OUT/'v5_residual_calibration_summary.json','w') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
