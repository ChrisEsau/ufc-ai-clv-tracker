from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('data/research/prop_mispricing')
SRC=ROOT/'xgboost_v5_exact_reproduction_test_predictions.csv'
THRESH=0.20

def logit(p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    return np.log(p/(1-p))

def summarize(g):
    n=len(g); profit=float(g.profit_units.sum()) if n else 0.0
    if n:
        c=g.profit_units.cumsum(); peak=np.maximum.accumulate(np.r_[0.0,c.to_numpy()]); dd=peak[1:]-c.to_numpy(); maxdd=float(dd.max())
    else: maxdd=0.0
    return {'bets':int(n),'wins':int(g.won.sum()) if n else 0,'losses':int(n-g.won.sum()) if n else 0,
            'profit_units':profit,'roi':profit/n if n else None,'max_drawdown_units':maxdd,
            'mean_abs_logit_residual':float(g.abs_logit_residual.mean()) if n else None,
            'mean_fair_market_p':float(g.fair_market_p.mean()) if n else None}

df=pd.read_csv(SRC)
df['date']=pd.to_datetime(df.date)
# Exact V5 test-prediction file is side-level. Pick only the side V5 prices above the market.
df['signed_logit_residual']=logit(df.model_p)-logit(df.fair_market_p)
df['abs_logit_residual']=np.abs(df.signed_logit_residual)
sel=df[(df.edge>0)&(df.abs_logit_residual>=THRESH)].copy()
sel=sel.sort_values(['date','fight_id']).reset_index(drop=True)
# Cold-start intentionally OFF: no fighter-history eligibility filter is applied.
sel['profit_units']=np.where(sel.won.astype(int).eq(1),sel.profit_per_100.astype(float)/100.0,-1.0)
sel['market_role']=np.where(sel.fair_market_p>=0.5,'favorite','underdog')
sel['year']=sel.date.dt.year
sel['cum_profit_units']=sel.profit_units.cumsum()
sel['running_peak_units']=np.maximum.accumulate(np.r_[0.0,sel.cum_profit_units.to_numpy()])[1:]
sel['drawdown_units']=sel.running_peak_units-sel.cum_profit_units

by_year=[]
for y,g in sel.groupby('year',sort=True):
    x=summarize(g); x['year']=int(y); by_year.append(x)
by_role=[]
for role,g in sel.groupby('market_role',sort=True):
    x=summarize(g); x['market_role']=role; by_role.append(x)
by_card=[]
for (date,event),g in sel.groupby(['date','event_name'],sort=True):
    x=summarize(g); x['date']=date.date().isoformat(); x['event_name']=event; by_card.append(x)

summary={
 'experiment':'frozen_v5_abs_logit_residual_ge_0.20_later_validation_v1',
 'rule':{'abs_logit_residual_min':THRESH,'stake_units':1.0,'cold_start_filter':False},
 'source':'xgboost_v5_exact_reproduction_test_predictions.csv',
 'selection_note':'Threshold frozen from 2021-2024 development diagnostic before this later-period read; no later-period optimization.',
 'date_min':sel.date.min().date().isoformat() if len(sel) else None,
 'date_max':sel.date.max().date().isoformat() if len(sel) else None,
 'overall':summarize(sel),
 'by_year':by_year,
 'by_market_role':by_role,
}
sel.to_csv(ROOT/'v5_frozen_logit020_later_ledger.csv',index=False)
pd.DataFrame(by_year).to_csv(ROOT/'v5_frozen_logit020_later_by_year.csv',index=False)
pd.DataFrame(by_role).to_csv(ROOT/'v5_frozen_logit020_later_by_market_role.csv',index=False)
pd.DataFrame(by_card).to_csv(ROOT/'v5_frozen_logit020_later_by_card.csv',index=False)
with open(ROOT/'v5_frozen_logit020_later_summary.json','w') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
print('\nBY YEAR\n',pd.DataFrame(by_year).to_string(index=False))
print('\nBY ROLE\n',pd.DataFrame(by_role).to_string(index=False))
print('\nBETS\n',sel[['date','event_name','outcome_label','american_odds','fair_market_p','model_p','abs_logit_residual','won','profit_units']].to_string(index=False))
