from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('data/research/prop_mispricing')
OOF = ROOT / 'v5_depth1_vs_depth2_oof.csv'
MARKET = Path('data/market/historical_market_outcomes.parquet')


def logit(p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    return np.log(p/(1-p))


oof=pd.read_csv(OOF)
oof['abs_logit_residual']=np.abs(logit(oof['depth1_p'])-logit(oof['market_p']))
oof['bet_side']=np.where(oof['depth1_p']>oof['market_p'],'red','blue')
oof['bet_won']=np.where(oof['bet_side'].eq('red'),oof['won'].astype(int),1-oof['won'].astype(int))

# Authoritative V5 market source: legacy_consensus graded two-way moneyline rows.
m=pd.read_parquet(MARKET).copy()
m['implied_probability']=pd.to_numeric(m['implied_probability'],errors='coerce')
m=m[(m['bookmaker']=='legacy_consensus')&(m['result_status']=='graded')&(m['market_key']=='moneyline')].dropna(subset=['fight_id','outcome_side','implied_probability']).copy()
m['outcome_side']=m['outcome_side'].astype(str)
good=m.groupby('fight_id').size(); good=good[good==2].index
m=m[m['fight_id'].isin(good)].copy()
# Raw implied probability is the actual consensus price input before vig removal.
quotes=m[['fight_id','outcome_side','implied_probability']].drop_duplicates(['fight_id','outcome_side'],keep='last')
oof=oof.merge(quotes,left_on=['fight_id','bet_side'],right_on=['fight_id','outcome_side'],how='left',validate='one_to_one')
if oof['implied_probability'].isna().any():
    raise RuntimeError(f"missing moneyline quote for {int(oof['implied_probability'].isna().sum())} OOF bets")
raw_side_p=oof['implied_probability'].to_numpy(float)
if ((raw_side_p<=0)|(raw_side_p>=1)).any():
    raise RuntimeError('invalid raw implied probability encountered')
oof['bet_raw_implied_p']=raw_side_p
oof['bet_decimal_odds']=1.0/raw_side_p
oof['profit_units']=np.where(oof['bet_won']==1,oof['bet_decimal_odds']-1.0,-1.0)
oof['logit_decile']=pd.qcut(oof['abs_logit_residual'],10,labels=False,duplicates='drop')+1

def summarize(g):
    n=len(g); prof=float(g['profit_units'].sum())
    return pd.Series({'n':n,'wins':int(g['bet_won'].sum()),'win_rate':float(g['bet_won'].mean()),'mean_abs_logit_residual':float(g['abs_logit_residual'].mean()),'mean_raw_implied_p':float(g['bet_raw_implied_p'].mean()),'mean_decimal_odds':float(g['bet_decimal_odds'].mean()),'profit_units':prof,'roi':prof/n if n else np.nan})

dec=oof.groupby('logit_decile',observed=True).apply(summarize,include_groups=False).reset_index()
tails=[]
for k in range(1,11):
    g=oof[oof['logit_decile']>=k]; s=summarize(g).to_dict(); s['min_decile']=k; tails.append(s)
tails=pd.DataFrame(tails)
cutoffs=[0.05,0.10,0.15,0.20,0.25,0.30,0.40]
rows=[]
for c in cutoffs:
    g=oof[oof['abs_logit_residual']>=c]; s=summarize(g).to_dict(); s['min_abs_logit_residual']=c; rows.append(s)
thresholds=pd.DataFrame(rows)
year=[]
for y,g in oof[oof['logit_decile']>=9].groupby('fold'):
    s=summarize(g).to_dict(); s['year']=int(y); year.append(s)
year=pd.DataFrame(year)
summary={'experiment':'v5_logit_residual_flat_bet_roi_diagnostic_v1','n':int(len(oof)),'pricing_semantics':'flat 1u bets at authoritative legacy_consensus raw implied moneyline probability from the exact V5 historical market snapshot; decimal odds = 1/raw implied probability','selection_note':'measurement only on 2021-2024 OOF; no threshold promoted','overall_if_bet_every_v5_direction':summarize(oof).to_dict(),'top_decile':summarize(oof[oof.logit_decile==10]).to_dict(),'top_two_deciles':summarize(oof[oof.logit_decile>=9]).to_dict(),'top_three_deciles':summarize(oof[oof.logit_decile>=8]).to_dict()}
dec.to_csv(ROOT/'v5_logit_residual_roi_deciles.csv',index=False)
tails.to_csv(ROOT/'v5_logit_residual_roi_top_tails.csv',index=False)
thresholds.to_csv(ROOT/'v5_logit_residual_roi_thresholds.csv',index=False)
year.to_csv(ROOT/'v5_logit_residual_roi_top20_by_year.csv',index=False)
oof.to_csv(ROOT/'v5_logit_residual_roi_ledger.csv',index=False)
with open(ROOT/'v5_logit_residual_roi_summary.json','w') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
print('\nDECILES\n',dec.to_string(index=False))
print('\nTOP TAILS\n',tails.to_string(index=False))
print('\nFIXED THRESHOLDS\n',thresholds.to_string(index=False))
print('\nTOP 20% BY YEAR\n',year.to_string(index=False))
