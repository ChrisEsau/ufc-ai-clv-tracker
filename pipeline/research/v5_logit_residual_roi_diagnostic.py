from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('data/research/prop_mispricing')
OOF = ROOT / 'v5_depth1_vs_depth2_oof.csv'
FEAT = Path('data/features/moneyline_feature_view.parquet')


def logit(p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    return np.log(p/(1-p))


oof=pd.read_csv(OOF)
oof['abs_logit_residual']=np.abs(logit(oof['depth1_p'])-logit(oof['market_p']))
oof['model_side_red']=oof['depth1_p']>oof['market_p']
oof['bet_won']=np.where(oof['model_side_red'],oof['won'].astype(int),1-oof['won'].astype(int))

feat=pd.read_parquet(FEAT)
if 'fight_id' not in feat.columns or 'market_overround' not in feat.columns:
    raise RuntimeError(f'feature view missing required columns; cols={list(feat.columns)}')
mo=feat[['fight_id','market_overround']].copy()
mo=mo.dropna(subset=['fight_id']).drop_duplicates('fight_id',keep='last')
oof=oof.merge(mo,on='fight_id',how='left',validate='one_to_one')
if oof['market_overround'].isna().any():
    missing=int(oof['market_overround'].isna().sum())
    raise RuntimeError(f'missing market_overround for {missing} OOF fights')

# Recover the raw two-way implied probability for the side V5 favors.
# V5 fair_market_p was defined as raw_implied_probability / market_overround.
# Thus raw side p = fair side p * overround, and decimal payout = 1/raw side p.
fair_side=np.where(oof['model_side_red'],oof['market_p'],1-oof['market_p'])
raw_side_p=fair_side*oof['market_overround'].to_numpy(float)
if ((raw_side_p<=0)|(raw_side_p>=1)).any():
    bad=int(((raw_side_p<=0)|(raw_side_p>=1)).sum())
    raise RuntimeError(f'invalid recovered raw implied probabilities: {bad}')
oof['bet_fair_market_p']=fair_side
oof['bet_raw_implied_p']=raw_side_p
oof['bet_decimal_odds']=1.0/raw_side_p
oof['profit_units']=np.where(oof['bet_won']==1,oof['bet_decimal_odds']-1.0,-1.0)

oof['logit_decile']=pd.qcut(oof['abs_logit_residual'],10,labels=False,duplicates='drop')+1

def summarize(g):
    n=len(g); prof=float(g['profit_units'].sum())
    return pd.Series({
        'n':n,
        'wins':int(g['bet_won'].sum()),
        'win_rate':float(g['bet_won'].mean()),
        'mean_abs_logit_residual':float(g['abs_logit_residual'].mean()),
        'mean_fair_market_p':float(g['bet_fair_market_p'].mean()),
        'mean_decimal_odds':float(g['bet_decimal_odds'].mean()),
        'profit_units':prof,
        'roi':prof/n if n else np.nan,
    })

dec=oof.groupby('logit_decile',observed=True).apply(summarize,include_groups=False).reset_index()
# Cumulative top-tail portfolio: decile k means bet deciles k..10.
tails=[]
for k in range(1,11):
    g=oof[oof['logit_decile']>=k]
    s=summarize(g).to_dict(); s['min_decile']=k
    tails.append(s)
tails=pd.DataFrame(tails)

# Fixed, predeclared logit cutoffs for readability; measurement only, not selection.
cutoffs=[0.05,0.10,0.15,0.20,0.25,0.30,0.40]
rows=[]
for c in cutoffs:
    g=oof[oof['abs_logit_residual']>=c]
    s=summarize(g).to_dict(); s['min_abs_logit_residual']=c
    rows.append(s)
thresholds=pd.DataFrame(rows)

# Year stability for top quintile (deciles 9-10), a natural rank-based diagnostic not a tuned threshold.
year=[]
for y,g in oof[oof['logit_decile']>=9].groupby('fold'):
    s=summarize(g).to_dict(); s['year']=int(y)
    year.append(s)
year=pd.DataFrame(year)

summary={
    'experiment':'v5_logit_residual_flat_bet_roi_diagnostic_v1',
    'n':int(len(oof)),
    'pricing_semantics':'flat 1u bets at recovered raw two-way consensus price used to construct V5 market input: raw side implied p = fair side p * market_overround; decimal odds = 1/raw side implied p',
    'selection_note':'measurement only on 2021-2024 OOF; no threshold is promoted by this script',
    'overall_if_bet_every_v5_direction':summarize(oof).to_dict(),
    'top_decile':summarize(oof[oof.logit_decile==10]).to_dict(),
    'top_two_deciles':summarize(oof[oof.logit_decile>=9]).to_dict(),
    'top_three_deciles':summarize(oof[oof.logit_decile>=8]).to_dict(),
}

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
