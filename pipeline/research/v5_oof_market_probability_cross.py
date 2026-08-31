import pandas as pd
import numpy as np

OOF='data/research/prop_mispricing/xgboost_v5_exact_reproduction_selected_oof.csv'
MARKET='data/research/prop_mispricing/_v5_frozen_historical_market_outcomes.parquet'
OUT='data/research/prop_mispricing/v5_oof_2021_2024_market_probability_cross.csv'
BETS_OUT='data/research/prop_mispricing/v5_oof_2021_2024_positive_edge_fights.csv'

oof=pd.read_csv(OOF)
oof['date']=pd.to_datetime(oof['date'])
for c in ['won','model_p','market_p']:
    oof[c]=pd.to_numeric(oof[c],errors='coerce')
oof=oof[(oof.date>=pd.Timestamp('2021-01-01'))&(oof.date<=pd.Timestamp('2024-12-31'))].dropna().copy()

m=pd.read_parquet(MARKET)
m=m[(m['market_key']=='moneyline')&(m['bookmaker']=='legacy_consensus')&(m['result_status']=='graded')].copy()
m['implied_probability']=pd.to_numeric(m['implied_probability'],errors='coerce')
m['profit_per_100']=pd.to_numeric(m['profit_per_100'],errors='coerce')
m=m.dropna(subset=['fight_id','implied_probability','profit_per_100'])
counts=m.groupby('fight_id').size()
valid=counts[counts==2].index
m=m[m.fight_id.isin(valid)].copy()
m['overround']=m.groupby('fight_id')['implied_probability'].transform('sum')
m['fair_p']=m['implied_probability']/m['overround']

rows=[]
for r in oof.itertuples(index=False):
    q=m[m.fight_id==r.fight_id].copy()
    if len(q)!=2:
        continue
    # OOF market_p is canonical RED fair probability. Identify the RED market row by exact/nearest fair-p match.
    q['red_match_err']=(q.fair_p-float(r.market_p)).abs()
    q=q.sort_values('red_match_err').reset_index(drop=True)
    red=q.iloc[0]; blue=q.iloc[1]
    if float(red.red_match_err)>1e-10:
        raise RuntimeError(f'Cannot map RED market side for {r.fight_id}: OOF={r.market_p}, fairs={q.fair_p.tolist()}')
    red_edge=float(r.model_p-r.market_p)
    if red_edge>=0:
        liked=red
        liked_market=float(r.market_p)
        liked_model=float(r.model_p)
        liked_win=int(r.won)
        liked_side='RED'
    else:
        liked=blue
        liked_market=float(1.0-r.market_p)
        liked_model=float(1.0-r.model_p)
        liked_win=int(1-r.won)
        liked_side='BLUE'
    edge=liked_model-liked_market
    payout=float(liked.profit_per_100)/100.0 if liked_win==1 else -1.0
    rows.append({
        'fight_id':r.fight_id,'date':r.date,'liked_side':liked_side,
        'fair_market_p':liked_market,'model_p':liked_model,'edge':edge,
        'bet_win':liked_win,'profit_per_100':float(liked.profit_per_100),'profit_units':payout,
    })

sides=pd.DataFrame(rows).sort_values(['date','fight_id']).reset_index(drop=True)
if len(sides)!=len(oof):
    raise RuntimeError(f'Expected {len(oof)} mapped OOF fights, got {len(sides)}')
sides.to_csv(BETS_OUT,index=False)

p_bins=[0,.30,.40,.50,.60,.70,.80,1.000001]
p_labels=['<30%','30-40%','40-50%','50-60%','60-70%','70-80%','80%+']
sides['market_p_bucket']=pd.cut(sides.fair_market_p,bins=p_bins,labels=p_labels,right=False,include_lowest=True)
e_bins=[0,.02,.04,.06,.075,float('inf')]
e_labels=['0-2%','2-4%','4-6%','6-7.5%','7.5%+']
sides['edge_bucket_cross']=pd.cut(sides.edge,bins=e_bins,labels=e_labels,right=False,include_lowest=True)
periods=[('2021_2024',sides)]+[(str(y),sides[sides.date.dt.year==y]) for y in [2021,2022,2023,2024]]
res=[]
for period,base in periods:
    for pb in p_labels:
        q=base[base.market_p_bucket==pb]; n=len(q); w=int(q.bet_win.sum()) if n else 0; pr=float(q.profit_units.sum()) if n else 0.0
        res.append({'period':period,'analysis':'market_probability','market_p_bucket':pb,'edge_bucket':'ALL','bets':n,'wins':w,'losses':n-w,'win_rate':w/n if n else np.nan,'profit_units':pr,'roi':pr/n if n else np.nan,'avg_market_p':q.fair_market_p.mean() if n else np.nan,'avg_v5_p':q.model_p.mean() if n else np.nan,'avg_edge':q.edge.mean() if n else np.nan})
    for pb in p_labels:
        for eb in e_labels:
            q=base[(base.market_p_bucket==pb)&(base.edge_bucket_cross==eb)]; n=len(q); w=int(q.bet_win.sum()) if n else 0; pr=float(q.profit_units.sum()) if n else 0.0
            res.append({'period':period,'analysis':'market_x_edge','market_p_bucket':pb,'edge_bucket':eb,'bets':n,'wins':w,'losses':n-w,'win_rate':w/n if n else np.nan,'profit_units':pr,'roi':pr/n if n else np.nan,'avg_market_p':q.fair_market_p.mean() if n else np.nan,'avg_v5_p':q.model_p.mean() if n else np.nan,'avg_edge':q.edge.mean() if n else np.nan})
res=pd.DataFrame(res)
res.to_csv(OUT,index=False)
print('FIGHTS',len(sides),'DATE_RANGE',sides.date.min().date(),sides.date.max().date())
for period in ['2021_2024','2021','2022','2023','2024']:
    print('\n===',period,' MARKET PROBABILITY ===')
    print(res[(res.period==period)&(res.analysis=='market_probability')].to_string(index=False))
    print('\n===',period,' CROSS n>=10 ===')
    print(res[(res.period==period)&(res.analysis=='market_x_edge')&(res.bets>=10)].to_string(index=False))
