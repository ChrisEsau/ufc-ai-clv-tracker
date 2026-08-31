import pandas as pd
from pathlib import Path

SRC='data/research/prop_mispricing/xgboost_v5_exact_reproduction_oof_predictions.csv'
OUT='data/research/prop_mispricing/v5_oof_2021_2024_market_probability_cross.csv'

df=pd.read_csv(SRC)
df['date']=pd.to_datetime(df['date'])
for c in ['edge','fair_market_p','model_p','profit_per_100','won']:
    df[c]=pd.to_numeric(df[c], errors='coerce')
df=df[(df['date']>=pd.Timestamp('2021-01-01')) & (df['date']<=pd.Timestamp('2024-12-31'))].copy()
df=df[(df['market_key']=='moneyline') & (df['bookmaker']=='legacy_consensus') & (df['result_status']=='graded')].copy()
df=df.dropna(subset=['edge','fair_market_p','model_p','profit_per_100','won'])
idx=df.groupby('fight_id')['edge'].idxmax()
sides=df.loc[idx].copy().sort_values(['date','fight_id']).reset_index(drop=True)
sides=sides[sides.edge>=-1e-12].copy()
sides['edge']=sides.edge.clip(lower=0)
sides['bet_win']=sides.won.astype(int)
sides['profit_units']=sides.apply(lambda r: float(r.profit_per_100)/100.0 if int(r.bet_win)==1 else -1.0, axis=1)

p_bins=[0,.30,.40,.50,.60,.70,.80,1.000001]
p_labels=['<30%','30-40%','40-50%','50-60%','60-70%','70-80%','80%+']
sides['market_p_bucket']=pd.cut(sides.fair_market_p,bins=p_bins,labels=p_labels,right=False,include_lowest=True)
e_bins=[0,.02,.04,.06,.075,float('inf')]
e_labels=['0-2%','2-4%','4-6%','6-7.5%','7.5%+']
sides['edge_bucket_cross']=pd.cut(sides.edge,bins=e_bins,labels=e_labels,right=False,include_lowest=True)

periods=[('2021_2024',sides)] + [(str(y),sides[sides.date.dt.year==y]) for y in [2021,2022,2023,2024]]
rows=[]
for period,base in periods:
    for pb in p_labels:
        q=base[base.market_p_bucket==pb]; n=len(q); w=int(q.bet_win.sum()) if n else 0; pr=float(q.profit_units.sum()) if n else 0.0
        rows.append({'period':period,'analysis':'market_probability','market_p_bucket':pb,'edge_bucket':'ALL','bets':n,'wins':w,'losses':n-w,'win_rate':w/n if n else None,'profit_units':pr,'roi':pr/n if n else None,'avg_market_p':q.fair_market_p.mean() if n else None,'avg_v5_p':q.model_p.mean() if n else None,'avg_edge':q.edge.mean() if n else None})
    for pb in p_labels:
        for eb in e_labels:
            q=base[(base.market_p_bucket==pb)&(base.edge_bucket_cross==eb)]; n=len(q); w=int(q.bet_win.sum()) if n else 0; pr=float(q.profit_units.sum()) if n else 0.0
            rows.append({'period':period,'analysis':'market_x_edge','market_p_bucket':pb,'edge_bucket':eb,'bets':n,'wins':w,'losses':n-w,'win_rate':w/n if n else None,'profit_units':pr,'roi':pr/n if n else None,'avg_market_p':q.fair_market_p.mean() if n else None,'avg_v5_p':q.model_p.mean() if n else None,'avg_edge':q.edge.mean() if n else None})
res=pd.DataFrame(rows)
res.to_csv(OUT,index=False)
print('FIGHTS',len(sides),'DATE_RANGE',sides.date.min().date(),sides.date.max().date())
for period in ['2021_2024','2021','2022','2023','2024']:
    print('\n===',period,' MARKET PROBABILITY ===')
    print(res[(res.period==period)&(res.analysis=='market_probability')].to_string(index=False))
    print('\n===',period,' CROSS n>=10 ===')
    print(res[(res.period==period)&(res.analysis=='market_x_edge')&(res.bets>=10)].to_string(index=False))
