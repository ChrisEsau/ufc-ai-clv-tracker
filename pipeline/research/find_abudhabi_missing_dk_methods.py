import pandas as pd
p='data/market/market_intelligence_history.parquet'
df=pd.read_parquet(p)
df['refresh_timestamp']=pd.to_datetime(df['refresh_timestamp'],utc=True,errors='coerce')
for term in ['Jacoby','Saidov','Gibson','Hussein','Dulatov','Turman']:
    mask=pd.Series(False,index=df.index)
    for c in ['fight_display','outcome_display','comparison_key','fighter_name']:
        mask |= df[c].astype(str).str.contains(term,case=False,na=False)
    x=df[(df.bookmaker=='DraftKings') & mask & df.market_key.isin(['win_by_ko_tko_dq','win_by_submission','win_by_decision'])]
    print('\nTERM',term,'ROWS',len(x))
    if len(x):
        print(x[['refresh_timestamp','event_name','fight_id','fight_display','market_key','outcome_display','american_odds','comparison_key']].sort_values('refresh_timestamp').tail(30).to_string(index=False))
