from pathlib import Path
import pandas as pd

P=Path('data/market/market_intelligence_history.parquet')
OUT=Path('data/research/prop_mispricing/abudhabi_dk_method_snapshot_inventory.csv')
df=pd.read_parquet(P)
df['refresh_timestamp']=pd.to_datetime(df['refresh_timestamp'],utc=True,errors='coerce')
x=df[(df['bookmaker']=='DraftKings') & (df['event_name']=='UFC Fight Night: Ankalaev vs. Guskov') & df['market_key'].isin(['win_by_ko_tko_dq','win_by_submission','win_by_decision'])].copy()
cols=['refresh_timestamp','fight_id','fight_display','market_key','outcome_key','comparison_key','outcome_display','side','fighter_name','american_odds','implied_probability','provider_event_id','provider_market_id','provider_selection_id']
x=x[cols].sort_values(['refresh_timestamp','fight_id','market_key','outcome_display'])
x.to_csv(OUT,index=False)
print('ROWS',len(x))
print('REFRESHES')
print(x.groupby('refresh_timestamp').size().to_string())
print('\nFIGHTS')
print(x[['fight_id','fight_display']].drop_duplicates().to_string(index=False))
print('\nLATEST_ROWS_SAMPLE')
latest=x[x.refresh_timestamp==x.refresh_timestamp.max()]
print(latest[['refresh_timestamp','fight_id','fight_display','market_key','outcome_display','american_odds','comparison_key','outcome_key']].to_string(index=False))
print('OUTPUT',OUT)
