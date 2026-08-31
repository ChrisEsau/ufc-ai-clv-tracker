from pathlib import Path
import json
import pandas as pd
import re

ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'data/market/market_intelligence_history.parquet'
F=ROOT/'data/features/moneyline_feature_view.parquet'
O=ROOT/'data/market/historical_market_outcomes.parquet'
OUT=ROOT/'data/research/prop_mispricing/market_intelligence_feature_join_diagnostic.json'

h=pd.read_parquet(H)
f=pd.read_parquet(F)
o=pd.read_parquet(O)
for d in (h,f,o):
    if 'fight_id' in d.columns: d['fight_id']=d['fight_id'].astype(str)
h['refresh_timestamp']=pd.to_datetime(h['refresh_timestamp'],errors='coerce',utc=True)
h=h[h['bookmaker'].astype(str).str.contains('DraftKings',case=False,na=False)].copy()
req=['moneyline','win_by_ko_tko_dq','win_by_submission','win_by_decision']
h=h[h['market_key'].isin(req)].copy()
counts=h.groupby(['fight_id','refresh_timestamp','market_key']).size().unstack(fill_value=0)
for k in req:
    if k not in counts.columns: counts[k]=0
good=counts[(counts[req]>=2).all(axis=1)].reset_index()
latest=good.sort_values('refresh_timestamp').groupby('fight_id',as_index=False).tail(1)[['fight_id','refresh_timestamp']]
chosen=h.merge(latest,on=['fight_id','refresh_timestamp'],how='inner')
ids=set(latest.fight_id)
fids=set(f.fight_id)
oids=set(o.fight_id)
nameish=[c for c in f.columns if any(t in c.lower() for t in ['fighter','name','red','blue','r_','b_'])][:100]
# Representative rows including identity-bearing fields.
sample=[]
for fid in latest.fight_id.head(12):
    z=chosen[chosen.fight_id.eq(fid)]
    cols=[c for c in ['fight_id','event_name','fight_display','market_key','market_display','outcome_key','comparison_key','outcome_display','side','fighter_name','american_odds','implied_probability','provider_selection_id'] if c in z.columns]
    sample.extend(z[cols].to_dict(orient='records'))
# Samples of feature rows sharing event/fighter-ish fields even if IDs differ.
f_sample_cols=[c for c in ['fight_id','date','event_name']+nameish if c in f.columns]
result={
 'market_unique_fights_all':int(h.fight_id.nunique()),
 'complete_latest_fights':int(len(latest)),
 'feature_unique_fights':int(f.fight_id.nunique()),
 'historical_outcome_unique_fights':int(o.fight_id.nunique()),
 'direct_feature_id_overlap':int(len(ids&fids)),
 'direct_outcome_id_overlap':int(len(ids&oids)),
 'feature_columns':list(f.columns),
 'feature_nameish_columns':nameish,
 'complete_fights': chosen.groupby('fight_id',as_index=False).first()[['fight_id','event_name','fight_display','refresh_timestamp']].head(100).astype(str).to_dict(orient='records'),
 'market_row_sample':sample,
 'feature_tail_sample':f[f_sample_cols].tail(30).astype(str).to_dict(orient='records'),
}
OUT.write_text(json.dumps(result,indent=2,default=str))
print(json.dumps(result,indent=2,default=str))
