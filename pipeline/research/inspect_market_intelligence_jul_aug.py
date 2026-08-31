from pathlib import Path
import json
import pandas as pd

p=Path('data/market/market_intelligence_history.parquet')
df=pd.read_parquet(p)
print('COLUMNS', list(df.columns))
for c in ['refresh_timestamp','snapshot_timestamp','event_start_timestamp','event_date']:
    if c in df.columns:
        df[c]=pd.to_datetime(df[c],errors='coerce',utc=True)

start=pd.Timestamp('2026-07-01',tz='UTC'); end=pd.Timestamp('2026-08-31 23:59:59',tz='UTC')
timecol='snapshot_timestamp' if 'snapshot_timestamp' in df.columns else 'refresh_timestamp'
x=df[(df[timecol]>=start)&(df[timecol]<=end)].copy()

summary={'rows_all':len(df),'rows_jul_aug':len(x),'timecol':timecol}
for c in ['source','bookmaker','market_family','market_key','outcome_type','provider_subcategory_name']:
    if c in x.columns:
        summary[c+'_counts']=x[c].astype(str).value_counts().head(100).to_dict()

# DraftKings subset
mask=pd.Series(True,index=x.index)
if 'bookmaker' in x.columns: mask &= x['bookmaker'].astype(str).str.contains('DraftKings',case=False,na=False)
elif 'source' in x.columns: mask &= x['source'].astype(str).str.contains('draftkings',case=False,na=False)
dk=x[mask].copy()
summary['draftkings_rows']=len(dk)
for c in ['market_family','market_key','provider_subcategory_name']:
    if c in dk.columns: summary['dk_'+c+'_counts']=dk[c].astype(str).value_counts().head(100).to_dict()

# exact method-like rows
meth=pd.Series(False,index=dk.index)
for c in ['market_family','market_key','provider_market_type_name','provider_market_name','market_name']:
    if c in dk.columns:
        s=dk[c].astype(str)
        meth |= s.str.contains('method|win_by|ko|submission|decision',case=False,na=False)
dkm=dk[meth].copy()
summary['draftkings_method_like_rows']=len(dkm)
show=[c for c in ['refresh_timestamp','snapshot_timestamp','event_name','event_start_timestamp','provider_event_id','market_family','market_key','fighter_name','side','provider_market_name','provider_selection_name','american_odds','decimal_odds','implied_probability','provider_market_id','provider_selection_id'] if c in dkm.columns]
summary['method_sample']=dkm[show].head(100).astype(str).to_dict(orient='records')
if 'event_start_timestamp' in dkm.columns:
    summary['method_event_date_counts']=dkm.assign(_d=dkm['event_start_timestamp'].dt.date.astype(str))['_d'].value_counts().sort_index().to_dict()
if 'event_name' in dkm.columns:
    summary['method_event_counts']=dkm['event_name'].astype(str).value_counts().head(100).to_dict()

out=Path('data/research/prop_mispricing/market_intelligence_jul_aug_summary.json')
out.write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
