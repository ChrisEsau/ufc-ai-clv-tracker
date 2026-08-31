from pathlib import Path
import json
import pandas as pd

OUT=Path('data/research/prop_mispricing'); OUT.mkdir(parents=True,exist_ok=True)
paths=[
 'data/market/historical_market_outcomes.parquet',
 'data/market/market_intelligence_history.parquet',
 'data/market/draftkings_raw_index.parquet',
]
report={}
for p in paths:
    path=Path(p)
    if not path.exists():
        report[p]={'exists':False}; continue
    df=pd.read_parquet(path)
    info={'exists':True,'rows':int(len(df)),'columns':list(df.columns)}
    if 'date' in df: 
        d=pd.to_datetime(df['date'],errors='coerce'); info['date_min']=str(d.min()); info['date_max']=str(d.max())
    elif 'captured_at' in df:
        d=pd.to_datetime(df['captured_at'],errors='coerce'); info['date_min']=str(d.min()); info['date_max']=str(d.max())
    for c in ['market_key','market_type','bookmaker','source','outcome_type','selection_type']:
        if c in df.columns:
            info[c+'_counts']=df[c].astype(str).value_counts(dropna=False).head(30).to_dict()
    report[p]=info

# Detailed historical_market_outcomes method-like coverage
p=Path('data/market/historical_market_outcomes.parquet')
if p.exists():
    df=pd.read_parquet(p)
    if 'date' in df: df['date']=pd.to_datetime(df.date,errors='coerce')
    mk=df['market_key'].astype(str) if 'market_key' in df else pd.Series('',index=df.index)
    method=df[mk.str.contains('method|ko|sub|decision|dec',case=False,regex=True,na=False)].copy()
    detail={'method_like_rows':int(len(method))}
    if len(method):
        detail['market_key_counts']=method['market_key'].astype(str).value_counts().to_dict() if 'market_key' in method else {}
        if 'date' in method:
            detail['year_counts']=method.assign(year=method.date.dt.year).groupby('year').size().to_dict()
            detail['date_min']=str(method.date.min()); detail['date_max']=str(method.date.max())
        if 'fight_id' in method:
            detail['unique_fights']=int(method.fight_id.nunique())
            if 'date' in method:
                detail['unique_fights_by_year']=method.assign(year=method.date.dt.year).groupby('year').fight_id.nunique().to_dict()
    report['historical_method_detail']=detail

with open(OUT/'v5_method_market_coverage_audit.json','w') as f: json.dump(report,f,indent=2,default=str)
print(json.dumps(report,indent=2,default=str))
