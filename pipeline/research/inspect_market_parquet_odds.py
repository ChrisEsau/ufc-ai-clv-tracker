from pathlib import Path
import json
import pandas as pd

ROOT = Path('data/market')
OUT = Path('data/research/prop_mispricing/market_parquet_odds_inventory.json')
OUT.parent.mkdir(parents=True, exist_ok=True)

KEYWORDS = ('odds','price','prob','implied','decimal','american','market','selection','event','fighter','snapshot','timestamp','date')
report = {}
for p in sorted(ROOT.glob('*.parquet')):
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        report[p.name] = {'error': str(e)}
        continue
    cols = list(df.columns)
    interesting = [c for c in cols if any(k in c.lower() for k in KEYWORDS)]
    item = {
        'rows': int(len(df)),
        'columns': cols,
        'interesting_columns': interesting,
    }
    # date/time ranges
    for c in cols:
        lc=c.lower()
        if any(k in lc for k in ('date','timestamp','time')):
            try:
                s=pd.to_datetime(df[c], errors='coerce', utc=True)
                if s.notna().any(): item[f'{c}_range']={'min':str(s.min()),'max':str(s.max())}
            except Exception:
                pass
    # summarize likely odds/probability columns
    for c in cols:
        lc=c.lower()
        if any(k in lc for k in ('odds','price','prob','implied','decimal','american')):
            s=df[c]
            item[f'{c}_non_null']=int(s.notna().sum())
            try:
                nums=pd.to_numeric(s,errors='coerce')
                if nums.notna().any():
                    item[f'{c}_numeric']={'min':float(nums.min()),'max':float(nums.max()),'sample':nums.dropna().head(8).astype(float).tolist()}
                else:
                    item[f'{c}_sample']=s.dropna().astype(str).head(8).tolist()
            except Exception:
                item[f'{c}_sample']=s.dropna().astype(str).head(8).tolist()
    # July-Aug 2026 sample rows if a date-like column exists
    date_col = next((c for c in cols if c.lower() in ('snapshot_timestamp','event_start_timestamp','event_date','date','fight_date')), None)
    if date_col:
        try:
            ts=pd.to_datetime(df[date_col],errors='coerce',utc=True)
            m=(ts>=pd.Timestamp('2026-07-01',tz='UTC'))&(ts<=pd.Timestamp('2026-08-31 23:59:59',tz='UTC'))
            item['jul_aug_2026_rows']=int(m.sum())
            sample_cols=[c for c in cols if c in interesting][:20]
            item['jul_aug_2026_sample']=df.loc[m,sample_cols].head(12).astype(str).to_dict(orient='records') if m.any() else []
        except Exception:
            pass
    report[p.name]=item

OUT.write_text(json.dumps(report,indent=2,default=str))
print(json.dumps(report,indent=2,default=str))
