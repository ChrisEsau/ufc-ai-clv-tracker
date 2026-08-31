from pathlib import Path
import json
import pandas as pd

ROOT = Path('data/market')
OUT = Path('data/research/prop_mispricing')
OUT.mkdir(parents=True, exist_ok=True)
FILES = [
    'draftkings_event_index.parquet',
    'draftkings_raw_index.parquet',
    'draftkings_market_catalog.parquet',
    'draftkings_event_card_matches.parquet',
]
START = pd.Timestamp('2026-07-01')
END = pd.Timestamp('2026-08-31 23:59:59')

summary = {}
for name in FILES:
    p = ROOT / name
    df = pd.read_parquet(p)
    info = {'rows': int(len(df)), 'columns': list(df.columns), 'dtypes': {c: str(df[c].dtype) for c in df.columns}}
    date_cols = [c for c in df.columns if any(tok in c.lower() for tok in ['date','time','commence','timestamp'])]
    info['date_columns'] = date_cols
    samples = {}
    for c in date_cols:
        s = pd.to_datetime(df[c], errors='coerce', utc=True)
        if s.notna().any():
            samples[c] = {'min': str(s.min()), 'max': str(s.max()), 'non_null': int(s.notna().sum())}
    info['date_ranges'] = samples
    summary[name] = info

# Event index: export likely July-Aug rows using every parseable date-like column.
e = pd.read_parquet(ROOT / 'draftkings_event_index.parquet').copy()
mask = pd.Series(False, index=e.index)
for c in [c for c in e.columns if any(tok in c.lower() for tok in ['date','time','commence','timestamp'])]:
    s = pd.to_datetime(e[c], errors='coerce', utc=True)
    mask |= (s >= START.tz_localize('UTC')) & (s <= END.tz_localize('UTC'))
sel = e[mask].copy()
sel.to_csv(OUT / 'draftkings_jul_aug_2026_event_index_rows.csv', index=False)
summary['event_index_selected_rows'] = int(len(sel))
summary['event_index_selected_preview'] = sel.head(50).astype(str).to_dict(orient='records')

# Raw/catalog: keep rows whose text contains 2026-07 or 2026-08 as fallback inventory.
for name in ['draftkings_raw_index.parquet','draftkings_market_catalog.parquet','draftkings_event_card_matches.parquet']:
    df = pd.read_parquet(ROOT / name).copy()
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        if any(tok in c.lower() for tok in ['date','time','commence','timestamp']):
            s = pd.to_datetime(df[c], errors='coerce', utc=True)
            mask |= (s >= START.tz_localize('UTC')) & (s <= END.tz_localize('UTC'))
    out = df[mask].copy()
    out.to_csv(OUT / f'{Path(name).stem}_jul_aug_2026.csv', index=False)
    summary[f'{name}_selected_rows'] = int(len(out))
    summary[f'{name}_selected_preview'] = out.head(30).astype(str).to_dict(orient='records')

(OUT / 'draftkings_jul_aug_2026_inventory_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
