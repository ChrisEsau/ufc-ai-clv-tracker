from pathlib import Path
import json
import pandas as pd

BASE = Path('data/market')


def dt_range(df, col):
    if col not in df.columns:
        return None
    s = pd.to_datetime(df[col], errors='coerce', utc=True)
    if not s.notna().any():
        return None
    return {'min': str(s.min()), 'max': str(s.max()), 'n': int(s.notna().sum())}


def summarize(path: Path):
    df = pd.read_parquet(path)
    out = {
        'file': path.name,
        'rows': int(len(df)),
        'cols': int(len(df.columns)),
        'columns': [str(c) for c in df.columns],
    }
    date_cols = {}
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ('date','time','timestamp','captured','snapshot','start','created','updated')):
            r = dt_range(df, c)
            if r:
                date_cols[str(c)] = r
    out['date_ranges'] = date_cols

    # useful categorical summaries
    for c in ('source','bookmaker','market_family','market_key','outcome_key','event_name','provider_event_id','snapshot_run_id'):
        if c in df.columns:
            out[f'{c}_unique'] = int(df[c].nunique(dropna=True))
            out[f'{c}_top'] = {str(k): int(v) for k,v in df[c].dropna().astype(str).value_counts().head(12).items()}

    # method-like content detection
    method_mask = pd.Series(False, index=df.index)
    for c in df.columns:
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            vals = s.fillna('').astype(str).str.lower()
            method_mask |= vals.str.contains(r'ko|tko|submission|decision|method', regex=True)
    out['method_like_rows'] = int(method_mask.sum())

    # likely historical/snapshot indicators
    out['has_snapshot_timestamp'] = 'snapshot_timestamp' in df.columns
    out['has_raw_payload_path'] = 'raw_payload_path' in df.columns
    if 'raw_payload_path' in df.columns:
        out['raw_payload_unique'] = int(df['raw_payload_path'].nunique(dropna=True))
        out['raw_payload_examples'] = df['raw_payload_path'].dropna().astype(str).drop_duplicates().head(8).tolist()
    return out


def main():
    files = sorted(BASE.rglob('*.parquet'))
    result = {'count': len(files), 'files': [summarize(p) for p in files]}
    print('DATA_MARKET_PARQUET_INVENTORY=' + json.dumps(result, separators=(',', ':'), default=str))

if __name__ == '__main__':
    main()
