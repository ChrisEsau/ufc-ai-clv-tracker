from pathlib import Path
import json
import pandas as pd

BASE = Path('data/market')
FILES = [
    'draftkings_market_catalog.parquet',
    'fanduel_market_catalog.parquet',
    'canonical_market_catalog.parquet',
]
METHOD_TERMS = ('method', 'ko', 'tko', 'submission', 'decision')
KEY_HINTS = ('market', 'category', 'type', 'outcome', 'selection', 'label', 'fighter', 'event', 'fight', 'participant', 'name')


def summarize(path: Path):
    df = pd.read_parquet(path)
    out = {
        'file': path.name,
        'rows': int(len(df)),
        'columns': [str(c) for c in df.columns],
        'dtypes': {str(c): str(df[c].dtype) for c in df.columns},
    }

    date_cov = {}
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ('date', 'time', 'captured', 'scrape', 'start')):
            s = pd.to_datetime(df[c], errors='coerce', utc=True)
            if s.notna().any():
                date_cov[str(c)] = {
                    'n': int(s.notna().sum()),
                    'min': str(s.min()),
                    'max': str(s.max()),
                }
    out['date_coverage'] = date_cov

    categorical = {}
    method_hits = {}
    method_mask = pd.Series(False, index=df.index)
    for c in df.columns:
        s = df[c]
        if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
            vals = s.fillna('').astype(str)
            low = vals.str.lower()
            hit = low.apply(lambda x: any(t in x for t in METHOD_TERMS))
            method_mask |= hit
            if hit.any():
                method_hits[str(c)] = vals[hit].value_counts().head(50).to_dict()
            if any(k in str(c).lower() for k in KEY_HINTS):
                categorical[str(c)] = vals[vals.ne('')].value_counts().head(30).to_dict()

    m = df[method_mask]
    out['method_like_rows'] = int(len(m))
    out['method_hits'] = method_hits
    out['key_values'] = categorical

    # Compact uniqueness counts for likely identifiers.
    uniq = {}
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ('event', 'fight', 'bout', 'market', 'fighter', 'participant')):
            try:
                uniq[str(c)] = int(df[c].nunique(dropna=True))
            except Exception:
                pass
    out['unique_counts'] = uniq

    # Keep only 8 method rows so schema/value semantics are visible in logs.
    if len(m):
        sample_cols = [c for c in df.columns if any(k in str(c).lower() for k in KEY_HINTS) or str(c).lower() in ('odds','price','american_odds','captured_at','snapshot_at')]
        sample_cols = sample_cols[:20] or list(df.columns[:12])
        sm = m[sample_cols].head(8).copy()
        out['method_samples'] = sm.astype(object).where(pd.notna(sm), None).to_dict('records')
    else:
        out['method_samples'] = []
    return out


def main():
    result = {'files': [summarize(BASE / f) for f in FILES if (BASE / f).exists()]}
    print('COMPACT_METHOD_COVERAGE=' + json.dumps(result, default=str, separators=(',', ':')))

if __name__ == '__main__':
    main()
