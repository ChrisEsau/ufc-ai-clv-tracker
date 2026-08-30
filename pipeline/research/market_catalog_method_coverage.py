from pathlib import Path
import json
import pandas as pd

BASE = Path('data/market')
OUT = Path('data/research/prop_mispricing/market_catalog_method_coverage.json')

FILES = [
    'draftkings_market_catalog.parquet',
    'fanduel_market_catalog.parquet',
    'canonical_market_catalog.parquet',
]

METHOD_TERMS = ('method', 'ko', 'tko', 'submission', 'decision')


def summarize(path: Path):
    df = pd.read_parquet(path)
    out = {
        'file': str(path),
        'rows': int(len(df)),
        'columns': list(map(str, df.columns)),
    }

    # Date/time coverage across plausible columns.
    date_cov = {}
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ('date', 'time', 'captured', 'scrape', 'event_start')):
            s = pd.to_datetime(df[c], errors='coerce', utc=True)
            if s.notna().any():
                date_cov[str(c)] = {
                    'non_null': int(s.notna().sum()),
                    'min': str(s.min()),
                    'max': str(s.max()),
                }
    out['date_coverage'] = date_cov

    # Enumerate categorical-ish text columns and method-related values.
    text_summary = {}
    method_hits = {}
    for c in df.columns:
        s = df[c]
        if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
            vals = s.dropna().astype(str)
            nunique = vals.nunique()
            if nunique <= 300:
                vc = vals.value_counts().head(100)
                text_summary[str(c)] = {
                    'nunique': int(nunique),
                    'top_values': {str(k): int(v) for k, v in vc.items()},
                }
            mask = vals.str.lower().apply(lambda x: any(t in x for t in METHOD_TERMS))
            if mask.any():
                vc = vals[mask].value_counts().head(200)
                method_hits[str(c)] = {str(k): int(v) for k, v in vc.items()}
    out['text_summary'] = text_summary
    out['method_hits'] = method_hits

    # Row-level method-like records based on all text columns.
    method_mask = pd.Series(False, index=df.index)
    for c in df.columns:
        s = df[c]
        if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
            low = s.fillna('').astype(str).str.lower()
            method_mask |= low.apply(lambda x: any(t in x for t in METHOD_TERMS))
    m = df[method_mask].copy()
    out['method_like_rows'] = int(len(m))

    # Save representative rows as records, bounded for JSON size.
    if len(m):
        sample = m.head(100).copy()
        for c in sample.columns:
            if pd.api.types.is_datetime64_any_dtype(sample[c]):
                sample[c] = sample[c].astype(str)
        out['method_sample_rows'] = sample.astype(object).where(pd.notna(sample), None).to_dict('records')
    else:
        out['method_sample_rows'] = []

    return out


def main():
    result = {'files': []}
    for name in FILES:
        p = BASE / name
        if p.exists():
            result['files'].append(summarize(p))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    print(json.dumps(result, indent=2, default=str))

if __name__ == '__main__':
    main()
