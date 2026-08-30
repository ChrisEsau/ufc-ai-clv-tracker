from pathlib import Path
import json
import pandas as pd

BASE = Path('data/market')
FILES = [
    'draftkings_raw_index.parquet',
    'draftkings_event_index.parquet',
    'draftkings_event_card_matches.parquet',
    'fanduel_event_index.parquet',
    'fanduel_event_card_matches.parquet',
]


def summarize(path: Path):
    df = pd.read_parquet(path)
    out = {'file': path.name, 'rows': int(len(df)), 'columns': [str(c) for c in df.columns]}
    date_cov = {}
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ('date','time','timestamp','snapshot','capture','scrape','created','start')):
            s = pd.to_datetime(df[c], errors='coerce', utc=True)
            if s.notna().any():
                date_cov[str(c)] = {'n': int(s.notna().sum()), 'min': str(s.min()), 'max': str(s.max())}
    out['date_coverage'] = date_cov
    uniq = {}
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ('run_id','event','fight','bout','path','url','file')):
            try:
                uniq[str(c)] = int(df[c].nunique(dropna=True))
            except Exception:
                pass
    out['unique_counts'] = uniq
    samples = {}
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ('run_id','path','url','event_name','event_title','source')):
            vals = df[c].dropna().astype(str)
            if len(vals):
                samples[str(c)] = vals.drop_duplicates().head(20).tolist()
    out['samples'] = samples
    return out


def main():
    result = {'files':[summarize(BASE/f) for f in FILES if (BASE/f).exists()]}
    print('RAW_INDEX_HISTORY=' + json.dumps(result, default=str, separators=(',',':')))

if __name__ == '__main__':
    main()
