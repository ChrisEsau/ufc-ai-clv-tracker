from pathlib import Path
import json
import pandas as pd

BASE = Path('data/market')
FILES = [
    'draftkings_market_catalog.parquet',
    'fanduel_market_catalog.parquet',
    'canonical_market_catalog.parquet',
]

REQUIRED_METHOD_KEYS = {
    'win_by_ko_tko_dq',
    'win_by_submission',
    'win_by_decision',
}


def summarize(path: Path):
    df = pd.read_parquet(path)
    out = {
        'file': path.name,
        'rows': int(len(df)),
        'snapshot_min': None,
        'snapshot_max': None,
        'event_min': None,
        'event_max': None,
        'unique_events': int(df['provider_event_id'].nunique()) if 'provider_event_id' in df else None,
    }
    for col, lo, hi in [
        ('snapshot_timestamp', 'snapshot_min', 'snapshot_max'),
        ('event_start_timestamp', 'event_min', 'event_max'),
    ]:
        if col in df:
            s = pd.to_datetime(df[col], errors='coerce', utc=True)
            if s.notna().any():
                out[lo] = str(s.min())
                out[hi] = str(s.max())

    fm = df[df.get('market_family', pd.Series('', index=df.index)).astype(str).eq('fighter_method_props')].copy()
    out['fighter_method_rows'] = int(len(fm))
    out['fighter_method_events'] = int(fm['provider_event_id'].nunique()) if len(fm) else 0

    event_details = []
    complete = 0
    incomplete = 0
    if len(fm):
        for event_id, g in fm.groupby('provider_event_id', sort=True):
            event_name = str(g['event_name'].iloc[0])
            fighters = sorted(set(g['fighter_name'].dropna().astype(str)))
            keys_by_fighter = {
                f: sorted(set(g.loc[g['fighter_name'].astype(str).eq(f), 'market_key'].dropna().astype(str)))
                for f in fighters
            }
            rows_by_fighter = {f: int((g['fighter_name'].astype(str) == f).sum()) for f in fighters}
            is_complete = (
                len(fighters) == 2
                and all(set(keys_by_fighter[f]) == REQUIRED_METHOD_KEYS for f in fighters)
                and len(g) == 6
            )
            complete += int(is_complete)
            incomplete += int(not is_complete)
            event_details.append({
                'provider_event_id': str(event_id),
                'event_name': event_name,
                'rows': int(len(g)),
                'fighters': fighters,
                'keys_by_fighter': keys_by_fighter,
                'rows_by_fighter': rows_by_fighter,
                'complete_six_way': bool(is_complete),
            })
    out['complete_six_way_events'] = complete
    out['incomplete_six_way_events'] = incomplete
    out['events'] = event_details
    return out


def main():
    result = {'files': [summarize(BASE / f) for f in FILES if (BASE / f).exists()]}
    print('SIX_WAY_COVERAGE=' + json.dumps(result, default=str, separators=(',', ':')))

if __name__ == '__main__':
    main()
