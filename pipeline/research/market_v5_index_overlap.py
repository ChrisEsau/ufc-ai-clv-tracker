from __future__ import annotations

import json
import pandas as pd

idx = pd.read_parquet('data/market/draftkings_raw_index.parquet')
h = pd.read_parquet('data/market/market_intelligence_history.parquet')
fv = pd.read_parquet('data/features/moneyline_feature_view.parquet')
master = pd.read_parquet('data/master/ufc_master.parquet')

runs = set(idx['snapshot_run_id'].astype(str))
h = h[(h['bookmaker'].eq('DraftKings')) & (h['market_key'].eq('moneyline'))].copy()
h['source_run_id'] = h['source_run_id'].astype(str)
h = h[h['source_run_id'].isin(runs)].copy()

sample_cols = [c for c in ['fight_id','event_name','fight_display','fighter_name','side','outcome_key','comparison_key','outcome_display','market_display','american_odds','source_run_id'] if c in h.columns]
report = {
    'index_runs': int(idx['snapshot_run_id'].nunique()),
    'index_provider_events': int(idx['provider_event_id'].nunique()),
    'history_indexed_ml_rows': int(len(h)),
    'history_indexed_fights': int(h['fight_id'].nunique()),
    'feature_rows': int(len(fv)),
    'feature_fights': int(fv['fight_id'].nunique()),
    'feature_date_min': str(pd.to_datetime(fv['date'], errors='coerce').min()) if 'date' in fv else None,
    'feature_date_max': str(pd.to_datetime(fv['date'], errors='coerce').max()) if 'date' in fv else None,
    'history_feature_fight_overlap': int(len(set(h['fight_id'].dropna().astype(str)) & set(fv['fight_id'].dropna().astype(str)))),
    'master_rows': int(len(master)),
    'master_date_max': str(pd.to_datetime(master['date'], errors='coerce').max()) if 'date' in master else None,
    'history_master_fight_overlap': int(len(set(h['fight_id'].dropna().astype(str)) & set(master['fight_id'].dropna().astype(str)))),
    'history_side_values': h['side'].astype(str).value_counts(dropna=False).head(20).to_dict() if 'side' in h else None,
    'history_outcome_key_values': h['outcome_key'].astype(str).value_counts(dropna=False).head(20).to_dict() if 'outcome_key' in h else None,
    'feature_name_columns': [c for c in ['r_name','b_name','red_fighter','blue_fighter','red_name','blue_name'] if c in fv.columns],
    'master_name_columns': [c for c in ['r_name','b_name','red_fighter','blue_fighter','red_name','blue_name'] if c in master.columns],
    'history_sample': h[sample_cols].drop_duplicates().head(30).to_dict('records'),
}
print('V5_INDEX_OVERLAP=' + json.dumps(report, default=str, separators=(',', ':')), flush=True)
