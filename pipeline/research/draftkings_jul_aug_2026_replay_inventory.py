from pathlib import Path
import json
import subprocess
import pandas as pd

from pipeline.market.providers.draftkings_public import DraftKingsSnapshot, flatten_market_diagnostics
from pipeline.market.normalizers.draftkings import normalize_draftkings_diagnostic_rows

ROOT=Path('data/market')
OUT=Path('data/research/prop_mispricing')
OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2026-07-01',tz='UTC')
END=pd.Timestamp('2026-08-31 23:59:59',tz='UTC')

# Build one path->blob map for all historical DK raw files. This is much faster
# than walking Git history separately for every indexed payload.
obj_out=subprocess.check_output(['git','rev-list','--all','--objects','--','data/market/raw/draftkings'],text=True)
BLOBS={}
for line in obj_out.splitlines():
    parts=line.split(' ',1)
    if len(parts)==2 and parts[1].startswith('data/market/raw/draftkings/'):
        BLOBS.setdefault(parts[1],parts[0])


def _historical_text(path: str) -> str | None:
    p=Path(path)
    if p.exists(): return p.read_text()
    sha=BLOBS.get(path)
    if not sha: return None
    try: return subprocess.check_output(['git','cat-file','-p',sha],text=True)
    except Exception: return None

idx=pd.read_parquet(ROOT/'draftkings_raw_index.parquet').copy()
idx['_ts']=pd.to_datetime(idx['snapshot_timestamp'],errors='coerce',utc=True)
idx=idx[(idx['_ts']>=START)&(idx['_ts']<=END)&idx['status'].astype(str).eq('success')&idx['raw_payload_path'].notna()].copy()
frames=[]; missing=[]; recovered=0
for _,r in idx.iterrows():
    path=str(r['raw_payload_path'])
    try:
        txt=_historical_text(path)
        if txt is None: missing.append(path); continue
        if not Path(path).exists(): recovered += 1
        payload=json.loads(txt)
        snap=DraftKingsSnapshot(str(r['snapshot_run_id']),str(r['snapshot_timestamp']),Path(path))
        reg={'subcategory_id':r.get('provider_subcategory_id'),'name':r.get('provider_subcategory_name'),'family':r.get('registry_family')}
        d=flatten_market_diagnostics(payload,snapshot=snap,request_url=r.get('request_url'),registry_entry=reg)
        if not d.empty: frames.append(d)
    except Exception as e: missing.append(f'{path}: {e}')
diag=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
can=normalize_draftkings_diagnostic_rows(diag) if not diag.empty else pd.DataFrame()
if not can.empty:
    can['event_start_dt']=pd.to_datetime(can['event_start_timestamp'],errors='coerce',utc=True)
    can=can[(can['event_start_dt']>=START)&(can['event_start_dt']<=END)].copy()
can.to_csv(OUT/'draftkings_jul_aug_2026_replayed_catalog.csv',index=False)
summary={'index_rows':int(len(idx)),'git_raw_paths':len(BLOBS),'historical_blobs_recovered':int(recovered),'missing_count':len(missing),'missing_payloads':missing[:50],'replayed_rows':int(len(can))}
if not can.empty:
    summary['events']=can.groupby(['provider_event_id','event_name','event_start_timestamp'],dropna=False).agg(rows=('provider_selection_id','size'),families=('market_family',lambda s: sorted(set(map(str,s.dropna()))))).reset_index().astype(str).to_dict(orient='records')
    summary['market_family_counts']=can['market_family'].value_counts(dropna=False).astype(int).to_dict()
    summary['market_key_counts']=can['market_key'].value_counts(dropna=False).astype(int).to_dict()
    summary['fighter_method_rows']=can[can['market_key'].astype(str).str.startswith('win_by_')][['snapshot_timestamp','provider_event_id','event_name','event_start_timestamp','market_key','fighter_name','american_odds','implied_probability','raw_payload_path']].astype(str).head(200).to_dict(orient='records')
rs=pd.read_parquet('data/fight_details/ufc_round_stats.parquet')
summary['round_stats_columns']=list(rs.columns)
for c in ['date','event_date','ufcstats_event_date']:
    if c in rs.columns:
        s=pd.to_datetime(rs[c],errors='coerce',utc=True); summary[f'round_stats_{c}_range']={'min':str(s.min()),'max':str(s.max())}
fv=pd.read_parquet('data/features/moneyline_feature_view.parquet')
summary['feature_columns']=list(fv.columns)
if 'date' in fv.columns:
    s=pd.to_datetime(fv['date'],errors='coerce',utc=True)
    summary['feature_date_range']={'min':str(s.min()),'max':str(s.max()),'jul_aug_rows':int(((s>=START)&(s<=END)).sum())}
    cols=[c for c in ['fight_id','date','event_name','r_name','b_name','method','winner','winner_side','result'] if c in fv.columns]
    summary['feature_jul_aug_preview']=fv.loc[(s>=START)&(s<=END),cols].astype(str).head(200).to_dict(orient='records')
(OUT/'draftkings_jul_aug_2026_replay_inventory_summary.json').write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
