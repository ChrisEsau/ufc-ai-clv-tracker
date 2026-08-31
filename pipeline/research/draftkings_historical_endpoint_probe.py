from pathlib import Path
import json
import pandas as pd
import requests

idx=pd.read_parquet('data/market/draftkings_raw_index.parquet').copy()
idx['ts']=pd.to_datetime(idx['snapshot_timestamp'],errors='coerce',utc=True)
idx=idx[(idx['ts']>=pd.Timestamp('2026-07-01',tz='UTC'))&(idx['ts']<=pd.Timestamp('2026-08-31 23:59:59',tz='UTC'))&idx['status'].astype(str).eq('success')].copy()
# Probe distinct historical events, prioritizing fighter-method and main-line endpoints.
rows=[]
seen=set()
for _,r in idx.sort_values('ts').iterrows():
    key=(str(r.get('provider_event_id')),str(r.get('registry_family')))
    if key in seen: continue
    fam=str(r.get('registry_family'))
    if fam not in {'fighter_method_props','main_lines','exact_method'}: continue
    seen.add(key)
    u=str(r['request_url'])
    try:
        resp=requests.get(u,timeout=20,headers={'User-Agent':'Mozilla/5.0','Accept':'application/json,text/plain,*/*'})
        rec={'event_id':str(r.get('provider_event_id')),'family':fam,'snapshot_ts':str(r.get('snapshot_timestamp')),'url':u,'status':resp.status_code,'content_type':resp.headers.get('content-type'),'bytes':len(resp.content),'text_prefix':resp.text[:1000]}
        try:
            j=resp.json(); rec['json_top_keys']=list(j.keys()) if isinstance(j,dict) else [type(j).__name__]
            rec['json_preview']=str(j)[:2500]
        except Exception: pass
        rows.append(rec)
    except Exception as e:
        rows.append({'event_id':str(r.get('provider_event_id')),'family':fam,'url':u,'error':repr(e)})
    if len(rows)>=12: break
out=Path('data/research/prop_mispricing/draftkings_historical_endpoint_probe.json')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
