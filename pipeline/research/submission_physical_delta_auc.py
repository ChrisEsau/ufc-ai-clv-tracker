#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.submission_stat_auc_screen import auc_binary, method_family

ALIASES={
 'name':['fighter_name','name','fighter','fighter_full_name'],
 'reach':['reach_in','reach_inches','reach','fighter_reach'],
 'height':['height_in','height_inches','height','fighter_height'],
 'dob':['dob','date_of_birth','birth_date','fighter_dob'],
 'age':['age','age_years','fighter_age'],
}
def pick(cols,names):
    m={c.lower():c for c in cols}
    return next((m[n] for n in names if n in m),None)
def inches(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float,np.number)): return float(x)
    s=str(x).strip().lower().replace('"','')
    try: return float(s)
    except: pass
    m=re.match(r"\s*(\d+)\s*['-]\s*(\d+)",s)
    if m: return 12*float(m.group(1))+float(m.group(2))
    m=re.match(r"\s*(\d+)\s*ft\s*(\d+)",s)
    if m: return 12*float(m.group(1))+float(m.group(2))
    return np.nan

def load_profiles(root:Path):
    rows=[]; schema=[]
    for p in root.rglob('*.parquet'):
        try:
            df=pd.read_parquet(p)
        except Exception: continue
        cols=list(df.columns); nc=pick(cols,ALIASES['name'])
        rc=pick(cols,ALIASES['reach']); hc=pick(cols,ALIASES['height']); dc=pick(cols,ALIASES['dob']); ac=pick(cols,ALIASES['age'])
        if nc and any([rc,hc,dc,ac]):
            schema.append({'path':str(p),'name':nc,'reach':rc,'height':hc,'dob':dc,'age':ac,'rows':len(df)})
            keep=pd.DataFrame({'fighter':df[nc].astype(str).str.strip()})
            keep['reach']=df[rc].map(inches) if rc else np.nan
            keep['height']=df[hc].map(inches) if hc else np.nan
            keep['dob']=pd.to_datetime(df[dc],errors='coerce') if dc else pd.NaT
            keep['age_static']=pd.to_numeric(df[ac],errors='coerce') if ac else np.nan
            rows.append(keep)
    if not rows: raise RuntimeError('No parquet with fighter name plus reach/height/dob/age found')
    x=pd.concat(rows,ignore_index=True); x['key']=x.fighter.str.lower()
    # Prefer non-null values, median for dimensions; earliest consistent DOB.
    prof=x.groupby('key').agg(fighter=('fighter','first'),reach=('reach','median'),height=('height','median'),dob=('dob','min'),age_static=('age_static','median')).reset_index()
    return prof,schema

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--holdout-from',default='2025-01-01'); ap.add_argument('--output-dir',type=Path,default=Path('data/diagnostics/submission_physical_delta_auc')); args=ap.parse_args()
    bouts=build_bouts(pd.read_parquet('data/master/ufc_master.parquet')).copy(); bouts['date']=pd.to_datetime(bouts.date); prof,schema=load_profiles(Path('data'))
    pr=prof.add_prefix('r_'); pb=prof.add_prefix('b_')
    bouts['r_key']=bouts.red_fighter.astype(str).str.strip().str.lower(); bouts['b_key']=bouts.blue_fighter.astype(str).str.strip().str.lower()
    bouts=bouts.merge(pr,left_on='r_key',right_on='r_key',how='left').merge(pb,left_on='b_key',right_on='b_key',how='left')
    for side in ['r','b']:
        dob=bouts[f'{side}_dob']; dyn=(bouts.date-dob).dt.days/365.2425
        static=bouts[f'{side}_age_static']; bouts[f'{side}_age']=dyn.where(dyn.notna(),static)
    bouts['age_delta']=bouts.r_age-bouts.b_age
    bouts['reach_delta']=bouts.r_reach-bouts.b_reach
    bouts['height_delta']=bouts.r_height-bouts.b_height
    bouts['abs_age_delta']=bouts.age_delta.abs(); bouts['abs_reach_delta']=bouts.reach_delta.abs(); bouts['abs_height_delta']=bouts.height_delta.abs()
    m=bouts.method.map(method_family); bouts['y_fight_sub']=(m=='SUB').astype(int)
    bouts['y_r_sub']=((m=='SUB')&(bouts.winner==bouts.red_fighter)).astype(int)
    bouts['y_b_sub']=((m=='SUB')&(bouts.winner==bouts.blue_fighter)).astype(int)
    cut=pd.Timestamp(args.holdout_from); h=bouts[bouts.date>=cut].copy(); out=[]
    for feat in ['age_delta','reach_delta','height_delta','abs_age_delta','abs_reach_delta','abs_height_delta']:
        # fight-level SUB
        auc,n,n1,n0=auc_binary(h.y_fight_sub,h[feat]); out.append({'target':'fight_sub','feature':feat,'auc':auc,'discrimination_auc':max(auc,1-auc) if np.isfinite(auc) else np.nan,'n':n,'positives':n1})
        # side-specific: orient delta from candidate fighter perspective
        vals=pd.concat([h[feat if feat.startswith('abs_') else feat], (-h[feat] if not feat.startswith('abs_') else h[feat])],ignore_index=True)
        y=pd.concat([h.y_r_sub,h.y_b_sub],ignore_index=True)
        auc,n,n1,n0=auc_binary(y,vals); out.append({'target':'side_sub_win','feature':feat,'auc':auc,'discrimination_auc':max(auc,1-auc) if np.isfinite(auc) else np.nan,'n':n,'positives':n1})
    res=pd.DataFrame(out).sort_values(['target','discrimination_auc'],ascending=[True,False])
    args.output_dir.mkdir(parents=True,exist_ok=True); res.to_csv(args.output_dir/'physical_delta_auc.csv',index=False)
    bouts[['date','bout_id','red_fighter','blue_fighter','y_fight_sub','y_r_sub','y_b_sub','age_delta','reach_delta','height_delta']].to_csv(args.output_dir/'holdout_physical_features.csv',index=False)
    with open(args.output_dir/'schema_sources.json','w') as f: json.dump(schema,f,indent=2,default=str)
    print(res.to_string(index=False)); print('\ncoverage', {k:int(h[k].notna().sum()) for k in ['age_delta','reach_delta','height_delta']})
if __name__=='__main__': main()
