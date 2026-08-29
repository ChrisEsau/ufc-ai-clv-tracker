#!/usr/bin/env python3
"""Leakage-safe submission stat AUC screen using composite fight keys.
Research-only; no Brain, FSR, or market inputs.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.submission_stat_auc_screen import (
    discover_pairs, derive_pair_features, method_family, auc_binary
)


def pick_col(df, names):
    lower={c.lower():c for c in df.columns}
    for n in names:
        if n.lower() in lower: return lower[n.lower()]
    return None

def norm_name(x): return str(x).strip().lower()
def norm_date(x): return pd.to_datetime(x, errors='coerce').normalize()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--round-stats',type=Path,default=Path('data/fight_details/ufc_round_stats.parquet'))
    ap.add_argument('--master',type=Path,default=Path('data/master/ufc_master.parquet'))
    ap.add_argument('--holdout-from',default='2025-01-01')
    ap.add_argument('--min-prior-fights',type=int,default=1)
    ap.add_argument('--output-dir',type=Path,default=Path('data/diagnostics/submission_stat_auc_screen'))
    args=ap.parse_args()
    raw=pd.read_parquet(args.round_stats)
    bouts=build_bouts(pd.read_parquet(args.master)).copy()

    dcol=pick_col(raw,['date','event_date','fight_date'])
    rcol=pick_col(raw,['r_name','red_fighter','r_fighter','red_name'])
    bcol=pick_col(raw,['b_name','blue_fighter','b_fighter','blue_name'])
    if not all([dcol,rcol,bcol]):
        raise RuntimeError(f'Cannot identify composite fight key columns. columns={list(raw.columns)}')

    raw=raw.copy(); raw['_date']=raw[dcol].map(norm_date); raw['_r']=raw[rcol].map(norm_name); raw['_b']=raw[bcol].map(norm_name)
    raw['_fight_key']=raw['_date'].astype(str)+'|'+raw['_r']+'|'+raw['_b']
    bouts['_date']=bouts.date.map(norm_date); bouts['_r']=bouts.red_fighter.map(norm_name); bouts['_b']=bouts.blue_fighter.map(norm_name)
    bouts['_fight_key']=bouts['_date'].astype(str)+'|'+bouts['_r']+'|'+bouts['_b']

    pairs=discover_pairs(raw)
    if not pairs: raise RuntimeError(f'No red/blue stat pairs discovered. columns={list(raw.columns)}')
    derived={'_fight_key':raw['_fight_key']}; feature_names=[]; source_map={}
    for base,rc,bc in pairs:
        for fname,rs,bs in derive_pair_features(raw,base,rc,bc):
            derived['R__'+fname]=rs.astype(float); derived['B__'+fname]=bs.astype(float)
            feature_names.append(fname); source_map[fname]={'red_column':rc,'blue_column':bc}
    if not feature_names: raise RuntimeError('No parseable paired numeric stats')
    d=pd.DataFrame(derived)
    agg=[c for c in d.columns if c!='_fight_key']
    bs=d.groupby('_fight_key',as_index=False)[agg].sum(min_count=1)
    bouts=bouts.merge(bs,on='_fight_key',how='left').sort_values(['date','bout_id']).reset_index(drop=True)
    matched=int(bouts[agg].notna().any(axis=1).sum())
    if matched < 100: raise RuntimeError(f'Composite key matched only {matched} fights; refusing weak alignment')

    career=defaultdict(lambda:defaultdict(lambda:[0.0,0.0,0])); side_rows=[]; fight_rows=[]
    for b in bouts.itertuples(index=False):
        r,bl=b.red_fighter,b.blue_fighter; am=method_family(getattr(b,'method',''))
        rsub=int(am=='SUB' and b.winner==r); bsub=int(am=='SUB' and b.winner==bl); fsub=int(am=='SUB')
        cache={'R':{},'B':{}}
        for f in feature_names:
            for side,fig,opp in [('R',r,bl),('B',bl,r)]:
                cf=career[fig][f]; co=career[opp][f]
                own=cf[0]/cf[2] if cf[2]>=args.min_prior_fights else np.nan
                allowed=co[1]/co[2] if co[2]>=args.min_prior_fights else np.nan
                cache[side][f]=(own,allowed)
        for side,fig,opp,y in [('R',r,bl,rsub),('B',bl,r,bsub)]:
            row={'date':b.date,'bout_id':b.bout_id,'side':side,'fighter':fig,'opponent':opp,'y_sub_win':y}
            for f in feature_names:
                own,allow=cache[side][f]
                row[f+'__own']=own; row[f+'__opp_allowed']=allow
                row[f+'__matchup_mean']=np.nanmean([own,allow]) if np.isfinite(own) or np.isfinite(allow) else np.nan
                row[f+'__matchup_sum']=own+allow if np.isfinite(own) and np.isfinite(allow) else np.nan
                row[f+'__matchup_diff']=own-allow if np.isfinite(own) and np.isfinite(allow) else np.nan
            side_rows.append(row)
        fr={'date':b.date,'bout_id':b.bout_id,'y_fight_sub':fsub}
        for f in feature_names:
            vals=[]
            for side in ['R','B']:
                own,allow=cache[side][f]; vals.append(np.nanmean([own,allow]) if np.isfinite(own) or np.isfinite(allow) else np.nan)
            finite=[x for x in vals if np.isfinite(x)]
            fr[f+'__fight_mean']=float(np.mean(finite)) if finite else np.nan
            fr[f+'__fight_max']=float(np.max(finite)) if finite else np.nan
            fr[f+'__fight_sum']=float(np.sum(finite)) if len(finite)==2 else np.nan
        fight_rows.append(fr)
        for f in feature_names:
            rv=getattr(b,'R__'+f,np.nan); bv=getattr(b,'B__'+f,np.nan)
            if np.isfinite(rv) and np.isfinite(bv):
                cr=career[r][f]; cb=career[bl][f]
                cr[0]+=float(rv); cr[1]+=float(bv); cr[2]+=1
                cb[0]+=float(bv); cb[1]+=float(rv); cb[2]+=1

    sides=pd.DataFrame(side_rows); fights=pd.DataFrame(fight_rows); cutoff=pd.Timestamp(args.holdout_from); results=[]
    def screen(frame,ycol,scope):
        skip={'date','bout_id','side','fighter','opponent',ycol}
        for col in frame.columns:
            if col in skip or not pd.api.types.is_numeric_dtype(frame[col]): continue
            parts=col.split('__'); fname=parts[0]; transform='__'.join(parts[1:])
            for split,mask in [('train',frame.date<cutoff),('holdout',frame.date>=cutoff)]:
                auc,n,n1,n0=auc_binary(frame.loc[mask,ycol].to_numpy(),frame.loc[mask,col].to_numpy())
                if np.isfinite(auc): results.append({'scope':scope,'feature':fname,'transform':transform,'split':split,'auc':auc,'discrimination_auc':max(auc,1-auc),'direction':1 if auc>=.5 else -1,'n':n,'positives':n1,'negatives':n0})
    screen(sides,'y_sub_win','side_sub_win'); screen(fights,'y_fight_sub','fight_sub')
    res=pd.DataFrame(results); hold=res[(res.split=='holdout')&(res.n>=100)].copy()
    idx=hold.groupby(['scope','feature'])['discrimination_auc'].idxmax(); best=hold.loc[idx].sort_values(['scope','discrimination_auc'],ascending=[True,False]).reset_index(drop=True)
    tr=res[res.split=='train'].set_index(['scope','feature','transform'])
    best['train_auc_same_transform']=[tr.loc[(x.scope,x.feature,x.transform),'auc'] if (x.scope,x.feature,x.transform) in tr.index else np.nan for x in best.itertuples(index=False)]
    best['train_discrimination_same_transform']=best.train_auc_same_transform.map(lambda x:max(x,1-x) if np.isfinite(x) else np.nan)
    best['source_red_column']=best.feature.map(lambda x:source_map.get(x,{}).get('red_column')); best['source_blue_column']=best.feature.map(lambda x:source_map.get(x,{}).get('blue_column'))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    res.to_csv(args.output_dir/'all_auc_results.csv',index=False); best.to_csv(args.output_dir/'best_stat_auc_rankings.csv',index=False)
    sides.to_csv(args.output_dir/'side_prefight_features.csv',index=False); fights.to_csv(args.output_dir/'fight_prefight_features.csv',index=False)
    with open(args.output_dir/'schema.json','w') as f: json.dump({'round_stats_columns':list(raw.columns),'key_columns':[dcol,rcol,bcol],'matched_fights':matched,'discovered_pairs':pairs,'derived_features':feature_names,'source_map':source_map},f,indent=2)
    print('matched fights:',matched,'derived features:',len(feature_names))
    for scope in ['fight_sub','side_sub_win']:
        print('\nTOP',scope.upper()); print(best[best.scope==scope].head(40).to_string(index=False))
if __name__=='__main__': main()
