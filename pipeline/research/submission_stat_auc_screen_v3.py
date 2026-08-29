#!/usr/bin/env python3
"""Leakage-safe submission stat AUC screen for long-format UFC round stats.
Research-only; no Brain, FSR, or market inputs.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.research.prefight_strength_elo import build_bouts

STAT_COLS = [
    'kd','sig_str_landed','sig_str_attempted','total_str_landed','total_str_attempted',
    'td_landed','td_attempted','sub_att','rev','ctrl_sec','head_landed','head_attempted',
    'body_landed','body_attempted','leg_landed','leg_attempted','distance_landed',
    'distance_attempted','clinch_landed','clinch_attempted','ground_landed','ground_attempted'
]

def norm_name(x): return re.sub(r'\s+',' ',str(x or '').strip().lower())
def norm_date(x): return pd.to_datetime(x, errors='coerce').normalize()
def method_family(m):
    s=str(m or '').lower()
    if 'submission' in s or re.search(r'\bsub\b',s): return 'SUB'
    if 'decision' in s: return 'DEC'
    if 'ko' in s or 'tko' in s: return 'KO'
    return None

def auc_binary(y,score):
    y=np.asarray(y); s=np.asarray(score,float); mask=np.isfinite(s)&np.isfinite(y)
    y=y[mask].astype(int); s=s[mask]; n1=int((y==1).sum()); n0=int((y==0).sum())
    if not n1 or not n0: return np.nan,len(y),n1,n0
    r=pd.Series(s).rank(method='average').to_numpy(float)
    auc=(r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
    return float(auc),len(y),n1,n0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--round-stats',type=Path,default=Path('data/fight_details/ufc_round_stats.parquet'))
    ap.add_argument('--master',type=Path,default=Path('data/master/ufc_master.parquet'))
    ap.add_argument('--holdout-from',default='2025-01-01')
    ap.add_argument('--min-prior-fights',type=int,default=1)
    ap.add_argument('--output-dir',type=Path,default=Path('data/diagnostics/submission_stat_auc_screen'))
    args=ap.parse_args()
    raw=pd.read_parquet(args.round_stats).copy(); bouts=build_bouts(pd.read_parquet(args.master)).copy()
    required={'fight_id','event_date','corner','fighter_name','opponent_name'}
    miss=required-set(raw.columns)
    if miss: raise RuntimeError(f'missing required long-format columns: {sorted(miss)}')
    stats=[c for c in STAT_COLS if c in raw.columns]
    if not stats: raise RuntimeError('no known numeric stat columns found')
    for c in stats: raw[c]=pd.to_numeric(raw[c],errors='coerce')
    raw['corner_norm']=raw.corner.astype(str).str.strip().str.lower()

    # Aggregate all recorded rounds to fighter-fight totals, then pivot by corner.
    side=(raw.groupby(['fight_id','event_date','corner_norm','fighter_name','opponent_name'],as_index=False)[stats]
            .sum(min_count=1))
    # one red and one blue row per fight where available
    reds=side[side.corner_norm.str.startswith('r')].copy(); blues=side[side.corner_norm.str.startswith('b')].copy()
    reds=reds.drop_duplicates('fight_id'); blues=blues.drop_duplicates('fight_id')
    keep_meta=['fight_id','event_date','fighter_name','opponent_name']
    r=reds[keep_meta+stats].rename(columns={'fighter_name':'raw_red','opponent_name':'raw_blue',**{c:'R__'+c for c in stats}})
    b=blues[['fight_id']+stats].rename(columns={c:'B__'+c for c in stats})
    fs=r.merge(b,on='fight_id',how='inner')
    if len(fs)<100: raise RuntimeError(f'only {len(fs)} fights had both red and blue round-stat rows')

    # Align to master. Prefer stable fight_id == bout_id; otherwise use date + unordered names.
    bouts['bout_id_str']=bouts.bout_id.astype(str); fs['fight_id_str']=fs.fight_id.astype(str)
    idmatch=bouts.merge(fs.drop(columns=['fight_id']),left_on='bout_id_str',right_on='fight_id_str',how='left')
    statcols=['R__'+c for c in stats]+['B__'+c for c in stats]
    id_count=int(idmatch[statcols].notna().any(axis=1).sum())
    if id_count>=100:
        aligned=idmatch; align_mode='fight_id_to_bout_id'; matched=id_count
    else:
        fs['_date']=fs.event_date.map(norm_date); fs['_a']=fs.raw_red.map(norm_name); fs['_b']=fs.raw_blue.map(norm_name)
        fs['_n1']=np.minimum(fs._a,fs._b); fs['_n2']=np.maximum(fs._a,fs._b)
        fs['_key']=fs._date.astype(str)+'|'+fs._n1+'|'+fs._n2
        fs2=fs.sort_values('fight_id_str').drop_duplicates('_key')
        bouts['_date']=bouts.date.map(norm_date); bouts['_a']=bouts.red_fighter.map(norm_name); bouts['_b']=bouts.blue_fighter.map(norm_name)
        bouts['_n1']=np.minimum(bouts._a,bouts._b); bouts['_n2']=np.maximum(bouts._a,bouts._b)
        bouts['_key']=bouts._date.astype(str)+'|'+bouts._n1+'|'+bouts._n2
        aligned=bouts.merge(fs2[['_key','raw_red']+statcols],on='_key',how='left')
        matched=int(aligned[statcols].notna().any(axis=1).sum()); align_mode='date_unordered_names'
        if matched<100: raise RuntimeError(f'alignment matched only {matched} fights (id direct={id_count})')
        # if raw red orientation is reversed vs master, swap R/B stats per row
        flip=aligned.raw_red.notna() & (aligned.raw_red.map(norm_name)!=aligned.red_fighter.map(norm_name))
        for c in stats:
            rc,bc='R__'+c,'B__'+c
            tmp=aligned.loc[flip,rc].copy(); aligned.loc[flip,rc]=aligned.loc[flip,bc].values; aligned.loc[flip,bc]=tmp.values

    aligned=aligned.sort_values(['date','bout_id']).reset_index(drop=True)
    career=defaultdict(lambda:defaultdict(lambda:[0.0,0.0,0]))
    side_rows=[]; fight_rows=[]
    for x in aligned.itertuples(index=False):
        rf,bf=x.red_fighter,x.blue_fighter; am=method_family(getattr(x,'method',''))
        cache={'R':{},'B':{}}
        for st in stats:
            for side_name,fig,opp in [('R',rf,bf),('B',bf,rf)]:
                cf=career[fig][st]; co=career[opp][st]
                own=cf[0]/cf[2] if cf[2]>=args.min_prior_fights else np.nan
                allowed=co[1]/co[2] if co[2]>=args.min_prior_fights else np.nan
                cache[side_name][st]=(own,allowed)
        for side_name,fig,opp,y in [('R',rf,bf,int(am=='SUB' and x.winner==rf)),('B',bf,rf,int(am=='SUB' and x.winner==bf))]:
            row={'date':x.date,'bout_id':x.bout_id,'side':side_name,'fighter':fig,'opponent':opp,'y_sub_win':y}
            for st in stats:
                own,allow=cache[side_name][st]
                row[st+'__own']=own; row[st+'__opp_allowed']=allow
                row[st+'__matchup_mean']=np.nanmean([own,allow]) if np.isfinite(own) or np.isfinite(allow) else np.nan
                row[st+'__matchup_sum']=own+allow if np.isfinite(own) and np.isfinite(allow) else np.nan
                row[st+'__matchup_diff']=own-allow if np.isfinite(own) and np.isfinite(allow) else np.nan
            side_rows.append(row)
        fr={'date':x.date,'bout_id':x.bout_id,'y_fight_sub':int(am=='SUB')}
        for st in stats:
            vals=[]
            for side_name in ['R','B']:
                own,allow=cache[side_name][st]; vals.append(np.nanmean([own,allow]) if np.isfinite(own) or np.isfinite(allow) else np.nan)
            fin=[v for v in vals if np.isfinite(v)]
            fr[st+'__fight_mean']=float(np.mean(fin)) if fin else np.nan
            fr[st+'__fight_max']=float(np.max(fin)) if fin else np.nan
            fr[st+'__fight_sum']=float(np.sum(fin)) if len(fin)==2 else np.nan
        fight_rows.append(fr)
        for st in stats:
            rv=getattr(x,'R__'+st,np.nan); bv=getattr(x,'B__'+st,np.nan)
            if np.isfinite(rv) and np.isfinite(bv):
                cr=career[rf][st]; cb=career[bf][st]
                cr[0]+=float(rv); cr[1]+=float(bv); cr[2]+=1
                cb[0]+=float(bv); cb[1]+=float(rv); cb[2]+=1

    sides=pd.DataFrame(side_rows); fights=pd.DataFrame(fight_rows); cutoff=pd.Timestamp(args.holdout_from); out=[]
    def screen(df,ycol,scope):
        skip={'date','bout_id','side','fighter','opponent',ycol}
        for col in df.columns:
            if col in skip or not pd.api.types.is_numeric_dtype(df[col]): continue
            st,transform=col.split('__',1)
            for split,mask in [('train',df.date<cutoff),('holdout',df.date>=cutoff)]:
                auc,n,n1,n0=auc_binary(df.loc[mask,ycol],df.loc[mask,col])
                if np.isfinite(auc): out.append({'scope':scope,'feature':st,'transform':transform,'split':split,'auc':auc,'discrimination_auc':max(auc,1-auc),'direction':1 if auc>=.5 else -1,'n':n,'positives':n1,'negatives':n0})
    screen(sides,'y_sub_win','side_sub_win'); screen(fights,'y_fight_sub','fight_sub')
    res=pd.DataFrame(out); hold=res[(res.split=='holdout')&(res.n>=100)].copy()
    idx=hold.groupby(['scope','feature'])['discrimination_auc'].idxmax(); best=hold.loc[idx].sort_values(['scope','discrimination_auc'],ascending=[True,False]).reset_index(drop=True)
    train=res[res.split=='train'].set_index(['scope','feature','transform'])
    best['train_auc_same_transform']=[train.loc[(q.scope,q.feature,q.transform),'auc'] if (q.scope,q.feature,q.transform) in train.index else np.nan for q in best.itertuples(index=False)]
    best['train_discrimination_same_transform']=best.train_auc_same_transform.map(lambda z:max(z,1-z) if np.isfinite(z) else np.nan)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    res.to_csv(args.output_dir/'all_auc_results.csv',index=False); best.to_csv(args.output_dir/'best_stat_auc_rankings.csv',index=False)
    sides.to_csv(args.output_dir/'side_prefight_features.csv',index=False); fights.to_csv(args.output_dir/'fight_prefight_features.csv',index=False)
    with open(args.output_dir/'schema.json','w') as f: json.dump({'align_mode':align_mode,'matched_fights':matched,'direct_id_matches':id_count,'stats':stats,'raw_columns':list(raw.columns)},f,indent=2)
    print('alignment:',align_mode,'matched:',matched,'direct_id_matches:',id_count)
    for scope in ['fight_sub','side_sub_win']:
        print('\nTOP',scope.upper()); print(best[best.scope==scope].head(40).to_string(index=False))
if __name__=='__main__': main()
