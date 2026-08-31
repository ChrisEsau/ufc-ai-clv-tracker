from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.research import run_hierarchical_v5_market_intelligence as base

_orig_build_rows = base.method._build_rows

def frozen_build_rows(*args, **kwargs):
    df, features, extra = _orig_build_rows(*args, **kwargs)
    df = df[pd.to_datetime(df['date'], errors='coerce') <= pd.Timestamp('2024-12-31')].copy()
    return df, features, extra

base.method._build_rows = frozen_build_rows


def pair_key(a,b):
    x=sorted([base.norm_name(a),base.norm_name(b)])
    return '||'.join(x) if all(x) else ''


def build_score_rows(chosen, fv):
    fv=fv.copy()
    fv['fight_id']=fv['fight_id'].astype(str)
    # Canonical feature-view orientation.
    red_col='r_name' if 'r_name' in fv.columns else 'r_name_x'
    blue_col='b_name' if 'b_name' in fv.columns else 'b_name_x'
    fv['_pair']=fv.apply(lambda r: pair_key(r.get(red_col),r.get(blue_col)),axis=1)
    fv['_event_norm']=fv['event_name'].astype(str).str.lower().str.replace(r'[^a-z0-9]+','',regex=True)

    fmeta=chosen.sort_values('refresh_timestamp').groupby('fight_id',as_index=False).last()[['fight_id','event_name','fight_display','refresh_timestamp']]
    fmeta['market_fight_id']=fmeta['fight_id'].astype(str)
    pairs=fmeta['fight_display'].apply(base.split_display)
    fmeta['_a']=[x[0] for x in pairs]; fmeta['_b']=[x[1] for x in pairs]
    fmeta['_pair']=fmeta.apply(lambda r: pair_key(r['_a'],r['_b']),axis=1)
    fmeta['_event_norm']=fmeta['event_name'].astype(str).str.lower().str.replace(r'[^a-z0-9]+','',regex=True)

    # First exact event+pair; fallback unique pair match.
    exact=fmeta.merge(fv,on=['_pair','_event_norm'],how='left',suffixes=('_market',''))
    rows=[]; skips=[]
    for _,m in fmeta.iterrows():
        cand=exact[exact['market_fight_id'].eq(m['market_fight_id'])].copy()
        cand=cand[cand['fight_id'].notna()]
        if len(cand)!=1:
            pc=fv[fv['_pair'].eq(m['_pair'])]
            if len(pc)==1: cand=pc.copy()
        if len(cand)!=1:
            skips.append({'market_fight_id':m['market_fight_id'],'event_name':m['event_name'],'fight_display':m['fight_display'],'reason':'feature_pair_match_count','candidate_count':int(len(cand))})
            continue
        r=cand.iloc[0]
        canonical_fid=str(r['fight_id'])
        red_name=str(r[red_col]); blue_name=str(r[blue_col])
        z=chosen[chosen['fight_id'].astype(str).eq(str(m['market_fight_id']))].copy()
        raw={}; ok=True
        for mk,suffix in [('moneyline','ml'),('win_by_ko_tko_dq','ko'),('win_by_submission','sub'),('win_by_decision','dec')]:
            zz=z[z['market_key'].eq(mk)]
            vals={'red':[],'blue':[]}
            for _,rr in zz.iterrows():
                side=base.classify_side(rr,red_name,blue_name)
                if side: vals[side].append(rr)
            if not vals['red'] or not vals['blue']:
                ok=False
                skips.append({'market_fight_id':m['market_fight_id'],'canonical_fight_id':canonical_fid,'event_name':m['event_name'],'fight_display':m['fight_display'],'reason':f'cannot_map_{mk}','red_fighter':red_name,'blue_fighter':blue_name})
                break
            for side in ['red','blue']:
                rr=vals[side][-1]
                raw[f'{side}_{suffix}_raw_p']=float(rr['implied_probability'])
                raw[f'{side}_{suffix}_american']=float(rr['american_odds']) if pd.notna(rr['american_odds']) else np.nan
        if not ok: continue
        ml_sum=raw['red_ml_raw_p']+raw['blue_ml_raw_p']
        raw['market_overround']=ml_sum
        raw['market_p_red_ml']=raw['red_ml_raw_p']/ml_sum
        mr=np.array([raw['red_ko_raw_p'],raw['red_sub_raw_p'],raw['red_dec_raw_p'],raw['blue_ko_raw_p'],raw['blue_sub_raw_p'],raw['blue_dec_raw_p']],float)
        mn=mr/mr.sum()
        out={'fight_id':canonical_fid,'market_fight_id':str(m['market_fight_id']),'event_name':m['event_name'],'fight_display':m['fight_display'],'refresh_timestamp':m['refresh_timestamp'],'red_fighter':red_name,'blue_fighter':blue_name,**raw}
        for j,slug in enumerate(base.SLUGS): out[f'market_{slug}']=float(mn[j])
        for c in fv.columns:
            if c not in ['fight_id','_pair','_event_norm'] and c in r.index: out[c]=r[c]
        rows.append(out)
    return pd.DataFrame(rows),pd.DataFrame(skips),red_col,blue_col

base.build_score_rows=build_score_rows
base.main()
