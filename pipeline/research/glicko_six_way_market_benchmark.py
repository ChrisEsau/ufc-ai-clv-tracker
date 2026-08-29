#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np
import pandas as pd

SIX=['R_KO','R_SUB','R_DEC','B_KO','B_SUB','B_DEC']
PCOLS=['p_red_ko','p_red_sub','p_red_dec','p_blue_ko','p_blue_sub','p_blue_dec']
METHOD_MAP={'win_by_ko_tko_dq':'KO','win_by_submission':'SUB','win_by_decision':'DEC'}
EPS=1e-12


def metrics(df,prefix):
    P=df[[f'{prefix}_{c}' for c in SIX]].to_numpy(float)
    idx=np.array([SIX.index(x) for x in df.actual_six])
    Y=np.zeros_like(P); Y[np.arange(len(df)),idx]=1
    pt=np.clip(P[np.arange(len(df)),idx],EPS,1)
    top=np.argmax(P,axis=1)
    M=np.c_[P[:,0]+P[:,3],P[:,1]+P[:,4],P[:,2]+P[:,5]]
    mlabs=['KO','SUB','DEC']; midx=np.array([mlabs.index(x) for x in df.actual_method])
    mp=np.clip(M[np.arange(len(df)),midx],EPS,1)
    W=np.c_[P[:,:3].sum(1),P[:,3:].sum(1)]
    widx=np.array([0 if x.startswith('R_') else 1 for x in df.actual_six])
    wp=np.clip(W[np.arange(len(df)),widx],EPS,1)
    WY=np.zeros_like(W); WY[np.arange(len(df)),widx]=1
    return {
      'n':int(len(df)),
      'six_way_accuracy':float(np.mean(top==idx)),
      'six_way_log_loss':float(-np.mean(np.log(pt))),
      'six_way_brier':float(np.mean(np.sum((P-Y)**2,axis=1))),
      'mean_probability_actual_outcome':float(np.mean(pt)),
      'method_accuracy':float(np.mean(np.argmax(M,1)==midx)),
      'method_log_loss':float(-np.mean(np.log(mp))),
      'winner_accuracy':float(np.mean(np.argmax(W,1)==widx)),
      'winner_log_loss':float(-np.mean(np.log(wp))),
      'winner_brier':float(np.mean(np.sum((W-WY)**2,axis=1))),
      'actual_method_shares':{m:float(np.mean(df.actual_method==m)) for m in mlabs},
      'predicted_method_shares':{m:float(M[:,i].mean()) for i,m in enumerate(mlabs)},
    }


def main():
    gpath=Path('data/diagnostics/standalone_glicko_six_way_enhanced_sub/holdout_predictions.csv')
    mpath=Path('/tmp/historical_market_outcomes.parquet')
    g=pd.read_csv(gpath)
    m=pd.read_parquet(mpath)
    m=m[m.market_key.isin(METHOD_MAP)].copy()
    m=m[m.outcome_side.isin(['red','blue'])].copy()
    m['method']=m.market_key.map(METHOD_MAP)
    m['cls']=m.outcome_side.map({'red':'R','blue':'B'})+'_'+m['method']
    m['implied_probability']=pd.to_numeric(m.implied_probability,errors='coerce')
    m=m.dropna(subset=['fight_id','cls','implied_probability'])
    # One historical consensus price per class/fight is expected; de-duplicate defensively.
    m=m.sort_values(['fight_id','cls']).drop_duplicates(['fight_id','cls'],keep='last')
    counts=m.groupby('fight_id').cls.nunique()
    complete=set(counts[counts==6].index)
    m=m[m.fight_id.isin(complete)]
    wide=m.pivot(index='fight_id',columns='cls',values='implied_probability')
    wide=wide[SIX].dropna()
    # six-way no-vig normalization
    wide=wide.div(wide.sum(axis=1),axis=0)
    wide.columns=[f'market_{c}' for c in SIX]
    wide=wide.reset_index()

    key='bout_id' if 'bout_id' in g.columns else ('fight_id' if 'fight_id' in g.columns else None)
    if key is None: raise RuntimeError(f'No fight key in Glicko predictions: {list(g.columns)}')
    if key != 'fight_id': g=g.rename(columns={key:'fight_id'})
    rename={p:f'glicko_{s}' for p,s in zip(PCOLS,SIX)}
    missing=[p for p in PCOLS if p not in g.columns]
    if missing: raise RuntimeError(f'Missing Glicko probability columns {missing}; cols={list(g.columns)}')
    g=g.rename(columns=rename)
    d=g.merge(wide,on='fight_id',how='inner')
    d=d[d.actual_six.notna()].copy()
    if len(d)==0: raise RuntimeError('No matched complete six-way market fights')

    outdir=Path('data/diagnostics/glicko_six_way_market_benchmark'); outdir.mkdir(parents=True,exist_ok=True)
    d.to_csv(outdir/'matched_predictions.csv',index=False)
    summary={
      'glicko_holdout_rows':int(len(g)),
      'market_complete_six_way_fights':int(len(wide)),
      'matched_fights':int(len(d)),
      'market_probability_method':'normalize six raw implied probabilities to sum 1',
      'glicko':metrics(d,'glicko'),
      'market_no_vig':metrics(d,'market'),
    }
    pd.DataFrame([{'model':'Enhanced Glicko-6',**summary['glicko']},{'model':'Market no-vig',**summary['market_no_vig']}]).to_csv(outdir/'comparison.csv',index=False)
    with open(outdir/'summary.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
