#!/usr/bin/env python3
"""Evaluate the original graded-outcome standalone Glicko winner model on the exact
complete six-way historical-market cohort used by the Glicko-6 benchmark.
Research only; market is evaluation/matching only, never a model input.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.prefight_strength_glicko_ablation import run as run_glicko

EPS=1e-12
METHOD_KEYS={'win_by_ko_tko_dq','win_by_submission','win_by_decision'}


def bin_metrics(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return {
        'n':int(len(y)),
        'accuracy':float(np.mean((p>0.5)==(y>0.5))),
        'brier':float(np.mean((p-y)**2)),
        'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
        'mean_probability_actual_winner':float(np.mean(np.where(y==1,p,1-p))),
    }


def main():
    bouts=build_bouts(pd.read_parquet('data/master/ufc_master.parquet'))
    g=run_glicko(bouts,graded=True,inflation=4.0)
    g['date']=pd.to_datetime(g.date)
    g=g[(g.date>=pd.Timestamp('2025-01-01')) & g.winner.notna()].copy()

    m=pd.read_parquet('/tmp/historical_market_outcomes.parquet')
    m=m[m.market_key.isin(METHOD_KEYS) & m.outcome_side.isin(['red','blue'])].copy()
    counts=(m.dropna(subset=['fight_id']).groupby('fight_id')
              .apply(lambda x: x[['market_key','outcome_side']].drop_duplicates().shape[0],include_groups=False))
    complete=set(counts[counts==6].index)

    # Match the exact 490 cohort definition used by the previous six-way benchmark:
    # complete six-way market + Glicko holdout fight ID + actual six-way outcome available.
    # Standard UFC master IDs and historical market fight_id are the same key.
    d=g[g.bout_id.isin(complete)].copy()
    d=d.rename(columns={'bout_id':'fight_id'})

    # Market winner probability from the six method prices, normalized across all six.
    mm=m[m.fight_id.isin(set(d.fight_id))].copy()
    mm['ip']=pd.to_numeric(mm.implied_probability,errors='coerce')
    mm=mm.dropna(subset=['ip']).sort_values(['fight_id','market_key','outcome_side']).drop_duplicates(['fight_id','market_key','outcome_side'],keep='last')
    side=mm.groupby(['fight_id','outcome_side']).ip.sum().unstack()
    side=side.dropna(subset=['red','blue'])
    denom=side.red+side.blue
    side['market_p_red']=side.red/denom
    d=d.merge(side[['market_p_red']],left_on='fight_id',right_index=True,how='inner')

    y=(d.winner==d.red_fighter).astype(float).to_numpy()
    summary={
        'cohort':'exact complete-six-way historical market overlap, 2025+ holdout',
        'standalone_glicko':'graded outcomes + RD inactivity inflation 4.0/day after 180d grace',
        'matched_fights':int(len(d)),
        'glicko_winner':bin_metrics(y,d.p_red.to_numpy()),
        'market_winner_from_six_way_no_vig':bin_metrics(y,d.market_p_red.to_numpy()),
    }
    out=Path('data/diagnostics/glicko_winner_market490'); out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/'matched_winner_predictions.csv',index=False)
    pd.DataFrame([
        {'model':'Standalone graded Glicko',**summary['glicko_winner']},
        {'model':'Market no-vig winner from six-way',**summary['market_winner_from_six_way_no_vig']},
    ]).to_csv(out/'comparison.csv',index=False)
    with open(out/'summary.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
