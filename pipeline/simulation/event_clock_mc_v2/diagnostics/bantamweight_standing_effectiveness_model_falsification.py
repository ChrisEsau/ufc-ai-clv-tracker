"""Falsification audit for standing-effectiveness representation.

Measurement only. No production FSR/MC changes.

Compare three leakage-safe predictors of next-fight standing accuracy on the same
bantamweight fighter-fight rows:
  1) direct fighter effectiveness: cumulative prior landed/attempted with a fixed
     population pseudo-count;
  2) current FSR V3 paired offense/defense replay matchup probability;
  3) joint online offense-defense logistic model, where attacker and defender
     effects are learned simultaneously from prior observations only.

The joint arm is intentionally a falsification test, not a proposed production
replacement. Same-event observations are predicted before that event updates the
model. Market is diagnostic only and is never a fitting target.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.paired_effectiveness import (
    build_effectiveness_fighter_fights,
    standing_effectiveness_spec,
    replay_paired_effectiveness,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import (
    build_two_way_market, MARKET_PATH,
)

OUT = Path('data/diagnostics/event_clock_mc_v2/bantamweight_standing_effectiveness_model_falsification')
DIVISION = 'bantamweight'
DIRECT_PRIOR_ATTEMPTS = 40.0
JOINT_ALPHA = 1e-4
SEED = 20260824


def weighted_logloss(y, n, p):
    y=np.asarray(y,float); n=np.asarray(n,float); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return float(-(y*np.log(p)+(n-y)*np.log(1-p)).sum()/max(n.sum(),1.0))


def score_frame(df, pred_col, arm):
    z=df[['landed','attempted','actual_accuracy',pred_col]].dropna().copy()
    z=z[z['attempted']>0]
    if len(z)==0:
        return {'arm':arm,'n':0}
    err=z[pred_col]-z['actual_accuracy']
    w=z['attempted'].to_numpy(float)
    return {
        'arm':arm,
        'n':len(z),
        'attempts':float(w.sum()),
        'mae_unweighted':float(np.abs(err).mean()),
        'mae_attempt_weighted':float(np.average(np.abs(err),weights=w)),
        'rmse_attempt_weighted':float(np.sqrt(np.average(err**2,weights=w))),
        'binomial_log_loss_per_attempt':weighted_logloss(z['landed'],z['attempted'],z[pred_col]),
        'corr':float(z[[pred_col,'actual_accuracy']].corr().iloc[0,1]),
        'pred_sd':float(z[pred_col].std()),
        'actual_sd':float(z['actual_accuracy'].std()),
    }


def _int32_sparse(x):
    """Keep scipy/sklearn sparse index dtypes compatible on hosted runners."""
    if hasattr(x, 'indices'):
        x.indices = x.indices.astype(np.int32, copy=False)
    if hasattr(x, 'indptr'):
        x.indptr = x.indptr.astype(np.int32, copy=False)
    return x


def build_joint_predictions(fights: pd.DataFrame) -> pd.DataFrame:
    """Online simultaneous attacker(+1)/defender(-1) effects, same-event delayed."""
    fighters=sorted(set(fights['fighter_id'].astype(str)) | set(fights['opponent_id'].astype(str)))
    vocab=[]
    for f in fighters:
        vocab.append(f'off={f}'); vocab.append(f'def={f}')
    vec=DictVectorizer(sparse=True)
    vec.fit([{k:1.0 for k in vocab}])
    clf=SGDClassifier(loss='log_loss', penalty='l2', alpha=JOINT_ALPHA,
                      fit_intercept=True, learning_rate='optimal', random_state=SEED,
                      average=True)
    initialized=False
    rows=[]
    global_y=0.0; global_n=0.0

    def feat(att, deff):
        return {f'off={att}':1.0, f'def={deff}':-1.0}

    for event_date,batch in fights.groupby('event_date',sort=True):
        recs=batch.to_dict('records')
        for r in recs:
            att=str(r['fighter_id']); deff=str(r['opponent_id'])
            if initialized:
                x=_int32_sparse(vec.transform([feat(att,deff)]))
                p=float(clf.predict_proba(x)[0,1])
            else:
                p=float(global_y/global_n) if global_n>0 else 0.40
            rows.append({'event_date':event_date,'fight_id':str(r['fight_id']),
                         'fighter_id':att,'joint_probability':p})
        Xdict=[]; yy=[]; ww=[]
        for r in recs:
            n=float(r['attempted']); y=float(r['landed'])
            if n<=0: continue
            d=feat(str(r['fighter_id']),str(r['opponent_id']))
            if y>0:
                Xdict.append(d); yy.append(1); ww.append(y)
            if n-y>0:
                Xdict.append(d); yy.append(0); ww.append(n-y)
            global_y += y; global_n += n
        if Xdict:
            X=_int32_sparse(vec.transform(Xdict))
            if not initialized:
                clf.partial_fit(X,np.asarray(yy),classes=np.array([0,1]),sample_weight=np.asarray(ww,float))
                initialized=True
            else:
                clf.partial_fit(X,np.asarray(yy),sample_weight=np.asarray(ww,float))
    return pd.DataFrame(rows)


def main():
    cfg=FSRV3Config()
    spec=standing_effectiveness_spec(cfg)
    fights=build_effectiveness_fighter_fights(spec)
    fights['event_date']=pd.to_datetime(fights['event_date']).dt.normalize()
    fights['fight_id']=fights['fight_id'].astype(str); fights['fighter_id']=fights['fighter_id'].astype(str)

    hist=replay_paired_effectiveness(fights,spec)
    cur=hist[hist['trait'].eq(spec.offense_trait)][['event_date','fight_id','fighter_id','matchup_expected_probability','pre_rating']].copy()
    cur=cur.rename(columns={'matchup_expected_probability':'current_probability','pre_rating':'current_offense'})

    x=fights.sort_values(['event_date','fight_id','fighter_id']).copy()
    global_y=0.0; global_n=0.0; state={}
    direct_rows=[]
    for event_date,batch in x.groupby('event_date',sort=True):
        pop=global_y/global_n if global_n>0 else 0.40
        for r in batch.to_dict('records'):
            f=str(r['fighter_id']); fy,fn=state.get(f,(0.0,0.0))
            p=(fy + DIRECT_PRIOR_ATTEMPTS*pop)/(fn + DIRECT_PRIOR_ATTEMPTS)
            direct_rows.append({'event_date':event_date,'fight_id':str(r['fight_id']),'fighter_id':f,
                                'direct_probability':float(p),'prior_attempts':fn})
        for r in batch.to_dict('records'):
            f=str(r['fighter_id']); y=float(r['landed']); n=float(r['attempted'])
            fy,fn=state.get(f,(0.0,0.0)); state[f]=(fy+y,fn+n)
            global_y += y; global_n += n
    direct=pd.DataFrame(direct_rows)
    joint=build_joint_predictions(fights)

    pred=(fights.merge(cur,on=['event_date','fight_id','fighter_id'],how='left',validate='one_to_one')
          .merge(direct,on=['event_date','fight_id','fighter_id'],how='left',validate='one_to_one')
          .merge(joint,on=['event_date','fight_id','fighter_id'],how='left',validate='one_to_one'))
    pred['actual_accuracy']=np.where(pred['attempted']>0,pred['landed']/pred['attempted'],np.nan)

    master=pd.read_parquet(MASTER_PATH).drop_duplicates('fight_id').copy()
    master['fight_id']=master['fight_id'].astype(str)
    master['event_date']=pd.to_datetime(master['date'],errors='coerce').dt.normalize()
    bw=master[master['division'].astype(str).str.strip().str.lower().eq(DIVISION)][['fight_id','event_date','r_id','b_id','r_name','b_name']].copy()
    bw['r_id']=bw['r_id'].astype(str); bw['b_id']=bw['b_id'].astype(str)
    pred=pred[pred['fight_id'].isin(set(bw['fight_id']))].copy()
    pred=pred.merge(bw,on=['fight_id','event_date'],how='left',validate='many_to_one')
    pred['side']=np.where(pred['fighter_id'].eq(pred['r_id']),'red',np.where(pred['fighter_id'].eq(pred['b_id']),'blue','other'))
    pred=pred[pred['side'].ne('other')].sort_values(['event_date','fight_id','side']).reset_index(drop=True)

    fight_dates=pred[['event_date','fight_id']].drop_duplicates().sort_values(['event_date','fight_id']).reset_index(drop=True)
    cut=int(len(fight_dates)*0.70)
    test_ids=set(fight_dates.iloc[cut:]['fight_id'])
    test=pred[pred['fight_id'].isin(test_ids)].copy()
    common=test.dropna(subset=['actual_accuracy','direct_probability','current_probability','joint_probability']).copy()

    summary=pd.DataFrame([
        score_frame(common,'direct_probability','direct_fighter_effectiveness'),
        score_frame(common,'current_probability','current_fsr_paired'),
        score_frame(common,'joint_probability','joint_online_offense_defense'),
    ])

    wide=[]
    for fid,g in test.groupby('fight_id'):
        if set(g['side']) != {'red','blue'}: continue
        r=g[g['side'].eq('red')].iloc[0]; b=g[g['side'].eq('blue')].iloc[0]
        rec={'fight_id':fid,'event_date':r['event_date'],'red':r['r_name'],'blue':r['b_name']}
        for arm,col in [('direct','direct_probability'),('current','current_probability'),('joint','joint_probability')]:
            rec[f'{arm}_red_minus_blue_accuracy_gap']=float(r[col]-b[col])
        rec['realized_red_minus_blue_accuracy_gap']=float(r['actual_accuracy']-b['actual_accuracy']) if pd.notna(r['actual_accuracy']) and pd.notna(b['actual_accuracy']) else np.nan
        wide.append(rec)
    fight_detail=pd.DataFrame(wide)
    market=build_two_way_market(MARKET_PATH).copy(); market['fight_id']=market['fight_id'].astype(str)
    mcols=['fight_id','favorite_id','market_favorite_fair_p']
    fight_detail=fight_detail.merge(market[mcols],on='fight_id',how='left')
    fight_detail=fight_detail.merge(bw[['fight_id','r_id','b_id']],on='fight_id',how='left')
    is_red=fight_detail['favorite_id'].astype(str).eq(fight_detail['r_id'].astype(str))
    is_blue=fight_detail['favorite_id'].astype(str).eq(fight_detail['b_id'].astype(str))
    fight_detail['favorite_side']=pd.Series(np.where(is_red,'red',np.where(is_blue,'blue','')),index=fight_detail.index,dtype='object').replace('',pd.NA)
    for arm in ['direct','current','joint']:
        gap=fight_detail[f'{arm}_red_minus_blue_accuracy_gap']
        fight_detail[f'{arm}_favorite_accuracy_gap']=np.where(fight_detail['favorite_side'].eq('red'),gap,-gap)
    rgap=fight_detail['realized_red_minus_blue_accuracy_gap']
    fight_detail['realized_favorite_accuracy_gap']=np.where(fight_detail['favorite_side'].eq('red'),rgap,-rgap)

    bucket_rows=[]
    for label,lo in [('fav70+',.70),('fav80+',.80)]:
        z=fight_detail[fight_detail['market_favorite_fair_p']>=lo].dropna(subset=['realized_favorite_accuracy_gap']).copy()
        for arm in ['direct','current','joint']:
            q=z.dropna(subset=[f'{arm}_favorite_accuracy_gap'])
            bucket_rows.append({'group':label,'arm':arm,'n':len(q),
                                'mean_predicted_favorite_accuracy_gap':q[f'{arm}_favorite_accuracy_gap'].mean(),
                                'mean_realized_favorite_accuracy_gap':q['realized_favorite_accuracy_gap'].mean(),
                                'corr_predicted_realized_gap':q[[f'{arm}_favorite_accuracy_gap','realized_favorite_accuracy_gap']].corr().iloc[0,1] if len(q)>2 else np.nan})
    buckets=pd.DataFrame(bucket_rows)

    OUT.mkdir(parents=True,exist_ok=True)
    pred.to_csv(OUT/'fighter_fight_predictions.csv',index=False)
    summary.to_csv(OUT/'heldout_common_summary.csv',index=False)
    fight_detail.to_csv(OUT/'heldout_fight_gaps.csv',index=False)
    buckets.to_csv(OUT/'strong_favorite_gap_summary.csv',index=False)

    print('BANTAMWEIGHT STANDING EFFECTIVENESS MODEL FALSIFICATION')
    print(f'bantam_fights={len(fight_dates)} test_fights={len(test_ids)} common_fighter_rows={len(common)} direct_prior_attempts={DIRECT_PRIOR_ATTEMPTS} joint_alpha={JOINT_ALPHA}')
    print('\nHELDOUT COMMON ROWS')
    print(summary.to_string(index=False,float_format=lambda v:f'{v:.5f}'))
    print('\nSTRONG FAVORITE GAP DIAGNOSTIC')
    print(buckets.to_string(index=False,float_format=lambda v:f'{v:.5f}'))
    print('\nDecision rule: the decomposition hypothesis is supported only if the joint arm materially improves next-fight accuracy proper scores/correlation on identical rows. If direct or current is as good or better, do not blame the split.')

if __name__=='__main__':
    main()
