#!/usr/bin/env python3
"""Full-holdout standalone Glicko-6 rerun with enhanced submission signal.

Research only. No Brain, FSR, or market inputs.

Base: pure direct joint Glicko-6 with non-method negative evidence weight 0.25.
SUB enhancement: train-only ridge-logistic correction using
  - latent SUB O/D logit
  - prefight submission-attempt opportunity
  - age delta
  - prefight TD-attempt opportunity
  - prefight control-share opportunity
The correction is fitted only on fights before the holdout cutoff, then applied
untouched to the holdout. Missing auxiliary features are imputed to the training
feature means, which corresponds to zero standardized incremental contribution.
Only R_SUB and B_SUB logits are adjusted; KO/DEC logits remain unchanged.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.standalone_glicko_six_way_partial_sweep import run as run_glicko, SIX_COLS, SIX_LABELS
from pipeline.research.submission_incremental_signal_test import (
    logit, fit_logistic, predict_logistic, binary_metrics, build_opportunity, add_age,
)
from pipeline.research.submission_td_control_incremental_test import build_td_ctrl_opportunity
from pipeline.research.submission_stat_auc_screen import method_family

EPS=1e-12
FEATURES=['od','opp','age','td','ctrl']


def make_side_frame(pred, opp, age, tc):
    d=(pred.merge(opp,on='bout_id',how='left')
           .merge(age,on='bout_id',how='left')
           .merge(tc,on='bout_id',how='left'))
    rows=[]
    for b in d.itertuples(index=False):
        am=method_family(getattr(b,'method',''))
        rows.append({'date':b.date,'bout_id':b.bout_id,'side':'R','y':int(am=='SUB' and b.winner==b.red_fighter),
                     'od':float(logit(b.q_r_sub)),'opp':b.r_subopp,'age':b.age_delta,'td':b.r_tdopp,'ctrl':b.r_ctrlopp})
        rows.append({'date':b.date,'bout_id':b.bout_id,'side':'B','y':int(am=='SUB' and b.winner==b.blue_fighter),
                     'od':float(logit(b.q_b_sub)),'opp':b.b_subopp,'age':-b.age_delta,'td':b.b_tdopp,'ctrl':b.b_ctrlopp})
    return pd.DataFrame(rows), d


def impute_with_train_means(train, frame, cols):
    means=train[cols].mean()
    return frame[cols].fillna(means).to_numpy(float), means


def six_metrics(df, P):
    d=df[df.actual_six.notna()].copy(); mask=df.actual_six.notna().to_numpy(); P=np.asarray(P,float)[mask]
    idx=np.array([SIX_LABELS.index(x) for x in d.actual_six]); y=np.zeros_like(P); y[np.arange(len(d)),idx]=1
    pt=np.clip(P[np.arange(len(d)),idx],EPS,1); pred=np.argmax(P,axis=1)
    out={'n':int(len(d)),'six_way_accuracy':float(np.mean(pred==idx)),'six_way_log_loss':float(-np.mean(np.log(pt))),
         'six_way_brier':float(np.mean(np.sum((P-y)**2,axis=1))),'mean_probability_actual_outcome':float(np.mean(pt))}
    methP=np.c_[P[:,0]+P[:,3],P[:,1]+P[:,4],P[:,2]+P[:,5]]; labs=['KO','SUB','DEC']; midx=np.array([labs.index(x) for x in d.actual_method])
    mp=np.clip(methP[np.arange(len(d)),midx],EPS,1)
    out['method_accuracy']=float(np.mean(np.argmax(methP,axis=1)==midx)); out['method_log_loss']=float(-np.mean(np.log(mp)))
    out['actual_method_shares']={m:float(np.mean(d.actual_method==m)) for m in labs}
    out['predicted_method_shares']={'KO':float(methP[:,0].mean()),'SUB':float(methP[:,1].mean()),'DEC':float(methP[:,2].mean())}
    submask=np.isin(idx,[1,4]); out['sub_recall_combined']=float(np.mean(pred[submask]==idx[submask])) if submask.any() else None
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--holdout-from',default='2025-01-01'); ap.add_argument('--output-dir',type=Path,default=Path('data/diagnostics/standalone_glicko_six_way_enhanced_sub')); args=ap.parse_args()
    cutoff=pd.Timestamp(args.holdout_from)
    bouts=build_bouts(pd.read_parquet('data/master/ufc_master.parquet'))
    pred=run_glicko(bouts,0.25)
    opp=build_opportunity(bouts,Path('data/fight_details/ufc_round_stats.parquet'))
    age=add_age(bouts)
    tc=build_td_ctrl_opportunity(bouts,Path('data/fight_details/ufc_round_stats.parquet'))
    side,fights=make_side_frame(pred,opp,age,tc); side['date']=pd.to_datetime(side.date)

    tr=side[side.date<cutoff].copy(); ho=side[side.date>=cutoff].copy()
    tr_complete=tr.dropna(subset=FEATURES).copy()
    model=fit_logistic(tr_complete[FEATURES].to_numpy(),tr_complete.y.to_numpy(),l2=1.0)
    Xho=ho[FEATURES].copy()
    train_means=tr_complete[FEATURES].mean()
    Xho_imp=Xho.fillna(train_means).to_numpy(float)
    p_side=predict_logistic(model,Xho_imp)
    side_metrics=binary_metrics(ho.y.to_numpy(),p_side)

    # OD-only reference model fit on same training rows for incremental logit delta.
    od_model=fit_logistic(tr_complete[['od']].to_numpy(),tr_complete.y.to_numpy(),l2=1.0)

    h=fights[pd.to_datetime(fights.date)>=cutoff].copy().reset_index(drop=True)
    P0=h[list(SIX_COLS)].to_numpy(float)
    # Build side feature matrices in exact fight order.
    r=pd.DataFrame({'od':logit(h.q_r_sub.to_numpy()),'opp':h.r_subopp,'age':h.age_delta,'td':h.r_tdopp,'ctrl':h.r_ctrlopp})
    b=pd.DataFrame({'od':logit(h.q_b_sub.to_numpy()),'opp':h.b_subopp,'age':-h.age_delta,'td':h.b_tdopp,'ctrl':h.b_ctrlopp})
    rimp=r.fillna(train_means).to_numpy(float); bimp=b.fillna(train_means).to_numpy(float)
    pr=predict_logistic(model,rimp); pb=predict_logistic(model,bimp)
    qr=predict_logistic(od_model,r[['od']].to_numpy()); qb=predict_logistic(od_model,b[['od']].to_numpy())
    dr=logit(pr)-logit(qr); db=logit(pb)-logit(qb)
    L=np.log(np.clip(P0,EPS,1)); L[:,1]+=dr; L[:,4]+=db; L-=L.max(axis=1,keepdims=True); E=np.exp(L); P=E/E.sum(axis=1,keepdims=True)

    enhanced=h.copy()
    for j,c in enumerate(SIX_COLS): enhanced[c]=P[:,j]
    enhanced['p_method_ko']=P[:,0]+P[:,3]; enhanced['p_method_sub']=P[:,1]+P[:,4]; enhanced['p_method_dec']=P[:,2]+P[:,5]
    base_metrics=six_metrics(h,P0); enhanced_metrics=six_metrics(enhanced,P)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    enhanced.to_csv(args.output_dir/'holdout_predictions.csv',index=False)
    ho2=ho.copy(); ho2['p_subwin_enhanced']=p_side; ho2.to_csv(args.output_dir/'side_sub_holdout.csv',index=False)
    summary={'architecture':'0.25 direct joint Glicko-6 + train-only enhanced SUB correction','holdout_from':args.holdout_from,
             'features':FEATURES,'train_complete_side_rows':int(len(tr_complete)),'holdout_side_rows':int(len(ho)),
             'missing_counts_holdout':{c:int(ho[c].isna().sum()) for c in FEATURES},'side_sub_metrics':side_metrics,
             'base_six_way':base_metrics,'enhanced_six_way':enhanced_metrics,
             'coefficients':{'intercept':float(model['coef'][0]),**{c:float(model['coef'][i+1]) for i,c in enumerate(FEATURES)}}}
    with open(args.output_dir/'summary.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
