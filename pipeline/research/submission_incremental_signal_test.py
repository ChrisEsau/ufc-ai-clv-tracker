#!/usr/bin/env python3
"""Leakage-safe incremental SUB signal test.

Research-only. No Brain, FSR, or market inputs.

Question: does raw submission-attempt opportunity and/or age delta add signal
beyond the 0.25 partial-evidence Glicko SUB O/D rating?

Protocol:
- Rebuild the 0.25 pure-joint Glicko predictions chronologically.
- Build prefight SUB-attempt opportunity from raw UFC round stats using only
  prior fights: fighter own career mean SUB attempts + opponent career mean
  SUB attempts allowed.
- Build side-oriented age delta from profile DOB/static age.
- Fit ridge logistic regressions ONLY on pre-2025 side-level observations.
- Evaluate untouched 2025+ side-specific SUB-win AUC/log loss/Brier.
- For six-way integration, use the train-fit incremental logit delta relative
  to the O/D-only logistic model, add it only to R_SUB/B_SUB logits of the
  frozen 0.25 six-way probabilities, renormalize, and report six-way metrics.
"""
from __future__ import annotations

import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.prefight_strength_elo import build_bouts
from pipeline.research.standalone_glicko_six_way_partial_sweep import run as run_glicko, SIX_COLS, SIX_LABELS
from pipeline.research.submission_physical_delta_auc import load_profiles
from pipeline.research.submission_stat_auc_screen import auc_binary, method_family

EPS=1e-12


def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return np.log(p)-np.log1p(-p)

def sigmoid(x):
    x=np.asarray(x,float)
    return 1/(1+np.exp(-np.clip(x,-40,40)))

def standardize_fit(X):
    mu=np.nanmean(X,axis=0); sd=np.nanstd(X,axis=0); sd=np.where(sd<1e-9,1.0,sd)
    return mu,sd

def fit_logistic(X,y,l2=1.0,max_iter=100):
    X=np.asarray(X,float); y=np.asarray(y,float)
    mu,sd=standardize_fit(X); Z=(X-mu)/sd; Z=np.c_[np.ones(len(Z)),Z]
    b=np.zeros(Z.shape[1])
    pen=np.eye(len(b))*l2; pen[0,0]=0
    for _ in range(max_iter):
        p=sigmoid(Z@b); w=np.clip(p*(1-p),1e-6,None)
        g=Z.T@(p-y)+pen@b
        H=Z.T@(Z*w[:,None])+pen
        step=np.linalg.solve(H,g); b2=b-step
        if np.max(np.abs(b2-b))<1e-8: b=b2; break
        b=b2
    return {'coef':b,'mu':mu,'sd':sd}
def predict_logistic(model,X):
    Z=(np.asarray(X,float)-model['mu'])/model['sd']; Z=np.c_[np.ones(len(Z)),Z]
    return sigmoid(Z@model['coef'])

def binary_metrics(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    auc, n, n1, n0=auc_binary(y,p)
    return {'n':int(n),'positives':int(n1),'auc':float(auc),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),'brier':float(np.mean((p-y)**2))}


def build_opportunity(bouts, round_path):
    raw=pd.read_parquet(round_path).copy()
    raw['event_date']=pd.to_datetime(raw.event_date,errors='coerce').dt.normalize()
    raw['fighter_key']=raw.fighter_name.astype(str).str.strip().str.lower()
    raw['opp_key']=raw.opponent_name.astype(str).str.strip().str.lower()
    raw['sub_att']=pd.to_numeric(raw.sub_att,errors='coerce').fillna(0.0)
    fs=(raw.groupby(['fight_id','event_date','fighter_key','opp_key'],as_index=False)
          .agg(sub_att=('sub_att','sum')))
    lookup={(r.event_date,r.fighter_key,r.opp_key):float(r.sub_att) for r in fs.itertuples(index=False)}
    career=defaultdict(lambda:[0.0,0.0,0]) # own sum, allowed sum, fights
    rows=[]
    for b in bouts.sort_values(['date','bout_id']).itertuples(index=False):
        dt=pd.Timestamp(b.date).normalize(); r=str(b.red_fighter).strip().lower(); bl=str(b.blue_fighter).strip().lower()
        cr,cb=career[r],career[bl]
        r_own=cr[0]/cr[2] if cr[2]>0 else np.nan; r_allow=cr[1]/cr[2] if cr[2]>0 else np.nan
        b_own=cb[0]/cb[2] if cb[2]>0 else np.nan; b_allow=cb[1]/cb[2] if cb[2]>0 else np.nan
        rows.append({'bout_id':b.bout_id,'r_subopp':r_own+b_allow if np.isfinite(r_own) and np.isfinite(b_allow) else np.nan,
                     'b_subopp':b_own+r_allow if np.isfinite(b_own) and np.isfinite(r_allow) else np.nan})
        rv=lookup.get((dt,r,bl),np.nan); bv=lookup.get((dt,bl,r),np.nan)
        if np.isfinite(rv) and np.isfinite(bv):
            career[r][0]+=rv; career[r][1]+=bv; career[r][2]+=1
            career[bl][0]+=bv; career[bl][1]+=rv; career[bl][2]+=1
    return pd.DataFrame(rows)


def add_age(bouts):
    x=bouts.copy(); x['date']=pd.to_datetime(x.date)
    prof,_=load_profiles(Path('data')); pr=prof.add_prefix('r_'); pb=prof.add_prefix('b_')
    x['r_key']=x.red_fighter.astype(str).str.strip().str.lower(); x['b_key']=x.blue_fighter.astype(str).str.strip().str.lower()
    x=x.merge(pr,left_on='r_key',right_on='r_key',how='left').merge(pb,left_on='b_key',right_on='b_key',how='left')
    for side in ['r','b']:
        dyn=(x.date-x[f'{side}_dob']).dt.days/365.2425
        x[f'{side}_age']=dyn.where(dyn.notna(),x[f'{side}_age_static'])
    x['age_delta']=x.r_age-x.b_age
    return x[['bout_id','age_delta']]


def make_side_frame(pred,opp,age):
    d=pred.merge(opp,on='bout_id',how='left').merge(age,on='bout_id',how='left')
    rows=[]
    for b in d.itertuples(index=False):
        am=method_family(getattr(b,'method',''))
        rows.append({'date':b.date,'bout_id':b.bout_id,'side':'R','y':int(am=='SUB' and b.winner==b.red_fighter),
                     'od':logit(b.q_r_sub),'opp':b.r_subopp,'age':b.age_delta})
        rows.append({'date':b.date,'bout_id':b.bout_id,'side':'B','y':int(am=='SUB' and b.winner==b.blue_fighter),
                     'od':logit(b.q_b_sub),'opp':b.b_subopp,'age':-b.age_delta})
    return pd.DataFrame(rows), d


def six_metrics(df, P):
    d=df[df.actual_six.notna()].copy(); P=np.asarray(P,float)[df.actual_six.notna().to_numpy()]
    idx=np.array([SIX_LABELS.index(x) for x in d.actual_six]); y=np.zeros_like(P); y[np.arange(len(d)),idx]=1
    pt=np.clip(P[np.arange(len(d)),idx],EPS,1); pred=np.argmax(P,axis=1)
    return {'n':int(len(d)),'accuracy':float(np.mean(pred==idx)),'log_loss':float(-np.mean(np.log(pt))),'brier':float(np.mean(np.sum((P-y)**2,axis=1))),'mean_p_actual':float(np.mean(pt))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--holdout-from',default='2025-01-01'); ap.add_argument('--output-dir',type=Path,default=Path('data/diagnostics/submission_incremental_signal')); args=ap.parse_args()
    bouts=build_bouts(pd.read_parquet('data/master/ufc_master.parquet'))
    pred=run_glicko(bouts,0.25); opp=build_opportunity(bouts,Path('data/fight_details/ufc_round_stats.parquet')); age=add_age(bouts)
    side,fights=make_side_frame(pred,opp,age); side['date']=pd.to_datetime(side.date); cutoff=pd.Timestamp(args.holdout_from)
    specs={'OD':['od'],'OD+OPP':['od','opp'],'OD+AGE':['od','age'],'OD+OPP+AGE':['od','opp','age']}
    results=[]; models={}
    for name,cols in specs.items():
        tr=side[side.date<cutoff].dropna(subset=cols); ho=side[side.date>=cutoff].dropna(subset=cols)
        m=fit_logistic(tr[cols].to_numpy(),tr.y.to_numpy(),l2=1.0); models[name]=m
        pm=predict_logistic(m,ho[cols].to_numpy()); bm=binary_metrics(ho.y,pm)
        results.append({'model':name,'features':'+'.join(cols),'train_n':len(tr),**bm,'coef_intercept':float(m['coef'][0]),**{f'coef_{c}':float(m['coef'][i+1]) for i,c in enumerate(cols)}})
    res=pd.DataFrame(results).sort_values('auc',ascending=False)

    # Six-way test on common complete-case holdout. Add only incremental SUB-logit delta vs OD-only.
    h=fights[pd.to_datetime(fights.date)>=cutoff].copy(); baseP=h[list(SIX_COLS)].to_numpy(float)
    six=[]
    common=h[['r_subopp','b_subopp','age_delta']].notna().all(axis=1).to_numpy()
    hb=h.loc[common].copy(); P0=baseP[common]
    six.append({'model':'BASE_0.25','complete_case_n':int(common.sum()),**six_metrics(hb,P0)})
    od_model=models['OD']
    for name in ['OD+OPP','OD+AGE','OD+OPP+AGE']:
        m=models[name]; cols=specs[name]
        Xr=pd.DataFrame({'od':logit(hb.q_r_sub.to_numpy()),'opp':hb.r_subopp.to_numpy(),'age':hb.age_delta.to_numpy()})[cols].to_numpy()
        Xb=pd.DataFrame({'od':logit(hb.q_b_sub.to_numpy()),'opp':hb.b_subopp.to_numpy(),'age':-hb.age_delta.to_numpy()})[cols].to_numpy()
        pr=predict_logistic(m,Xr); pb=predict_logistic(m,Xb)
        qr=predict_logistic(od_model,logit(hb.q_r_sub.to_numpy())[:,None]); qb=predict_logistic(od_model,logit(hb.q_b_sub.to_numpy())[:,None])
        dr=logit(pr)-logit(qr); db=logit(pb)-logit(qb)
        L=np.log(np.clip(P0,EPS,1)); L[:,1]+=dr; L[:,4]+=db; L-=L.max(axis=1,keepdims=True); E=np.exp(L); P=E/E.sum(axis=1,keepdims=True)
        six.append({'model':name,'complete_case_n':len(hb),**six_metrics(hb,P)})
    sixdf=pd.DataFrame(six).sort_values('log_loss')

    args.output_dir.mkdir(parents=True,exist_ok=True); res.to_csv(args.output_dir/'side_incremental_results.csv',index=False); sixdf.to_csv(args.output_dir/'sixway_incremental_results.csv',index=False); side.to_csv(args.output_dir/'side_features.csv',index=False)
    summary={'holdout_from':args.holdout_from,'negative_evidence_weight':0.25,'side_results':res.to_dict('records'),'sixway_complete_case_results':sixdf.to_dict('records')}
    with open(args.output_dir/'summary.json','w') as f: json.dump(summary,f,indent=2)
    print('\nSIDE-SPECIFIC SUB WIN\n',res.to_string(index=False)); print('\nSIX-WAY COMPLETE CASE\n',sixdf.to_string(index=False))

if __name__=='__main__': main()
