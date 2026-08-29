#!/usr/bin/env python3
"""Research-only WHR-style KO/SUB attacker/defender survival traits.

Fits time-varying latent attacker offense and defender resistance states jointly over
whole history with Gaussian random-walk priors. Observation likelihood matches the
existing exponential survival clock; the fitted fighter RR multiplies the already
validated round-piecewise population baseline. No Brain/production changes.

Selection: choose w on 2023-24 using history before 2023.
Confirmation: refit through 2024-12-31 and predict frozen 2025+.
Comparator: existing empirical-Bayes attacker/defender piecewise clock, with its
prior-events hyperparameter selected on the same 2023-24 window.
"""
from __future__ import annotations

import json, math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from pipeline.research import ko_time_survival_oos as ko
from pipeline.research import sub_time_survival_oos as sub

OUT=Path('data/diagnostics/finish_hazard_whr_oos')
INNER=pd.Timestamp('2023-01-01')
TEST=pd.Timestamp('2025-01-01')
W_GRID=[0.02,0.03,0.05,0.08,0.12,0.20,0.35,0.50]
PRIOR_GRID=[0.5,1.0,2.0,5.0,10.0]

@dataclass
class Fit:
    beta: float; w: float; off: dict[str,float]; deff: dict[str,float]
    converged: bool; nit: int; objective: float; states: int

def make_states(ff,event_col):
    f=ff.sort_values(['event_date','fight_id','fighter_id']).copy()
    # Keep all fighter appearances as states because each side supplies defense evidence.
    keys=sorted(set(zip(f.fighter_id.astype(str),pd.to_datetime(f.event_date))))
    idx={k:i for i,k in enumerate(keys)}; n=len(keys)
    obs=[]
    for r in f.itertuples(index=False):
        ak=(str(r.fighter_id),pd.Timestamp(r.event_date)); dk=(str(r.opponent_id),pd.Timestamp(r.event_date))
        if dk not in idx: continue
        obs.append((idx[ak],n+idx[dk],float(getattr(r,event_col)),float(r.fight_seconds)))
    def links(offset):
        by=defaultdict(list)
        for fighter,dt in keys: by[fighter].append(pd.Timestamp(dt))
        out=[]
        for fighter,dates in by.items():
            dates=sorted(dates)
            for a,b in zip(dates[:-1],dates[1:]):
                yrs=max((b-a).days/365.25,1/365.25)
                out.append((offset+idx[(fighter,a)],offset+idx[(fighter,b)],yrs))
        return out
    return keys,idx,obs,links(0),links(n)

def fit_whr(ff,event_col,w):
    keys,idx,obs,olinks,dlinks=make_states(ff,event_col); n=len(keys); ns=2*n
    ev=float(ff[event_col].sum()); sec=float(ff.fight_seconds.sum()); beta0=math.log(max(ev/sec,1e-9))
    x0=np.zeros(1+ns); x0[0]=beta0
    first={};
    for k in keys: first.setdefault(k[0],idx[k])
    def fg(x):
        beta=x[0]; s=x[1:]; val=0.5*(beta/3.0)**2; g=np.zeros_like(x); g[0]=beta/9.0
        for oi,di,y,t in obs:
            eta=beta+s[oi]-s[di]; h=math.exp(np.clip(eta,-20,5)); H=h*t
            val += H - y*eta
            r=H-y; g[0]+=r; g[1+oi]+=r; g[1+di]-=r
        # anchor first offense/defense states around population
        for i in first.values():
            val+=0.5*(s[i]/0.8)**2; g[1+i]+=s[i]/0.64
            j=n+i; val+=0.5*(s[j]/0.8)**2; g[1+j]+=s[j]/0.64
        w2=max(w*w,1e-8)
        for i,j,yrs in olinks+dlinks:
            var=w2*yrs; d=s[j]-s[i]; val+=0.5*d*d/var; q=d/var; g[1+j]+=q; g[1+i]-=q
        return val,g
    res=minimize(lambda x:fg(x),x0,jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-9,'maxls':40})
    s=res.x[1:]; off={}; deff={}
    for fighter,dt in keys:
        off[fighter]=float(s[idx[(fighter,dt)]])
        deff[fighter]=float(s[n+idx[(fighter,dt)]])
    return Fit(float(res.x[0]),w,off,deff,bool(res.success),int(res.nit),float(res.fun),ns)

def piecewise_baseline(module,train,event_col):
    _,piece=module.train_baselines(train)
    return piece

def whr_score(module,fit,test,event_col,piece):
    off=test.fighter_id.astype(str).map(fit.off).fillna(0.0).to_numpy(float)
    de=test.opponent_id.astype(str).map(fit.deff).fillna(0.0).to_numpy(float)
    rr=np.clip(np.exp(off-de),0.05,20.0)
    y=test[event_col].to_numpy(int); t=test.fight_seconds.to_numpy(float)
    nll,_=module.survival_nll_piecewise(y,t,rr,piece)
    auc=roc_auc_score(y,rr) if np.unique(y).size==2 else np.nan
    return {'n':len(test),'events':int(y.sum()),'survival_nll':float(nll),'event_auc_prefight_hazard':float(auc),'mean_rr':float(rr.mean())},rr

def eb_score(module,train,test,event_col,pe):
    # Construct prefight cumulative history using only rows available before target cutoff.
    allx=module.add_prefight(pd.concat([train,test],ignore_index=True).sort_values(['event_date','fight_id','fighter_id']))
    target=allx[allx.fight_id.astype(str).isin(set(test.fight_id.astype(str))) & allx.fighter_id.astype(str).isin(set(test.fighter_id.astype(str)))]
    target=target[target.event_date>=test.event_date.min()].copy()
    # exact target row join avoids accidental extra rows
    target=test[['event_date','fight_id','fighter_id']].merge(target,on=['event_date','fight_id','fighter_id'],how='left',validate='one_to_one')
    p0,piece=module.train_baselines(train)
    rr=module.fighter_rr(target,p0,pe)
    y=target[event_col].to_numpy(int); t=target.fight_seconds.to_numpy(float)
    nll,_=module.survival_nll_piecewise(y,t,rr,piece)
    auc=roc_auc_score(y,rr) if np.unique(y).size==2 else np.nan
    return {'n':len(target),'events':int(y.sum()),'survival_nll':float(nll),'event_auc_prefight_hazard':float(auc),'mean_rr':float(rr.mean())}

def run_one(label,module,event_col):
    ff=module.load_fighter_fights().copy()
    tr0=ff[ff.event_date<INNER]; val=ff[(ff.event_date>=INNER)&(ff.event_date<TEST)]
    sweep=[]
    for w in W_GRID:
        fit=fit_whr(tr0,event_col,w); piece=piecewise_baseline(module,tr0,event_col); m,_=whr_score(module,fit,val,event_col,piece)
        sweep.append({'w':w,'converged':fit.converged,'iterations':fit.nit,'objective':fit.objective,'states':fit.states,**m})
        print(label,'W',w,m,'conv',fit.converged,'nit',fit.nit,flush=True)
    sw=pd.DataFrame(sweep).sort_values(['survival_nll','event_auc_prefight_hazard'],ascending=[True,False])
    best_w=float(sw.iloc[0].w)
    ebsel=[]
    for pe in PRIOR_GRID: ebsel.append({'prior_events':pe,**eb_score(module,tr0,val,event_col,pe)})
    ebdf=pd.DataFrame(ebsel).sort_values(['survival_nll','event_auc_prefight_hazard'],ascending=[True,False]); best_pe=float(ebdf.iloc[0].prior_events)
    train=ff[ff.event_date<TEST]; hold=ff[ff.event_date>=TEST]
    fit=fit_whr(train,event_col,best_w); piece=piecewise_baseline(module,train,event_col); wm,rr=whr_score(module,fit,hold,event_col,piece)
    em=eb_score(module,train,hold,event_col,best_pe)
    pop_rr=np.ones(len(hold)); y=hold[event_col].to_numpy(int); t=hold.fight_seconds.to_numpy(float); pnll,_=module.survival_nll_piecewise(y,t,pop_rr,piece)
    pauc=roc_auc_score(y,pop_rr) if np.unique(y).size==2 else np.nan
    pred=hold[['event_date','fight_id','fighter_id','fighter_name','opponent_id','fight_seconds',event_col]].copy(); pred['whr_rr']=rr
    return sw,ebdf,pred,{'selected_w':best_w,'selected_eb_prior_events':best_pe,'final_fit_converged':fit.converged,'final_fit_iterations':fit.nit,'final_fit_states':fit.states,'whr':wm,'existing_eb_piecewise':em,'population_piecewise':{'n':len(hold),'events':int(y.sum()),'survival_nll':float(pnll),'event_auc_prefight_hazard':float(pauc)}}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report={'architecture':'whole-history temporal attacker/defender survival ratings','cutoff':'2025-01-01','production_changed':False,'brain_used':False}
    for label,module,event in [('ko',ko,'ko_event'),('sub',sub,'sub_event')]:
        sw,eb,pred,res=run_one(label,module,event)
        sw.to_csv(OUT/f'{label}_w_sweep.csv',index=False); eb.to_csv(OUT/f'{label}_eb_prior_sweep.csv',index=False); pred.to_csv(OUT/f'{label}_holdout_predictions.csv',index=False); report[label]=res
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
