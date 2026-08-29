#!/usr/bin/env python3
from __future__ import annotations
import json, math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.paired_effectiveness import build_effectiveness_fighter_fights,replay_paired_effectiveness,takedown_effectiveness_spec

OUT=Path('data/diagnostics/td_effectiveness_whr_oos')
INNER_CUTOFF=pd.Timestamp('2023-01-01'); TEST_CUTOFF=pd.Timestamp('2025-01-01')
W_GRID=[0.03,0.05,0.08,0.12,0.20,0.35,0.50,0.75]; EPS=1e-9
@dataclass
class Fit:
    beta:float; w:float; off_latest:dict; def_latest:dict; objective:float; converged:bool; iterations:int; n_states:int

def make_states(f):
    f=f.sort_values(['event_date','fight_id','fighter_id']).copy()
    off_keys=sorted(set((str(a),pd.Timestamp(d)) for a,d in zip(f.fighter_id,f.event_date)))
    def_keys=sorted(set((str(a),pd.Timestamp(d)) for a,d in zip(f.opponent_id,f.event_date)))
    off_idx={k:i for i,k in enumerate(off_keys)}; n=len(off_keys); def_idx={k:n+i for i,k in enumerate(def_keys)}
    obs=[]
    for r in f.itertuples(index=False):
        ak=(str(r.fighter_id),pd.Timestamp(r.event_date)); dk=(str(r.opponent_id),pd.Timestamp(r.event_date))
        obs.append((off_idx[ak],def_idx[dk],float(r.landed),float(r.attempted)))
    def links(keys,idx):
        by=defaultdict(list)
        for fighter,dt in keys: by[fighter].append(dt)
        out=[]
        for fighter,dates in by.items():
            dates=sorted(dates)
            for a,b in zip(dates[:-1],dates[1:]): out.append((idx[(fighter,a)],idx[(fighter,b)],max((b-a).days/365.25,1/365.25)))
        return out
    return off_keys,def_keys,off_idx,def_idx,obs,links(off_keys,off_idx),links(def_keys,def_idx)

def fit_whr(fights,w):
    train=fights[fights.attempted>0].copy(); off_keys,def_keys,off_idx,def_idx,obs,ol,dl=make_states(train)
    ns=len(off_keys)+len(def_keys); rate=float(train.landed.sum()/train.attempted.sum()); x0=np.zeros(1+ns); x0[0]=float(logit(np.clip(rate,1e-4,1-1e-4)))
    first_o={}; first_d={}
    for k in off_keys: first_o.setdefault(k[0],off_idx[k])
    for k in def_keys: first_d.setdefault(k[0],def_idx[k])
    def fg(x):
        beta=float(x[0]); s=x[1:]; val=.5*(beta/3)**2; g=np.zeros_like(x); g[0]=beta/9
        for oi,di,y,n in obs:
            z=beta+s[oi]-s[di]; p=float(expit(z)); val-=y*math.log(max(p,EPS))+(n-y)*math.log(max(1-p,EPS)); r=n*p-y
            g[0]+=r; g[1+oi]+=r; g[1+di]-=r
        for i in first_o.values(): val+=.5*(s[i]/.35)**2; g[1+i]+=s[i]/(.35**2)
        for i in first_d.values(): val+=.5*(s[i]/.50)**2; g[1+i]+=s[i]/(.50**2)
        w2=max(w*w,1e-8)
        for i,j,yrs in ol+dl:
            var=w2*yrs; d=s[j]-s[i]; val+=.5*d*d/var; q=d/var; g[1+j]+=q; g[1+i]-=q
        return val,g
    res=minimize(lambda x:fg(x),x0,jac=True,method='L-BFGS-B',options={'maxiter':350,'ftol':1e-9,'maxls':30})
    s=res.x[1:]; oo={}; dd={}
    for fighter,dt in off_keys: oo[fighter]=float(s[off_idx[(fighter,dt)]])
    for fighter,dt in def_keys: dd[fighter]=float(s[def_idx[(fighter,dt)]])
    return Fit(float(res.x[0]),w,oo,dd,float(res.fun),bool(res.success),int(res.nit),ns)

def predict(fit,d,prefix):
    d=d.copy(); o=d.fighter_id.astype(str).map(fit.off_latest).fillna(0.).astype(float); v=d.opponent_id.astype(str).map(fit.def_latest).fillna(0.).astype(float)
    d[prefix+'_off']=o; d[prefix+'_opp_def']=v; d[prefix+'_p']=expit(fit.beta+o-v); return d

def fsr_frozen(allf,cutoff,target):
    spec=takedown_effectiveness_spec(FSRV3Config()); hist=replay_paired_effectiveness(allf[allf.event_date<cutoff].copy(),spec)
    off=hist[hist.trait.eq('takedown_offense')].sort_values(['event_date','fight_id']).groupby('fighter_id').tail(1).set_index('fighter_id').post_rating.to_dict()
    de=hist[hist.trait.eq('takedown_defense')].sort_values(['event_date','fight_id']).groupby('fighter_id').tail(1).set_index('fighter_id').post_rating.to_dict()
    pop=float(hist.sort_values(['event_date','fight_id']).population_baseline.dropna().iloc[-1]); beta=float(logit(np.clip(pop,1e-6,1-1e-6)))
    d=target.copy(); o=d.fighter_id.astype(str).map(off).fillna(0.).astype(float); v=d.opponent_id.astype(str).map(de).fillna(0.).astype(float); d['fsr_off']=o; d['fsr_opp_def']=v; d['fsr_p']=expit(beta+o-v); return d

def metrics(d,pcol):
    p=np.clip(d[pcol].to_numpy(float),1e-9,1-1e-9); y=d.landed.to_numpy(float); n=d.attempted.to_numpy(float); A=float(n.sum()); ll=float(-(y*np.log(p)+(n-y)*np.log(1-p)).sum())
    return {'fighter_fights':int(len(d)),'attempts':int(A),'landed':int(y.sum()),'actual_rate':float(y.sum()/A),'predicted_rate':float((n*p).sum()/A),'attempt_log_loss':ll/A,'weighted_brier':float((((y/n)-p)**2*n).sum()/A)}

def main():
    OUT.mkdir(parents=True,exist_ok=True); spec=takedown_effectiveness_spec(FSRV3Config()); fights=build_effectiveness_fighter_fights(spec,build_paired_rounds()).sort_values(['event_date','fight_id','fighter_id']).reset_index(drop=True)
    it=fights[fights.event_date<INNER_CUTOFF]; iv=fights[(fights.event_date>=INNER_CUTOFF)&(fights.event_date<TEST_CUTOFF)&(fights.attempted>0)]
    sweep=[]
    for w in W_GRID:
        fit=fit_whr(it,w); m=metrics(predict(fit,iv,'whr'),'whr_p'); sweep.append({'w':w,'converged':fit.converged,'iterations':fit.iterations,'objective':fit.objective,'states':fit.n_states,**m}); print('W',w,m,'conv',fit.converged,'iter',fit.iterations,flush=True)
    sw=pd.DataFrame(sweep).sort_values(['attempt_log_loss','weighted_brier']).reset_index(drop=True); best=float(sw.iloc[0].w)
    train=fights[fights.event_date<TEST_CUTOFF]; hold=fights[(fights.event_date>=TEST_CUTOFF)&(fights.attempted>0)].copy(); fit=fit_whr(train,best); whr=predict(fit,hold,'whr'); fsr=fsr_frozen(fights,TEST_CUTOFF,hold)
    key=['event_date','fight_id','fighter_id','fighter_name','opponent_id','opponent_name','landed','attempted']; merged=whr.merge(fsr[key+['fsr_off','fsr_opp_def','fsr_p']],on=key,how='left',validate='one_to_one'); merged['population_p']=float(train.landed.sum()/train.attempted.sum())
    summary={'architecture':'whole-history time-varying binomial TD offense/defense','selected_w':best,'final_fit_converged':fit.converged,'final_fit_iterations':fit.iterations,'final_fit_states':fit.n_states,'test_from':'2025-01-01','whr':metrics(merged,'whr_p'),'frozen_fsr_v3':metrics(merged,'fsr_p'),'population_only':metrics(merged,'population_p'),'note':'WHR and FSR both frozen at 2025-01-01; no holdout updates.'}
    sw.to_csv(OUT/'w_sweep.csv',index=False); merged.to_csv(OUT/'holdout_predictions.csv',index=False); pd.DataFrame([{'fighter_id':k,'td_off_whr':v,'td_def_whr':fit.def_latest.get(k,0.)} for k,v in fit.off_latest.items()]).to_csv(OUT/'prefight_2025_ratings.csv',index=False); (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
