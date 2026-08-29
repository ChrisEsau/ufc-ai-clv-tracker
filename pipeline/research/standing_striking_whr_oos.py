#!/usr/bin/env python3
"""Research-only WHR-style standing striking OOS validation.

Two independent whole-history temporal latent systems:
  1) standing strike attempts: NB2 rate likelihood
       log(mu) = log(exposure/900) + beta + TEND_attacker(t) + ALLOW_defender(t)
     where negative ALLOW means suppression.
  2) standing strike accuracy: Beta-Binomial likelihood
       logit(p) = beta + OFF_attacker(t) - DEF_defender(t)

Each fighter's latent state is indexed by appearance date and consecutive states
are linked with a Gaussian random-walk penalty var = w^2 * elapsed_years.

Strict OOS design:
  * choose w on <=2022 train / 2023-24 validation;
  * refit through 2024-12-31;
  * freeze and predict 2025+ without holdout updates;
  * compare against frozen current FSR V3 on the identical holdout.

No FSR or Brain production files are modified.
"""
from __future__ import annotations

import json, math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, digamma

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.math import nb2_log_likelihood, beta_binomial_log_likelihood
from pipeline.fsr_v3.replay.paired_effectiveness import (
    build_effectiveness_fighter_fights,
    replay_paired_effectiveness,
    standing_effectiveness_spec,
)
from pipeline.fsr_v3.replay.rate_families import (
    build_rate_fighter_fights,
    replay_tendency,
    replay_suppression,
    standing_spec,
)

OUT=Path("data/diagnostics/standing_striking_whr_oos")
INNER_CUTOFF=pd.Timestamp("2023-01-01")
TEST_CUTOFF=pd.Timestamp("2025-01-01")
W_GRID=[0.03,0.05,0.08,0.12,0.20,0.35,0.50,0.75]
EPS=1e-10

@dataclass
class Fit:
    beta: float
    w: float
    a_latest: dict[str,float]
    d_latest: dict[str,float]
    objective: float
    converged: bool
    iterations: int
    n_states: int
    extra: float


def _state_layout(frame:pd.DataFrame):
    f=frame.sort_values(["event_date","fight_id","fighter_id"]).copy()
    a_keys=sorted(set(zip(f.fighter_id.astype(str),pd.to_datetime(f.event_date))))
    d_keys=sorted(set(zip(f.opponent_id.astype(str),pd.to_datetime(f.event_date))))
    a_idx={k:i for i,k in enumerate(a_keys)}
    na=len(a_keys)
    d_idx={k:na+i for i,k in enumerate(d_keys)}
    obs=[]
    for r in f.itertuples(index=False):
        ak=(str(r.fighter_id),pd.Timestamp(r.event_date))
        dk=(str(r.opponent_id),pd.Timestamp(r.event_date))
        obs.append((a_idx[ak],d_idx[dk],r))
    def links(keys,idx):
        by=defaultdict(list)
        for fighter,dt in keys: by[fighter].append(pd.Timestamp(dt))
        out=[]
        for fighter,dates in by.items():
            dates=sorted(dates)
            for x,y in zip(dates[:-1],dates[1:]):
                yrs=max((y-x).days/365.25,1/365.25)
                out.append((idx[(fighter,x)],idx[(fighter,y)],yrs))
        return out
    return a_keys,d_keys,a_idx,d_idx,obs,links(a_keys,a_idx)+links(d_keys,d_idx)


def _first_indices(keys,idx):
    out={}
    for k in keys: out.setdefault(k[0],idx[k])
    return out


def _fit_nb2_population(frame):
    d=frame[frame.exposure_seconds>0].copy()
    y=d.numerator.to_numpy(float); e=d.exposure_seconds.to_numpy(float)
    q0=max(float(y.sum()/e.sum()*900),1e-3)
    def obj(th):
        q,alpha=np.exp(th)
        mu=e/900*q
        return -float(nb2_log_likelihood(y,mu,alpha).sum())
    r=minimize(obj,np.log([q0,0.3]),method="L-BFGS-B",bounds=[(-5,8),(-8,4)])
    q,alpha=np.exp(r.x)
    return float(math.log(q)),float(alpha)


def fit_rate(frame,w):
    train=frame[frame.exposure_seconds>0].copy()
    a_keys,d_keys,a_idx,d_idx,obs,links=_state_layout(train)
    n=len(a_keys)+len(d_keys)
    beta0,alpha=_fit_nb2_population(train)
    x0=np.zeros(1+n); x0[0]=beta0
    first_a=_first_indices(a_keys,a_idx); first_d=_first_indices(d_keys,d_idx)
    sig_a=0.45; sig_d=0.45; w2=max(w*w,1e-8)
    def fg(x):
        beta=float(x[0]); s=x[1:]; val=0.; g=np.zeros_like(x)
        val+=0.5*(beta/3)**2; g[0]+=beta/9
        size=1/max(alpha,1e-12)
        for ai,di,r in obs:
            y=float(r.numerator); e=float(r.exposure_seconds)
            eta=beta+s[ai]+s[di]+math.log(max(e/900,EPS))
            mu=math.exp(min(eta,20))
            val-=float(nb2_log_likelihood(np.array([y]),np.array([mu]),alpha)[0])
            resid=mu*(y+size)/(mu+size)-y
            g[0]+=resid; g[1+ai]+=resid; g[1+di]+=resid
        for i in first_a.values(): val+=0.5*(s[i]/sig_a)**2; g[1+i]+=s[i]/sig_a**2
        for i in first_d.values(): val+=0.5*(s[i]/sig_d)**2; g[1+i]+=s[i]/sig_d**2
        for i,j,yrs in links:
            var=w2*yrs; dd=s[j]-s[i]; val+=0.5*dd*dd/var; gg=dd/var
            g[1+j]+=gg; g[1+i]-=gg
        return val,g
    r=minimize(lambda x:fg(x),x0,jac=True,method="L-BFGS-B",options={"maxiter":500,"ftol":1e-9,"maxls":40})
    s=r.x[1:]
    al={}; dl={}
    for k in a_keys: al[k[0]]=float(s[a_idx[k]])
    for k in d_keys: dl[k[0]]=float(s[d_idx[k]])
    return Fit(float(r.x[0]),w,al,dl,float(r.fun),bool(r.success),int(r.nit),n,alpha)


def fit_accuracy(frame,w,rho):
    train=frame[frame.attempted>0].copy()
    a_keys,d_keys,a_idx,d_idx,obs,links=_state_layout(train)
    n=len(a_keys)+len(d_keys)
    p0=float(train.landed.sum()/train.attempted.sum()); beta0=float(logit(np.clip(p0,1e-4,1-1e-4)))
    x0=np.zeros(1+n); x0[0]=beta0
    first_a=_first_indices(a_keys,a_idx); first_d=_first_indices(d_keys,d_idx)
    sig_a=0.30; sig_d=0.30; w2=max(w*w,1e-8); c=1/rho-1
    def fg(x):
        beta=float(x[0]); s=x[1:]; val=0.; g=np.zeros_like(x)
        val+=0.5*(beta/3)**2; g[0]+=beta/9
        for ai,di,r in obs:
            y=float(r.landed); nn=float(r.attempted); z=beta+s[ai]-s[di]; p=float(expit(z))
            val-=float(beta_binomial_log_likelihood(np.array([y]),np.array([nn]),np.array([p]),rho)[0])
            a=p*c; b=(1-p)*c
            dldp=c*(digamma(y+a)-digamma(a)-digamma(nn-y+b)+digamma(b))
            dz=-dldp*p*(1-p)
            g[0]+=dz; g[1+ai]+=dz; g[1+di]-=dz
        for i in first_a.values(): val+=0.5*(s[i]/sig_a)**2; g[1+i]+=s[i]/sig_a**2
        for i in first_d.values(): val+=0.5*(s[i]/sig_d)**2; g[1+i]+=s[i]/sig_d**2
        for i,j,yrs in links:
            var=w2*yrs; dd=s[j]-s[i]; val+=0.5*dd*dd/var; gg=dd/var
            g[1+j]+=gg; g[1+i]-=gg
        return val,g
    r=minimize(lambda x:fg(x),x0,jac=True,method="L-BFGS-B",options={"maxiter":500,"ftol":1e-9,"maxls":40})
    s=r.x[1:]; al={}; dl={}
    for k in a_keys: al[k[0]]=float(s[a_idx[k]])
    for k in d_keys: dl[k[0]]=float(s[d_idx[k]])
    return Fit(float(r.x[0]),w,al,dl,float(r.fun),bool(r.success),int(r.nit),n,rho)


def predict_rate(fit,frame,prefix):
    d=frame.copy(); a=d.fighter_id.astype(str).map(fit.a_latest).fillna(0.).astype(float); b=d.opponent_id.astype(str).map(fit.d_latest).fillna(0.).astype(float)
    d[prefix+"_tend"]=a; d[prefix+"_allow"]=b
    d[prefix+"_mu"]=d.exposure_seconds/900*np.exp(fit.beta+a+b)
    return d


def predict_acc(fit,frame,prefix):
    d=frame.copy(); a=d.fighter_id.astype(str).map(fit.a_latest).fillna(0.).astype(float); b=d.opponent_id.astype(str).map(fit.d_latest).fillna(0.).astype(float)
    d[prefix+"_off"]=a; d[prefix+"_opp_def"]=b; d[prefix+"_p"]=expit(fit.beta+a-b)
    return d


def rate_metrics(d,mucol,alpha=None):
    x=d[d.exposure_seconds>0].copy(); y=x.numerator.to_numpy(float); mu=np.maximum(x[mucol].to_numpy(float),EPS)
    if alpha is None: alpha=0.3
    nll=-float(nb2_log_likelihood(y,mu,alpha).sum())/len(x)
    mae=float(np.mean(np.abs(y-mu))); rmse=float(np.sqrt(np.mean((y-mu)**2)))
    return {"fighter_fights":len(x),"attempts":int(y.sum()),"observed_mean":float(y.mean()),"predicted_mean":float(mu.mean()),"nb2_nll_per_fighter_fight":nll,"mae_attempts":mae,"rmse_attempts":rmse}


def acc_metrics(d,pcol):
    x=d[d.attempted>0].copy(); p=np.clip(x[pcol].to_numpy(float),1e-9,1-1e-9); y=x.landed.to_numpy(float); n=x.attempted.to_numpy(float)
    ll=-(y*np.log(p)+(n-y)*np.log(1-p)).sum()/n.sum(); b=((y/n-p)**2*n).sum()/n.sum()
    return {"fighter_fights":len(x),"attempts":int(n.sum()),"landed":int(y.sum()),"actual_rate":float(y.sum()/n.sum()),"predicted_rate":float((n*p).sum()/n.sum()),"attempt_log_loss":float(ll),"weighted_brier":float(b)}


def frozen_fsr_rate(allf,cutoff,target):
    spec=standing_spec(FSRV3Config()); hist=allf[allf.event_date<cutoff].copy(); th=replay_tendency(hist,spec); sh=replay_suppression(th,spec)
    tend=th.sort_values(["event_date","fight_id"]).groupby("fighter_id").tail(1).set_index("fighter_id")["post_rating"].to_dict()
    supp=sh.sort_values(["event_date","fight_id"]).groupby("fighter_id").tail(1).set_index("fighter_id")["post_rating"].to_dict()
    qpop=float(th.sort_values(["event_date","fight_id"])["population_rate_15m"].dropna().iloc[-1]); spop=float(sh.sort_values(["event_date","fight_id"])["population_multiplier"].dropna().iloc[-1]); alpha=float(sh.sort_values(["event_date","fight_id"])["observation_alpha"].dropna().iloc[-1])
    d=target.copy(); q=d.fighter_id.astype(str).map(tend).fillna(qpop).astype(float); s=d.opponent_id.astype(str).map(supp).fillna(spop).astype(float)
    d["fsr_tendency"]=q; d["fsr_opp_suppression"]=s; d["fsr_mu"]=d.exposure_seconds/900*q*s
    return d,alpha


def frozen_fsr_acc(allf,cutoff,target):
    spec=standing_effectiveness_spec(FSRV3Config()); hist=replay_paired_effectiveness(allf[allf.event_date<cutoff].copy(),spec)
    off=hist[hist.trait.eq("standing_striking_offense")].sort_values(["event_date","fight_id"]).groupby("fighter_id").tail(1).set_index("fighter_id")["post_rating"].to_dict()
    de=hist[hist.trait.eq("standing_striking_defense")].sort_values(["event_date","fight_id"]).groupby("fighter_id").tail(1).set_index("fighter_id")["post_rating"].to_dict()
    pop=float(hist.sort_values(["event_date","fight_id"])["population_baseline"].dropna().iloc[-1]); beta=float(logit(np.clip(pop,1e-6,1-1e-6)))
    d=target.copy(); o=d.fighter_id.astype(str).map(off).fillna(0.).astype(float); v=d.opponent_id.astype(str).map(de).fillna(0.).astype(float)
    d["fsr_off"]=o; d["fsr_opp_def"]=v; d["fsr_p"]=expit(beta+o-v)
    return d


def main():
    OUT.mkdir(parents=True,exist_ok=True); paired=build_paired_rounds(); cfg=FSRV3Config()
    rs=standing_spec(cfg); es=standing_effectiveness_spec(cfg)
    rate=build_rate_fighter_fights(rs,paired); acc=build_effectiveness_fighter_fights(es,paired)

    # Tune attempt-rate temporal drift on 2023-24.
    rin=rate[rate.event_date<INNER_CUTOFF]; rval=rate[(rate.event_date>=INNER_CUTOFF)&(rate.event_date<TEST_CUTOFF)&(rate.exposure_seconds>0)]
    rate_sweep=[]
    for w in W_GRID:
        f=fit_rate(rin,w); pr=predict_rate(f,rval,"whr"); m=rate_metrics(pr,"whr_mu",f.extra); rate_sweep.append({"w":w,"converged":f.converged,"iterations":f.iterations,"objective":f.objective,"states":f.n_states,**m}); print("RATE",w,m,"conv",f.converged,flush=True)
    rsw=pd.DataFrame(rate_sweep).sort_values(["nb2_nll_per_fighter_fight","rmse_attempts"]); rw=float(rsw.iloc[0].w)

    # Tune accuracy temporal drift on 2023-24.
    ain=acc[acc.event_date<INNER_CUTOFF]; aval=acc[(acc.event_date>=INNER_CUTOFF)&(acc.event_date<TEST_CUTOFF)&(acc.attempted>0)]
    acc_sweep=[]
    rho=float(es.rho)
    for w in W_GRID:
        f=fit_accuracy(ain,w,rho); pa=predict_acc(f,aval,"whr"); m=acc_metrics(pa,"whr_p"); acc_sweep.append({"w":w,"converged":f.converged,"iterations":f.iterations,"objective":f.objective,"states":f.n_states,**m}); print("ACC",w,m,"conv",f.converged,flush=True)
    asw=pd.DataFrame(acc_sweep).sort_values(["attempt_log_loss","weighted_brier"]); aw=float(asw.iloc[0].w)

    # Frozen 2025+ tests.
    rtrain=rate[rate.event_date<TEST_CUTOFF]; rhold=rate[(rate.event_date>=TEST_CUTOFF)&(rate.exposure_seconds>0)].copy(); rf=fit_rate(rtrain,rw); rp=predict_rate(rf,rhold,"whr"); fr,fsr_alpha=frozen_fsr_rate(rate,TEST_CUTOFF,rhold)
    rkey=["event_date","fight_id","fighter_id","fighter_name","opponent_id","opponent_name","numerator","exposure_seconds"]
    rm=rp.merge(fr[rkey+["fsr_tendency","fsr_opp_suppression","fsr_mu"]],on=rkey,validate="one_to_one")
    pop_rate=float(rtrain.numerator.sum()/rtrain.exposure_seconds.sum()*900); rm["population_mu"]=rm.exposure_seconds/900*pop_rate

    atrain=acc[acc.event_date<TEST_CUTOFF]; ahold=acc[(acc.event_date>=TEST_CUTOFF)&(acc.attempted>0)].copy(); af=fit_accuracy(atrain,aw,rho); ap=predict_acc(af,ahold,"whr"); fa=frozen_fsr_acc(acc,TEST_CUTOFF,ahold)
    akey=["event_date","fight_id","fighter_id","fighter_name","opponent_id","opponent_name","landed","attempted"]
    am=ap.merge(fa[akey+["fsr_off","fsr_opp_def","fsr_p"]],on=akey,validate="one_to_one"); pop_acc=float(atrain.landed.sum()/atrain.attempted.sum()); am["population_p"]=pop_acc

    summary={
      "cutoff":"2025-01-01",
      "rate":{"selected_w":rw,"fit_converged":rf.converged,"iterations":rf.iterations,"states":rf.n_states,"whr":rate_metrics(rm,"whr_mu",rf.extra),"frozen_fsr_v3":rate_metrics(rm,"fsr_mu",fsr_alpha),"population_only":rate_metrics(rm,"population_mu",rf.extra)},
      "accuracy":{"selected_w":aw,"rho":rho,"fit_converged":af.converged,"iterations":af.iterations,"states":af.n_states,"whr":acc_metrics(am,"whr_p"),"frozen_fsr_v3":acc_metrics(am,"fsr_p"),"population_only":acc_metrics(am,"population_p")},
      "note":"All 2025+ predictions frozen at 2025-01-01. WHR uses same NB2/Beta-Binomial observation families as current FSR but jointly fits time-linked opponent-adjusted latent states."
    }
    rsw.to_csv(OUT/"rate_w_sweep.csv",index=False); asw.to_csv(OUT/"accuracy_w_sweep.csv",index=False); rm.to_csv(OUT/"rate_holdout_predictions.csv",index=False); am.to_csv(OUT/"accuracy_holdout_predictions.csv",index=False); (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
