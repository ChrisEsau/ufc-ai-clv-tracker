#!/usr/bin/env python3
"""Vectorized optimizer wrapper for standing_striking_whr_oos.py.
Same model/data/specification; only likelihood/gradient evaluation is vectorized.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, digamma

import pipeline.research.standing_striking_whr_oos as b
from pipeline.fsr_v3.replay.math import nb2_log_likelihood, beta_binomial_log_likelihood


def fit_rate_fast(frame,w):
    train=frame[frame.exposure_seconds>0].copy()
    a_keys,d_keys,a_idx,d_idx,obs,links=b._state_layout(train)
    n=len(a_keys)+len(d_keys); beta0,alpha=b._fit_nb2_population(train)
    x0=np.zeros(1+n); x0[0]=beta0
    sig_a=sig_d=.45; w2=max(w*w,1e-8)
    ai=np.array([o[0] for o in obs],dtype=int); di=np.array([o[1] for o in obs],dtype=int)
    y=np.array([float(o[2].numerator) for o in obs]); e=np.array([float(o[2].exposure_seconds) for o in obs])
    off=np.log(np.maximum(e/900,b.EPS)); size=1/max(alpha,1e-12)
    first_a=np.array(list(b._first_indices(a_keys,a_idx).values()),dtype=int); first_d=np.array(list(b._first_indices(d_keys,d_idx).values()),dtype=int)
    li=np.array([x[0] for x in links],dtype=int); lj=np.array([x[1] for x in links],dtype=int); ly=np.array([x[2] for x in links],dtype=float)
    def fg(x):
        beta=x[0]; s=x[1:]; eta=np.clip(beta+s[ai]+s[di]+off,-20,20); mu=np.exp(eta)
        val=-float(nb2_log_likelihood(y,mu,alpha).sum()) + .5*(beta/3)**2
        resid=mu*(y+size)/(mu+size)-y
        g=np.zeros_like(x); g[0]=resid.sum()+beta/9
        gs=g[1:]; np.add.at(gs,ai,resid); np.add.at(gs,di,resid)
        if len(first_a): val+=.5*np.sum((s[first_a]/sig_a)**2); np.add.at(gs,first_a,s[first_a]/sig_a**2)
        if len(first_d): val+=.5*np.sum((s[first_d]/sig_d)**2); np.add.at(gs,first_d,s[first_d]/sig_d**2)
        if len(li):
            var=w2*ly; dd=s[lj]-s[li]; val+=.5*np.sum(dd*dd/var); gg=dd/var
            np.add.at(gs,lj,gg); np.add.at(gs,li,-gg)
        return val,g
    r=minimize(lambda x:fg(x),x0,jac=True,method="L-BFGS-B",options={"maxiter":500,"ftol":1e-9,"maxls":40})
    s=r.x[1:]; al={}; dl={}
    for k in a_keys: al[k[0]]=float(s[a_idx[k]])
    for k in d_keys: dl[k[0]]=float(s[d_idx[k]])
    return b.Fit(float(r.x[0]),w,al,dl,float(r.fun),bool(r.success),int(r.nit),n,alpha)


def fit_accuracy_fast(frame,w,rho):
    train=frame[frame.attempted>0].copy()
    a_keys,d_keys,a_idx,d_idx,obs,links=b._state_layout(train)
    n=len(a_keys)+len(d_keys); p0=float(train.landed.sum()/train.attempted.sum()); beta0=float(np.log(np.clip(p0,1e-4,1-1e-4)/(1-np.clip(p0,1e-4,1-1e-4))))
    x0=np.zeros(1+n); x0[0]=beta0
    sig_a=sig_d=.30; w2=max(w*w,1e-8); c=1/rho-1
    ai=np.array([o[0] for o in obs],dtype=int); di=np.array([o[1] for o in obs],dtype=int)
    y=np.array([float(o[2].landed) for o in obs]); nn=np.array([float(o[2].attempted) for o in obs])
    first_a=np.array(list(b._first_indices(a_keys,a_idx).values()),dtype=int); first_d=np.array(list(b._first_indices(d_keys,d_idx).values()),dtype=int)
    li=np.array([x[0] for x in links],dtype=int); lj=np.array([x[1] for x in links],dtype=int); ly=np.array([x[2] for x in links],dtype=float)
    def fg(x):
        beta=x[0]; s=x[1:]; z=beta+s[ai]-s[di]; p=expit(z)
        val=-float(beta_binomial_log_likelihood(y,nn,p,rho).sum()) + .5*(beta/3)**2
        aa=p*c; bb=(1-p)*c
        dldp=c*(digamma(y+aa)-digamma(aa)-digamma(nn-y+bb)+digamma(bb)); dz=-dldp*p*(1-p)
        g=np.zeros_like(x); g[0]=dz.sum()+beta/9; gs=g[1:]
        np.add.at(gs,ai,dz); np.add.at(gs,di,-dz)
        if len(first_a): val+=.5*np.sum((s[first_a]/sig_a)**2); np.add.at(gs,first_a,s[first_a]/sig_a**2)
        if len(first_d): val+=.5*np.sum((s[first_d]/sig_d)**2); np.add.at(gs,first_d,s[first_d]/sig_d**2)
        if len(li):
            var=w2*ly; dd=s[lj]-s[li]; val+=.5*np.sum(dd*dd/var); gg=dd/var
            np.add.at(gs,lj,gg); np.add.at(gs,li,-gg)
        return val,g
    r=minimize(lambda x:fg(x),x0,jac=True,method="L-BFGS-B",options={"maxiter":500,"ftol":1e-9,"maxls":40})
    s=r.x[1:]; al={}; dl={}
    for k in a_keys: al[k[0]]=float(s[a_idx[k]])
    for k in d_keys: dl[k[0]]=float(s[d_idx[k]])
    return b.Fit(float(r.x[0]),w,al,dl,float(r.fun),bool(r.success),int(r.nit),n,rho)


if __name__=="__main__":
    b.fit_rate=fit_rate_fast
    b.fit_accuracy=fit_accuracy_fast
    b.main()
