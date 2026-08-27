"""Robust empirical-Bayes KO prior evaluation.

Research only; production untouched.

Goal: preserve the calibrated S400 population prior for ordinary histories while
allowing statistically incompatible high/low fighter histories to escape toward
a weak prior. This is a robust MAP-style mixture prior, evaluated chronologically.

For an observed KO count k over significant-strike exposure n, use a mixture of
Beta priors with the same chronological population mean p0:
  strong: Beta(S*p0, S*(1-p0)), S=400
  weak:   Beta(W*p0, W*(1-p0)), W in grid
with prior mixture weight q on strong. Posterior component weights are determined
by the beta-binomial marginal likelihood of k|n. Posterior mean is the mixture of
component posterior means. Thus extreme evidence can automatically downweight the
strong prior without fighter-specific rules.

Attacker KO/Sig and defender KO-loss/Sig are treated separately, then their
population-centered logit deviations are combined exactly as in the validated
S400 architecture.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp
from sklearn.metrics import roc_auc_score
from pipeline.research import ko_v3_from_scratch_stage1 as s1

OUT=Path('data/research/ko_v3_robust_adaptive_prior')
STRONG=400.0
WEAKS=(5.0,10.0,25.0,50.0,100.0)
QSTRONG=(0.50,0.70,0.85,0.95,0.99)
EPS=1e-9

def clip(p): return np.clip(np.asarray(p,float),EPS,1-EPS)
def logit(p):
    p=clip(p); return np.log(p/(1-p))
def sigmoid(z): return 1/(1+np.exp(-np.clip(np.asarray(z,float),-30,30)))

def add_p0(frame):
    d=frame.copy(); d['event_date']=pd.to_datetime(d.event_date).dt.normalize()
    daily=(d[d.sig_landed>0].groupby('event_date',as_index=False).agg(k=('ko_win','sum'),n=('sig_landed','sum')).sort_values('event_date'))
    daily['pk']=daily.k.cumsum().shift(1,fill_value=0.0); daily['pn']=daily.n.cumsum().shift(1,fill_value=0.0)
    daily['p0']=np.where(daily.pn>0,daily.pk/daily.pn,np.nan)
    return d.merge(daily[['event_date','p0']],on='event_date',how='left',validate='many_to_one')

def beta_log_marg(k,n,p0,S):
    a=np.maximum(S*p0,1e-8); b=np.maximum(S*(1-p0),1e-8)
    return betaln(k+a,n-k+b)-betaln(a,b)

def robust_post(k,n,p0,weak,q):
    k=np.asarray(k,float); n=np.asarray(n,float); p0=np.asarray(p0,float)
    lm_s=beta_log_marg(k,n,p0,STRONG)+math.log(q)
    lm_w=beta_log_marg(k,n,p0,weak)+math.log(1-q)
    den=np.logaddexp(lm_s,lm_w)
    ws=np.exp(lm_s-den); ww=1-ws
    ps=(k+STRONG*p0)/(n+STRONG)
    pw=(k+weak*p0)/(n+weak)
    return ws*ps+ww*pw,ws

def fixed(k,n,p0,S=400.0): return (np.asarray(k,float)+S*np.asarray(p0,float))/(np.asarray(n,float)+S)

def hazards(df,weak=None,q=None):
    p0=df.p0.to_numpy(float)
    ak=df.prior_ko_wins.to_numpy(float); an=df.prior_sig_landed.to_numpy(float)
    dk=df.opp_prior_ko_losses.to_numpy(float); dn=df.opp_prior_sig_absorbed.to_numpy(float)
    if weak is None:
        pa=fixed(ak,an,p0); pdv=fixed(dk,dn,p0); wa=np.ones(len(df)); wd=np.ones(len(df))
    else:
        pa,wa=robust_post(ak,an,p0,weak,q); pdv,wd=robust_post(dk,dn,p0,weak,q)
    both=sigmoid(logit(p0)+(logit(pa)-logit(p0))+(logit(pdv)-logit(p0)))
    return both,pa,pdv,wa,wd

def metrics(df,h):
    y=df.ko_win.to_numpy(int); n=df.sig_landed.to_numpy(float); hc=clip(h)
    ll=-float(np.sum(y*np.log(hc)+(n-y)*np.log(1-hc))/np.sum(n))
    exp=float(np.sum(h*n)); act=float(np.sum(y)); auc=float(roc_auc_score(y,h))
    tmp=df[['fight_id','ko_win']].copy(); tmp['h']=h; cs=[]
    for _,g in tmp.groupby('fight_id'):
        if len(g)==2 and int(g.ko_win.sum())==1:
            a=float(g.loc[g.ko_win==1,'h'].iloc[0]); b=float(g.loc[g.ko_win==0,'h'].iloc[0]); cs.append(1 if a>b else .5 if a==b else 0)
    return {'strike_ll':ll,'eo':exp/act,'auc':auc,'correct':float(np.mean(cs)),'zeros':int(np.sum(np.asarray(h)<=0))}

def period(f,y0,y1): return f[f.event_date.dt.year.between(y0,y1)&(f.sig_landed>0)&f.p0.notna()].copy().reset_index(drop=True)

def main():
    ff,_=s1.load_raw_fighter_fights(); frame=add_p0(s1.build_matchup_frame(s1.build_prefight_states(ff)))
    sel=period(frame,2020,2024); conf=period(frame,2025,2026)
    base_h,*_=hazards(sel); base_sel=metrics(sel,base_h)
    base_hc,*_=hazards(conf); base_conf=metrics(conf,base_hc)
    rows=[]
    for w in WEAKS:
        for q in QSTRONG:
            hs,pa,pd,wa,wd=hazards(sel,w,q); m=metrics(sel,hs)
            rows.append({'weak':w,'q':q,**m,'mean_strong_weight_att':float(np.mean(wa)),'mean_strong_weight_def':float(np.mean(wd))})
    best=min(rows,key=lambda r:r['strike_ll'])
    hc,pa,pd,wa,wd=hazards(conf,best['weak'],best['q']); mc=metrics(conf,hc)
    # diagnostic examples by name at their latest eligible prefight row in confirmation
    examples={}
    names=['Alessandro Costa','Manel Kape']
    for name in names:
        g=conf[conf.fighter_name.astype(str).eq(name)].sort_values('event_date')
        if len(g):
            r=g.iloc[-1:]; h0,pa0,pd0,_,_=hazards(r); hr,par,pdr,war,wdr=hazards(r,best['weak'],best['q'])
            examples[name]={'date':str(r.event_date.iloc[0].date()),'prior_ko_wins':float(r.prior_ko_wins.iloc[0]),'prior_sig_landed':float(r.prior_sig_landed.iloc[0]),'p0':float(r.p0.iloc[0]),'fixed_s400_att':float(pa0[0]),'robust_att':float(par[0]),'strong_weight_att':float(war[0]),'fixed_matchup':float(h0[0]),'robust_matchup':float(hr[0])}
    out={'baseline_s400_selection':base_sel,'baseline_s400_confirmation':base_conf,'selection_grid':rows,'selected':{'weak':best['weak'],'q':best['q'],'selection':best,'confirmation':mc},'examples':examples,'production_changed':False}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'results.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    print('KO V3 ROBUST ADAPTIVE PRIOR')
    print('='*90)
    print(f"baseline S400 sel : LL={base_sel['strike_ll']:.8f} E/O={base_sel['eo']:.3f} AUC={base_sel['auc']:.4f} correct={base_sel['correct']:.4f}")
    print(f"baseline S400 conf: LL={base_conf['strike_ll']:.8f} E/O={base_conf['eo']:.3f} AUC={base_conf['auc']:.4f} correct={base_conf['correct']:.4f}")
    print(f"selected weak={best['weak']:.0f} qStrong={best['q']:.2f}")
    print(f"robust sel         : LL={best['strike_ll']:.8f} E/O={best['eo']:.3f} AUC={best['auc']:.4f} correct={best['correct']:.4f}")
    print(f"robust conf        : LL={mc['strike_ll']:.8f} E/O={mc['eo']:.3f} AUC={mc['auc']:.4f} correct={mc['correct']:.4f}")
    print('examples:')
    for n,v in examples.items(): print(n, json.dumps(v,sort_keys=True))
    print(f'Wrote {OUT / "results.json"}')
if __name__=='__main__': main()
