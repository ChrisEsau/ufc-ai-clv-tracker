#!/usr/bin/env python3
"""Research-only TD effectiveness Whole-History Rating prototype.

Fits time-varying latent takedown offense and defense states jointly over the
training history using the same matchup equation as FSR V3:

    logit(p_td) = beta_population + TD_OFF(attacker,t) - TD_DEF(defender,t)

Observation likelihood is binomial at the fighter-fight level. Consecutive
fighter states are linked by Gaussian random-walk penalties scaled by elapsed
time. This does not modify FSR or Brain.

Validation is strict frozen-cutoff OOS:
  * choose temporal drift w using train <= 2022 and validation 2023-24;
  * refit through 2024-12-31 with selected w;
  * predict 2025+ with the last pre-cutoff latent state only.

For an apples-to-apples comparator, the existing leakage-safe FSR V3 paired
TD-effectiveness replay is also frozen at the same cutoff and used to predict
the same 2025+ observations.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.paired_effectiveness import (
    build_effectiveness_fighter_fights,
    replay_paired_effectiveness,
    takedown_effectiveness_spec,
)

OUT = Path("data/diagnostics/td_effectiveness_whr_oos")
INNER_CUTOFF = pd.Timestamp("2023-01-01")
TEST_CUTOFF = pd.Timestamp("2025-01-01")
W_GRID = [0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.75]
EPS = 1e-9


@dataclass
class Fit:
    beta: float
    w: float
    off_latest: dict[str, float]
    def_latest: dict[str, float]
    objective: float
    converged: bool
    iterations: int
    n_states: int


def _make_states(fights: pd.DataFrame):
    """One offense and one defense state per fighter appearance date."""
    f = fights.sort_values(["event_date", "fight_id", "fighter_id"]).copy()
    off_keys = sorted(set(zip(f.fighter_id.astype(str), f.event_date)))
    # Every fighter-side is simultaneously opponent defense evidence, so the
    # same appearance grid is sufficient for defense states.
    def_keys = off_keys.copy()
    off_idx = {k: i for i, k in enumerate(off_keys)}
    n_off = len(off_keys)
    def_idx = {k: n_off + i for i, k in enumerate(def_keys)}

    # Lookup opponent appearance state on the same fight date.
    obs = []
    for r in f.itertuples(index=False):
        ak = (str(r.fighter_id), pd.Timestamp(r.event_date))
        dk = (str(r.opponent_id), pd.Timestamp(r.event_date))
        if dk not in def_idx:
            raise RuntimeError(f"missing reciprocal defense state {dk}")
        obs.append((off_idx[ak], def_idx[dk], float(r.landed), float(r.attempted)))

    def links(keys, idx):
        by = defaultdict(list)
        for fighter, dt in keys:
            by[fighter].append(pd.Timestamp(dt))
        out=[]
        for fighter, dates in by.items():
            dates=sorted(dates)
            for a,b in zip(dates[:-1], dates[1:]):
                years=max((b-a).days/365.25, 1.0/365.25)
                out.append((idx[(fighter,a)], idx[(fighter,b)], years))
        return out

    return off_keys, def_keys, off_idx, def_idx, obs, links(off_keys, off_idx), links(def_keys, def_idx)


def fit_whr(fights: pd.DataFrame, w: float, init: Fit | None = None) -> Fit:
    train=fights[fights.attempted>0].copy()
    if train.empty:
        raise RuntimeError("no takedown attempts in training data")
    off_keys, def_keys, off_idx, def_idx, obs, off_links, def_links = _make_states(train)
    n_states=len(off_keys)+len(def_keys)
    total_y=float(train.landed.sum()); total_n=float(train.attempted.sum())
    beta0=float(logit(np.clip(total_y/max(total_n,1.0), 1e-4, 1-1e-4)))
    x0=np.zeros(1+n_states,dtype=float); x0[0]=beta0

    sigma_off=0.35; sigma_def=0.50
    first_off={}
    for k in off_keys:
        first_off.setdefault(k[0], off_idx[k])
    first_def={}
    for k in def_keys:
        first_def.setdefault(k[0], def_idx[k])

    def fg(x):
        beta=float(x[0]); s=x[1:]
        val=0.0; grad=np.zeros_like(x)
        # weak intercept prior only for numerical stability
        val += 0.5*(beta/3.0)**2; grad[0]+=beta/9.0
        for oi,di,y,n in obs:
            if n<=0: continue
            z=beta+s[oi]-s[di]
            p=float(expit(z))
            val -= y*math.log(max(p,EPS)) + (n-y)*math.log(max(1-p,EPS))
            resid=n*p-y
            grad[0]+=resid; grad[1+oi]+=resid; grad[1+di]-=resid
        for i in first_off.values():
            v=s[i]; val+=0.5*(v/sigma_off)**2; grad[1+i]+=v/(sigma_off**2)
        for i in first_def.values():
            v=s[i]; val+=0.5*(v/sigma_def)**2; grad[1+i]+=v/(sigma_def**2)
        w2=max(w*w,1e-8)
        for i,j,years in off_links+def_links:
            var=w2*years
            d=s[j]-s[i]
            val+=0.5*d*d/var
            g=d/var
            grad[1+j]+=g; grad[1+i]-=g
        return val,grad

    res=minimize(lambda x: fg(x),x0,jac=True,method="L-BFGS-B",options={"maxiter":350,"ftol":1e-9,"maxls":30})
    s=res.x[1:]
    off_latest={}
    for fighter,dt in off_keys:
        off_latest[fighter]=float(s[off_idx[(fighter,dt)]])
    def_latest={}
    for fighter,dt in def_keys:
        def_latest[fighter]=float(s[def_idx[(fighter,dt)]])
    return Fit(float(res.x[0]),float(w),off_latest,def_latest,float(res.fun),bool(res.success),int(res.nit),int(n_states))


def predict(fit: Fit, frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    d=frame.copy()
    off=d.fighter_id.astype(str).map(fit.off_latest).fillna(0.0).astype(float)
    deff=d.opponent_id.astype(str).map(fit.def_latest).fillna(0.0).astype(float)
    p=expit(fit.beta+off-deff)
    d[f"{prefix}_off"]=off
    d[f"{prefix}_opp_def"]=deff
    d[f"{prefix}_p"]=p
    return d


def frozen_fsr_predictions(all_fights: pd.DataFrame, cutoff: pd.Timestamp, target: pd.DataFrame) -> pd.DataFrame:
    spec=takedown_effectiveness_spec(FSRV3Config())
    hist=replay_paired_effectiveness(all_fights[all_fights.event_date<cutoff].copy(),spec)
    off=hist[hist.trait.eq("takedown_offense")].sort_values(["event_date","fight_id"]).groupby("fighter_id").tail(1).set_index("fighter_id")["post_rating"].to_dict()
    deff=hist[hist.trait.eq("takedown_defense")].sort_values(["event_date","fight_id"]).groupby("fighter_id").tail(1).set_index("fighter_id")["post_rating"].to_dict()
    # Population baseline at cutoff from the last replay row.
    pop=float(hist.sort_values(["event_date","fight_id"])["population_baseline"].dropna().iloc[-1])
    beta=float(logit(np.clip(pop,1e-6,1-1e-6)))
    d=target.copy()
    o=d.fighter_id.astype(str).map(off).fillna(0.0).astype(float)
    dv=d.opponent_id.astype(str).map(deff).fillna(0.0).astype(float)
    d["fsr_off"]=o; d["fsr_opp_def"]=dv; d["fsr_p"]=expit(beta+o-dv)
    return d


def metrics(df: pd.DataFrame,pcol: str) -> dict:
    d=df[df.attempted>0].copy()
    p=np.clip(d[pcol].to_numpy(float),1e-9,1-1e-9)
    y=d.landed.to_numpy(float); n=d.attempted.to_numpy(float)
    ll=float(-(y*np.log(p)+(n-y)*np.log(1-p)).sum())
    attempts=float(n.sum())
    actual=float(y.sum()/attempts)
    pred=float((n*p).sum()/attempts)
    brier=float(((y/n-p)**2*n).sum()/attempts)
    return {"fighter_fights":int(len(d)),"attempts":int(attempts),"landed":int(y.sum()),"actual_rate":actual,"predicted_rate":pred,"attempt_log_loss":ll/attempts,"weighted_brier":brier}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=takedown_effectiveness_spec(FSRV3Config())
    fights=build_effectiveness_fighter_fights(spec,build_paired_rounds())
    fights=fights.sort_values(["event_date","fight_id","fighter_id"]).reset_index(drop=True)

    inner_train=fights[fights.event_date<INNER_CUTOFF]
    inner_val=fights[(fights.event_date>=INNER_CUTOFF)&(fights.event_date<TEST_CUTOFF)&(fights.attempted>0)]
    sweep=[]
    for w in W_GRID:
        fit=fit_whr(inner_train,w)
        pred=predict(fit,inner_val,"whr")
        m=metrics(pred,"whr_p")
        sweep.append({"w":w,"converged":fit.converged,"iterations":fit.iterations,"objective":fit.objective,"states":fit.n_states,**m})
        print("W",w,m,"converged",fit.converged,"iter",fit.iterations,flush=True)
    sw=pd.DataFrame(sweep).sort_values(["attempt_log_loss","weighted_brier"]).reset_index(drop=True)
    best_w=float(sw.iloc[0].w)

    train=fights[fights.event_date<TEST_CUTOFF]
    hold=fights[(fights.event_date>=TEST_CUTOFF)&(fights.attempted>0)].copy()
    fit=fit_whr(train,best_w)
    whr=predict(fit,hold,"whr")
    fsr=frozen_fsr_predictions(fights,TEST_CUTOFF,hold)
    cols=["event_date","fight_id","fighter_id","fighter_name","opponent_id","opponent_name","landed","attempted","fsr_off","fsr_opp_def","fsr_p"]
    merged=whr.merge(fsr[cols],on=["event_date","fight_id","fighter_id","fighter_name","opponent_id","opponent_name","landed","attempted"],how="left",validate="one_to_one")

    pop=float(train.landed.sum()/max(train.attempted.sum(),1.0))
    merged["population_p"]=pop
    summary={
        "architecture":"whole-history time-varying binomial TD offense/defense",
        "inner_train_before":str(INNER_CUTOFF.date()),
        "inner_validation":f"{INNER_CUTOFF.date()} through {(TEST_CUTOFF-pd.Timedelta(days=1)).date()}",
        "test_from":str(TEST_CUTOFF.date()),
        "selected_w":best_w,
        "final_fit_converged":fit.converged,
        "final_fit_iterations":fit.iterations,
        "final_fit_states":fit.n_states,
        "whr":metrics(merged,"whr_p"),
        "frozen_fsr_v3":metrics(merged,"fsr_p"),
        "population_only":metrics(merged,"population_p"),
        "note":"All test predictions are frozen at 2025-01-01; no 2025+ result updates either WHR or FSR comparator.",
    }
    sw.to_csv(OUT/"w_sweep.csv",index=False)
    merged.to_csv(OUT/"holdout_predictions.csv",index=False)
    pd.DataFrame([{"fighter_id":k,"td_off_whr":v,"td_def_whr":fit.def_latest.get(k,0.0)} for k,v in fit.off_latest.items()]).to_csv(OUT/"prefight_2025_ratings.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
