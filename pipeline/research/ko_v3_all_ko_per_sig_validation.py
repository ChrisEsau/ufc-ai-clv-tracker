"""Validate simple all-KO-per-significant-strike matchup hazard.

Research only. No simulator mechanics are changed.

Literal candidate requested for testing:
    attacker_rate = prior all KO/TKO wins / prior sig strikes landed
    defender_rate = opponent prior all KO/TKO losses / prior sig strikes absorbed
    combined_per_landed = 1 - (1-attacker_rate)*(1-defender_rate)
    fight_ko_probability = 1 - (1-combined_per_landed)**current_fight_sig_landed

All histories are same-date delayed by Stage 1. No shrinkage and no fitted logit are
used in the primary candidate. Cumulative and EWM95 variants are reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.research import ko_v3_from_scratch_stage1 as s1

OUT = Path("data/research/ko_v3_all_ko_per_sig_validation")
SELECTION_YEARS = tuple(range(2020, 2025))
CONFIRMATION_YEARS = (2025, 2026)


def safe_rate(k, n):
    k=np.asarray(k,float); n=np.asarray(n,float)
    return np.divide(k,n,out=np.full_like(k,np.nan,dtype=float),where=n>0)


def combine(a,d):
    a=np.clip(np.asarray(a,float),0,1); d=np.clip(np.asarray(d,float),0,1)
    return 1-(1-a)*(1-d)


def metrics(g: pd.DataFrame, p_fight: np.ndarray) -> dict:
    y=g["ko_win"].astype(int).to_numpy()
    p=np.clip(np.asarray(p_fight,float),1e-9,1-1e-9)
    return {
        "rows": int(len(g)),
        "ko_wins": int(y.sum()),
        "actual_ko_win_rate": float(y.mean()),
        "mean_predicted_ko_probability": float(p.mean()),
        "auc": float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan,
        "brier": float(brier_score_loss(y,p)),
        "log_loss": float(log_loss(y,p,labels=[0,1])),
        "top_decile_precision": float(y[p >= np.quantile(p,0.9)].mean()) if len(g) else np.nan,
        "extreme_false_positives_ge_50": int(((p>=0.50)&(y==0)).sum()),
        "mean_p_actual_ko_winners": float(p[y==1].mean()) if y.sum() else np.nan,
        "mean_p_non_ko": float(p[y==0].mean()) if (y==0).sum() else np.nan,
    }


def add_candidate(frame: pd.DataFrame, prefix: str, label: str) -> pd.DataFrame:
    x=frame.copy()
    att=safe_rate(x[f"{prefix}ko_wins"],x[f"{prefix}sig_landed"])
    deff=safe_rate(x[f"opp_{prefix}ko_losses"],x[f"opp_{prefix}sig_absorbed"])
    # Literal no-shrink rule requires observed history on both sides.
    valid=np.isfinite(att)&np.isfinite(deff)&x["sig_landed"].gt(0).to_numpy()
    x=x.loc[valid].copy()
    att=att[valid]; deff=deff[valid]
    p_sig=combine(att,deff)
    n=x["sig_landed"].to_numpy(float)
    p_fight=1-np.power(1-p_sig,n)
    x[f"{label}_att_ko_per_sig"]=att
    x[f"{label}_def_ko_loss_per_sig"]=deff
    x[f"{label}_combined_per_sig"]=p_sig
    x[f"{label}_p_fight"]=p_fight
    return x


def population_baseline(frame: pd.DataFrame, years) -> dict:
    g=frame[frame.test_year.isin(years)&frame.sig_landed.gt(0)].copy()
    # Fit population all-KO-per-landed only on data before first scored year.
    cutoff=pd.Timestamp(f"{min(years)}-01-01")
    train=frame[(frame.event_date<cutoff)&frame.sig_landed.gt(0)].copy()
    p0=float(train.ko_win.sum()/train.sig_landed.sum())
    p=1-np.power(1-p0,g.sig_landed.to_numpy(float))
    out=metrics(g,p); out["population_per_sig"]=p0
    return out


def score_variant(x: pd.DataFrame, pcol: str, years) -> dict:
    g=x[x.test_year.isin(years)].copy()
    return metrics(g,g[pcol].to_numpy(float))


def correct_side(frame: pd.DataFrame, pcol: str, years) -> dict:
    g=frame[frame.test_year.isin(years)].copy()
    rows=[]
    for fid,b in g.groupby("fight_id"):
        if len(b)!=2 or not bool(b.ko_win.any()):
            continue
        winner=b[b.ko_win].iloc[0]
        loser=b[~b.ko_win].iloc[0]
        rows.append(float(winner[pcol]) > float(loser[pcol]))
    return {"ko_fights":len(rows),"correct_side_rate":float(np.mean(rows)) if rows else np.nan}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ff,audit=s1.load_raw_fighter_fights(ROUND_STATS_PATH,MASTER_PATH)
    states=s1.build_prefight_states(ff)
    frame=s1.build_matchup_frame(states)

    cum=add_candidate(frame,"prior_","cum")
    ewm=add_candidate(frame,"ewm95_","ewm95")

    report={
        "study":"all KO/TKO wins per sig landed + opponent all KO/TKO losses per sig absorbed",
        "same_date_delayed":True,
        "uses_shrinkage":False,
        "uses_fitted_logit":False,
        "changes_mc":False,
        "raw_audit":audit,
        "selection_years":list(SELECTION_YEARS),
        "confirmation_years":list(CONFIRMATION_YEARS),
        "cumulative":{
            "selection":score_variant(cum,"cum_p_fight",SELECTION_YEARS),
            "confirmation":score_variant(cum,"cum_p_fight",CONFIRMATION_YEARS),
            "selection_correct_side":correct_side(cum,"cum_p_fight",SELECTION_YEARS),
            "confirmation_correct_side":correct_side(cum,"cum_p_fight",CONFIRMATION_YEARS),
        },
        "ewm95":{
            "selection":score_variant(ewm,"ewm95_p_fight",SELECTION_YEARS),
            "confirmation":score_variant(ewm,"ewm95_p_fight",CONFIRMATION_YEARS),
            "selection_correct_side":correct_side(ewm,"ewm95_p_fight",SELECTION_YEARS),
            "confirmation_correct_side":correct_side(ewm,"ewm95_p_fight",CONFIRMATION_YEARS),
        },
        "population":{
            "selection":population_baseline(frame,SELECTION_YEARS),
            "confirmation":population_baseline(frame,CONFIRMATION_YEARS),
        },
    }
    (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    cum[["event_date","fight_id","fighter_id","fighter_name","opponent_id","ko_win","sig_landed","cum_att_ko_per_sig","cum_def_ko_loss_per_sig","cum_combined_per_sig","cum_p_fight"]].to_csv(OUT/"cumulative_predictions.csv",index=False)
    ewm[["event_date","fight_id","fighter_id","fighter_name","opponent_id","ko_win","sig_landed","ewm95_att_ko_per_sig","ewm95_def_ko_loss_per_sig","ewm95_combined_per_sig","ewm95_p_fight"]].to_csv(OUT/"ewm95_predictions.csv",index=False)
    print("KO V3 ALL-KO PER-SIG VALIDATION")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
