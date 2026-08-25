"""Research-only audit of whether recent fight dominance explains FSR market residuals.

Builds fight-level dominance from raw UFC round stats, using only information
available before each target fight. No FSR, simulator, market, or raw data are
modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import (
    build_two_way_market, choose_trait_columns, build_matchups, safe_logit,
)
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

ROUND_PATH = Path("data/fight_details/ufc_round_stats.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
MARKET_PATH = Path("data/market/historical_market_outcomes.parquet")

STAT_ALIASES = {
    "sig": ["sig_str_landed", "sig_strikes_landed", "sig_landed", "significant_strikes_landed"],
    "kd": ["kd", "knockdowns", "knockdown"],
    "td": ["td_landed", "takedowns_landed", "takedown_landed"],
    "sub": ["sub_att", "submission_attempts", "sub_attempts"],
    "ctrl": ["ctrl_sec", "control_seconds", "control_time_sec", "ctrl_seconds"],
}


def find_stat_col(df, aliases):
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def resolve_date_col(df):
    for c in ("date", "event_date", "fight_date"):
        if c in df.columns:
            return c
    raise RuntimeError("master has no recognized date column")


def winner_side(row):
    wid, rid, bid = str(row.get("winner_id")), str(row.get("r_id")), str(row.get("b_id"))
    if wid == rid:
        return "red"
    if wid == bid:
        return "blue"
    return None


def build_fight_dominance(rounds, master):
    required = {"fight_id", "fighter_id", "corner"}
    missing = required.difference(rounds.columns)
    if missing:
        raise RuntimeError(f"round stats missing required fighter-row columns {sorted(missing)}; columns={list(rounds.columns)}")

    stat_cols = {stat: find_stat_col(rounds, aliases) for stat, aliases in STAT_ALIASES.items()}
    usable_stats = [s for s, c in stat_cols.items() if c is not None]
    if len(usable_stats) < 2:
        raise RuntimeError(f"could not resolve enough fighter-row stat columns; resolved={stat_cols}; columns={list(rounds.columns)}")

    x = rounds.copy()
    x["fight_id"] = x["fight_id"].astype(str)
    x["fighter_id"] = x["fighter_id"].astype(str)
    x["corner_norm"] = x["corner"].astype(str).str.strip().str.lower()
    x = x[x["corner_norm"].isin(["red", "blue"])].copy()
    if x.empty:
        raise RuntimeError(f"round stats corner column has no red/blue rows; values={rounds['corner'].value_counts(dropna=False).head(20).to_dict()}")

    for stat in usable_stats:
        x[f"stat__{stat}"] = pd.to_numeric(x[stat_cols[stat]], errors="coerce").fillna(0.0)

    agg_cols = [f"stat__{s}" for s in usable_stats]
    fighter_fight = (
        x.groupby(["fight_id", "corner_norm", "fighter_id"], as_index=False)[agg_cols]
        .sum()
    )

    dup = fighter_fight.duplicated(["fight_id", "corner_norm"], keep=False)
    if dup.any():
        bad = fighter_fight.loc[dup, ["fight_id", "corner_norm", "fighter_id"]].head(20).to_dict("records")
        raise RuntimeError(f"multiple fighters found for one fight/corner after aggregation; examples={bad}")

    wide = fighter_fight.pivot(index="fight_id", columns="corner_norm")
    wide.columns = [f"{corner}_{field.replace('stat__','')}" for field, corner in wide.columns]
    wide = wide.reset_index()
    complete = [c for c in ["red_fighter_id", "blue_fighter_id"] if c in wide.columns]
    if len(complete) != 2:
        raise RuntimeError(f"failed to pivot fighter-row stats into red/blue fight rows; columns={list(wide.columns)}")
    wide = wide[wide["red_fighter_id"].notna() & wide["blue_fighter_id"].notna()].copy()

    m = master.copy(); m["fight_id"] = m["fight_id"].astype(str)
    date_col = resolve_date_col(m)
    keep = [c for c in ["fight_id",date_col,"r_id","b_id","winner_id","method","weight_class","division"] if c in m.columns]
    f = wide.merge(m[keep].drop_duplicates("fight_id"), on="fight_id", how="left")
    f["fight_date"] = pd.to_datetime(f[date_col], errors="coerce").dt.normalize()
    div_col = "weight_class" if "weight_class" in f.columns else ("division" if "division" in f.columns else None)
    f["division_key"] = f[div_col].astype(str) if div_col else "ALL"
    f["era"] = (f["fight_date"].dt.year // 3 * 3).astype("Int64").astype(str)

    # Validate round-stats fighter identities against master where both are present.
    if {"r_id", "b_id"}.issubset(f.columns):
        r_ok = f["r_id"].isna() | (f["r_id"].astype(str) == f["red_fighter_id"].astype(str))
        b_ok = f["b_id"].isna() | (f["b_id"].astype(str) == f["blue_fighter_id"].astype(str))
        mismatch = ~(r_ok & b_ok)
        if mismatch.any():
            ex = f.loc[mismatch, ["fight_id","red_fighter_id","blue_fighter_id","r_id","b_id"]].head(10).to_dict("records")
            raise RuntimeError(f"round/master fighter identity mismatch; examples={ex}")

    rows = []
    weights = {"sig":1.0,"kd":2.0,"td":0.6,"sub":0.8,"ctrl":0.004}
    for _, r in f.iterrows():
        for side, opp in (("red","blue"),("blue","red")):
            score = 0.0
            parts = {}
            for stat in usable_stats:
                a = float(r.get(f"{side}_{stat}",0.0)); b = float(r.get(f"{opp}_{stat}",0.0))
                margin = (a-b)/(a+b+1.0)
                parts[f"margin_{stat}"] = margin
                score += weights.get(stat,1.0)*margin
            ws = winner_side(r)
            method = str(r.get("method","")).lower()
            is_finish = ws == side and "decision" not in method and method not in {"","nan","none"}
            finish_bonus = 0.75 if is_finish else 0.0
            rows.append({
                "fight_id":r["fight_id"], "fight_date":r["fight_date"],
                "fighter_id":str(r[f"{side}_fighter_id"]),
                "opponent_id":str(r[f"{opp}_fighter_id"]),
                "division_key":r["division_key"], "era":r["era"],
                "won":int(ws==side) if ws else np.nan,
                "finish_win":int(is_finish), "raw_dominance":score+finish_bonus,
                **parts,
            })
    d = pd.DataFrame(rows)
    grp = d.groupby(["division_key","era"])["raw_dominance"]
    mu = grp.transform("mean"); sd = grp.transform("std").replace(0,np.nan)
    d["dominance_z"] = ((d["raw_dominance"]-mu)/sd).fillna(0.0)
    return d, usable_stats, stat_cols


def add_prefight_dominance(matchups, dominance):
    by_fighter = {fid:g.sort_values(["fight_date","fight_id"]).reset_index(drop=True) for fid,g in dominance.groupby("fighter_id")}
    rows=[]
    for _,r in matchups.iterrows():
        rec=r.to_dict(); date=pd.Timestamp(r["fight_date"])
        for label,fid in (("fav",str(r["favorite_id"])),("dog",str(r["underdog_id"]))):
            h=by_fighter.get(fid,pd.DataFrame())
            if not h.empty: h=h[h["fight_date"]<date]
            vals=h["dominance_z"].to_numpy(float) if not h.empty else np.array([])
            rec[f"{label}_dom_last1"] = vals[-1] if len(vals)>=1 else np.nan
            rec[f"{label}_dom_last3"] = float(np.mean(vals[-3:])) if len(vals)>=1 else np.nan
            if len(vals):
                w=np.array([0.5**i for i in range(len(vals)-1,-1,-1)],float); w=w/w.sum()
                rec[f"{label}_dom_ewm"] = float(np.sum(vals*w))
            else: rec[f"{label}_dom_ewm"] = np.nan
        for k in ("dom_last1","dom_last3","dom_ewm"):
            a,b=rec[f"fav_{k}"],rec[f"dog_{k}"]
            rec[f"delta_{k}"] = a-b if pd.notna(a) and pd.notna(b) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def fit_market(train,test,features,label):
    tr=train.dropna(subset=features+["market_favorite_fair_p"]); te=test.dropna(subset=features+["market_favorite_fair_p"])
    model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
    ytr=safe_logit(tr["market_favorite_fair_p"]); yte=safe_logit(te["market_favorite_fair_p"])
    model.fit(tr[features],ytr); p=model.predict(te[features]); pp=1/(1+np.exp(-p))
    return {"model":label,"features":len(features),"train_n":len(tr),"test_n":len(te),
            "test_r2_logit":r2_score(yte,p),"test_rmse_logit":mean_squared_error(yte,p)**0.5,
            "mean_abs_residual_pp":float(np.mean(np.abs(100*(te["market_favorite_fair_p"].to_numpy()-pp))))}


def fit_winner(train,test,features,label):
    tr=train.dropna(subset=features+["favorite_won"]); te=test.dropna(subset=features+["favorite_won"])
    pipe=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=.25,max_iter=2000))])
    pipe.fit(tr[features],tr["favorite_won"].astype(int)); p=pipe.predict_proba(te[features])[:,1]; y=te["favorite_won"].astype(int).to_numpy()
    return {"model":label,"test_n":len(te),"auc":roc_auc_score(y,p),"brier":brier_score_loss(y,p),"logloss":log_loss(y,p)}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--out-dir",type=Path,required=True); args=ap.parse_args()
    rounds=pd.read_parquet(ROUND_PATH); master=pd.read_parquet(MASTER_PATH); fsr=pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    market=build_two_way_market(MARKET_PATH); traits=choose_trait_columns(fsr)
    matchups=build_matchups(market,fsr,master,traits)
    dominance,usable_stats,resolved=build_fight_dominance(rounds,master)
    frame=add_prefight_dominance(matchups,dominance).sort_values(["fight_date","fight_id"]).reset_index(drop=True)
    cut=int(len(frame)*.70); train=frame.iloc[:cut].copy(); test=frame.iloc[cut:].copy()
    fsr_features=[f"delta__{c}" for c in traits]
    dom_features=["delta_dom_last1","delta_dom_last3","delta_dom_ewm"]
    results=[]
    results.append(fit_market(train,test,fsr_features,"fsr_only"))
    for d in dom_features:
        results.append(fit_market(train,test,fsr_features+[d],f"fsr_plus_{d}"))
    results.append(fit_market(train,test,fsr_features+dom_features,"fsr_plus_all_dominance"))
    mr=pd.DataFrame(results).sort_values("test_rmse_logit")

    wr=[]
    wr.append(fit_winner(train,test,fsr_features,"fsr_only"))
    wr.append(fit_winner(train,test,fsr_features+dom_features,"fsr_plus_all_dominance"))
    wr=pd.DataFrame(wr)

    baseline=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
    tr=train.dropna(subset=fsr_features+["market_favorite_fair_p"]); te=test.dropna(subset=fsr_features+["market_favorite_fair_p"])
    baseline.fit(tr[fsr_features],safe_logit(tr["market_favorite_fair_p"])); pred=baseline.predict(te[fsr_features])
    te=te.copy(); te["fsr_residual_logit"]=safe_logit(te["market_favorite_fair_p"])-pred
    corr=[]
    for c in dom_features:
        z=te[[c,"fsr_residual_logit"]].dropna(); corr.append({"feature":c,"n":len(z),"corr_with_fsr_residual_logit":z[c].corr(z["fsr_residual_logit"])})
    corr=pd.DataFrame(corr)

    args.out_dir.mkdir(parents=True,exist_ok=True)
    dominance.to_csv(args.out_dir/"fight_fighter_dominance.csv",index=False)
    mr.to_csv(args.out_dir/"market_incremental_models.csv",index=False)
    wr.to_csv(args.out_dir/"winner_incremental_models.csv",index=False)
    corr.to_csv(args.out_dir/"dominance_residual_correlations.csv",index=False)
    pd.DataFrame([{"usable_stats":",".join(usable_stats),"resolved_columns":str(resolved),"joined_fights":len(frame),"cut_date":str(test["fight_date"].min().date())}]).to_csv(args.out_dir/"audit_metadata.csv",index=False)

    print("FSR DOMINANCE RESIDUAL AUDIT")
    print(f"joined fights={len(frame)} | dominance fighter-fights={len(dominance)} | stats={usable_stats} | cut={test['fight_date'].min().date()}")
    print("\nMARKET INCREMENTAL MODELS")
    print(mr.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nACTUAL WINNER INCREMENTAL MODELS")
    print(wr.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nDOMINANCE VS BASELINE FSR MARKET RESIDUAL")
    print(corr.to_string(index=False,float_format=lambda x:f"{x:.5f}"))

if __name__=="__main__": main()
