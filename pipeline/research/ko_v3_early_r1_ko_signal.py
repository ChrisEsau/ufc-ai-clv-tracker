"""Chronological test of early-R1 KO history as incremental KO signal.

Research only. Uses raw master/round data and the validated Stage-2 KO V3
histories. No FSR traits and no MC mechanics changes.

This file is intentionally standalone so the early-finish hypothesis can be
validated without changing any simulator code.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.research import ko_v3_from_scratch_stage1 as s1
from pipeline.research import ko_v3_from_scratch_stage2 as s2

OUT = Path("data/research/ko_v3_early_r1_ko_signal")
THRESHOLDS = (60, 90, 120, 180, 300)
EARLY_PRIORS = (2.0, 5.0, 10.0, 20.0)
SEL_YEARS = tuple(range(2020, 2025))
CONF_YEARS = (2025, 2026)


def build_early_history(master: pd.DataFrame) -> pd.DataFrame:
    m = master.drop_duplicates("fight_id").copy()
    m["event_date"] = pd.to_datetime(m["date"]).dt.normalize()
    m["fight_id"] = m["fight_id"].astype(str)
    for c in ("r_id", "b_id", "winner_id"):
        m[c] = m[c].astype(str)
    method = m["method"].fillna("").astype(str).str.upper()
    is_ko = method.str.contains(r"KO/TKO|\bTKO\b|\bKO\b", regex=True)
    fr = pd.to_numeric(m["finish_round"], errors="coerce")
    mt = pd.to_numeric(m["match_time_sec"], errors="coerce")

    rows = []
    for corner, opp in (("r", "b"), ("b", "r")):
        x = pd.DataFrame({
            "event_date": m["event_date"],
            "fight_id": m["fight_id"],
            "fighter_id": m[f"{corner}_id"].astype(str),
            "opponent_id": m[f"{opp}_id"].astype(str),
            "won": m[f"{corner}_id"].astype(str).eq(m["winner_id"]),
            "is_ko": is_ko,
            "finish_round": fr,
            "match_time_sec": mt,
        })
        rows.append(x)
    ff = pd.concat(rows, ignore_index=True)
    ff["ko_win"] = ff["won"] & ff["is_ko"]
    ff["ko_loss"] = (~ff["won"]) & ff["is_ko"]
    for t in THRESHOLDS:
        early = ff["finish_round"].eq(1) & ff["match_time_sec"].le(t)
        ff[f"early{t}_win"] = ff["ko_win"] & early
        ff[f"early{t}_loss"] = ff["ko_loss"] & early

    fields = ["fights"] + [f"early{t}_{side}" for t in THRESHOLDS for side in ("win", "loss")]
    states = defaultdict(lambda: {f: 0.0 for f in fields})
    out = []
    for date, batch in ff.sort_values(["event_date", "fight_id", "fighter_id"]).groupby("event_date", sort=True):
        for r in batch.itertuples(index=False):
            st = states[str(r.fighter_id)]
            rec = {"event_date": date, "fight_id": str(r.fight_id), "fighter_id": str(r.fighter_id), "opponent_id": str(r.opponent_id)}
            rec.update({f"early_ewm95_{k}": v for k, v in st.items()})
            out.append(rec)
        for r in batch.itertuples(index=False):
            st = states[str(r.fighter_id)]
            st["fights"] = 0.95 * st["fights"] + 1.0
            for t in THRESHOLDS:
                for side in ("win", "loss"):
                    k = f"early{t}_{side}"
                    st[k] = 0.95 * st[k] + float(getattr(r, k))
    own = pd.DataFrame(out)
    histcols = [c for c in own.columns if c.startswith("early_ewm95_")]
    opp = own[["event_date", "fight_id", "fighter_id"] + histcols].rename(
        columns={"fighter_id": "opponent_id", **{c: f"opp_{c}" for c in histcols}}
    )
    return own.merge(opp, on=["event_date", "fight_id", "opponent_id"], how="left", validate="one_to_one")


def encoder_fit(train: pd.DataFrame, numeric: list[str]):
    arm = s1.Arm("x", tuple(numeric), ("division_cat",))
    enc = s1.NumericCategoricalEncoder(arm.numeric, arm.categorical).fit(train)
    return enc


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, numeric: list[str]) -> np.ndarray:
    enc = encoder_fit(train, numeric)
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    model.fit(enc.transform(train), train["ko_win"].astype(int).to_numpy())
    return model.predict_proba(enc.transform(test))[:, 1]


def metrics(g: pd.DataFrame, p: np.ndarray) -> dict:
    y = g["ko_win"].astype(int).to_numpy(); p = np.clip(np.asarray(p,float), 1e-9, 1-1e-9)
    return {
        "n": int(len(g)), "actual": float(y.mean()), "predicted": float(p.mean()),
        "auc": float(roc_auc_score(y,p)) if np.unique(y).size == 2 else np.nan,
        "brier": float(brier_score_loss(y,p)),
        "log_loss": float(log_loss(y,p,labels=[0,1])),
    }


def prep_v3(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,list[str]]:
    kd0 = float(train.kd_scored.sum()/train.sig_landed.sum())
    dr0 = float(train.direct_ko_win.sum()/train.sig_landed.sum())
    tr = s2.add_shrunken(train, kind="kd", decay=.95, strength=200., p0=kd0)
    te = s2.add_shrunken(test, kind="kd", decay=.95, strength=200., p0=kd0)
    for x in (tr,te):
        x.rename(columns={"shr_att":"kd_att","shr_def":"kd_def","shr_att_log_exp":"kd_att_exp","shr_def_log_exp":"kd_def_exp"}, inplace=True)
    tr = s2.add_shrunken(tr, kind="direct", decay=.95, strength=400., p0=dr0)
    te = s2.add_shrunken(te, kind="direct", decay=.95, strength=400., p0=dr0)
    for x in (tr,te):
        x.rename(columns={"shr_att":"direct_att","shr_def":"direct_def","shr_att_log_exp":"direct_att_exp","shr_def_log_exp":"direct_def_exp"}, inplace=True)
    cols=["attacker_age","defender_age","kd_att","kd_def","kd_att_exp","kd_def_exp","direct_att","direct_def","direct_att_exp","direct_def_exp"]
    return tr,te,cols


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw,_ = s1.load_raw_fighter_fights(ROUND_STATS_PATH, MASTER_PATH)
    frame = s1.build_matchup_frame(s1.build_prefight_states(raw))
    frame["ko_win"] = frame["ko_win"].astype(int)
    early = build_early_history(pd.read_parquet(MASTER_PATH))
    ecols=[c for c in early.columns if c.startswith("early_ewm95_") or c.startswith("opp_early_ewm95_")]
    frame=frame.merge(early[["event_date","fight_id","fighter_id"]+ecols], on=["event_date","fight_id","fighter_id"], how="left", validate="one_to_one")

    rows=[]
    years=sorted(y for y in frame.test_year.unique() if y>=2020)
    for year in years:
        train=frame[frame.event_date < pd.Timestamp(f"{year}-01-01")].copy()
        test=frame[frame.test_year.eq(year)].copy()
        if len(train)<1000 or len(test)<100: continue
        tr,te,v3cols=prep_v3(train,test)
        p=fit_predict(tr,te,v3cols); m=metrics(te,p); m.update(test_year=year,arm="v3_validated"); rows.append(m)
        for t in THRESHOLDS:
            for prior in EARLY_PRIORS:
                p0w=float(train[f"early_ewm95_early{t}_win"].sum()/max(train["early_ewm95_fights"].sum(),1e-9))
                p0l=float(train[f"opp_early_ewm95_early{t}_loss"].sum()/max(train["opp_early_ewm95_fights"].sum(),1e-9))
                for x in (tr,te):
                    x[f"early{t}_att"]=(x[f"early_ewm95_early{t}_win"]+p0w*prior)/(x["early_ewm95_fights"]+prior)
                    x[f"early{t}_def"]=(x[f"opp_early_ewm95_early{t}_loss"]+p0l*prior)/(x["opp_early_ewm95_fights"]+prior)
                cols=v3cols+[f"early{t}_att",f"early{t}_def"]
                arm=f"v3_plus_early{t}_s{int(prior)}"
                p=fit_predict(tr,te,cols); m=metrics(te,p); m.update(test_year=year,arm=arm); rows.append(m)
    byyear=pd.DataFrame(rows)
    pooled=[]
    for years_name, years_set in (("selection",SEL_YEARS),("confirmation",CONF_YEARS)):
        z=byyear[byyear.test_year.isin(years_set)].copy()
        for arm,g in z.groupby("arm"):
            w=g.n.to_numpy(float); pooled.append({
                "period":years_name,"arm":arm,"n":int(w.sum()),
                "log_loss":float(np.average(g.log_loss,weights=w)),
                "brier":float(np.average(g.brier,weights=w)),
                "auc":float(np.average(g.auc,weights=w)),
            })
    pooled=pd.DataFrame(pooled)
    sel=pooled[(pooled.period=="selection") & pooled.arm.ne("v3_validated")].sort_values("log_loss").iloc[0]
    base_sel=pooled[(pooled.period=="selection") & pooled.arm.eq("v3_validated")].iloc[0]
    conf=pooled[(pooled.period=="confirmation") & pooled.arm.eq(sel.arm)].iloc[0]
    base_conf=pooled[(pooled.period=="confirmation") & pooled.arm.eq("v3_validated")].iloc[0]
    report={
        "selected_on_2020_2024":str(sel.arm),
        "selection_delta_log_loss":float(sel.log_loss-base_sel.log_loss),
        "selection_delta_auc":float(sel.auc-base_sel.auc),
        "confirmation_delta_log_loss":float(conf.log_loss-base_conf.log_loss),
        "confirmation_delta_auc":float(conf.auc-base_conf.auc),
        "selected_confirmation":conf.to_dict(),
        "v3_confirmation":base_conf.to_dict(),
        "interpretation":"negative delta log loss and positive delta AUC favor early-R1 feature",
        "uses_fsr_traits":False,
        "changes_mc_mechanics":False,
    }
    byyear.to_csv(OUT/"by_year.csv",index=False); pooled.to_csv(OUT/"pooled.csv",index=False)
    (OUT/"report.json").write_text(json.dumps(report,indent=2,default=str)+"\n")
    print("KO V3 EARLY R1 KO SIGNAL")
    print(json.dumps(report,indent=2,default=str))
    print("\nSELECTION TOP 10")
    print(pooled[pooled.period.eq("selection")].sort_values("log_loss").head(10).to_string(index=False))
    print("\nCONFIRMATION SELECTED VS BASE")
    print(pooled[(pooled.period.eq("confirmation")) & pooled.arm.isin([str(sel.arm),"v3_validated"])].to_string(index=False))

if __name__ == "__main__":
    main()
