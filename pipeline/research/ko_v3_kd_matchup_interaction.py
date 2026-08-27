"""Leakage-free KD-create x KD-allow matchup interaction study.

Research only. Reads raw UFC round stats through the KO V3 Stage-1 loader.
No FSR traits and no MC changes.

Question: does attacker KD creation x defender KD susceptibility add future
KO/TKO-win signal beyond the same two additive terms?

Hyperparameters are selected on 2020-2024. 2025-2026 is confirmation only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.research import ko_v3_from_scratch_stage1 as s1

OUT = Path("data/research/ko_v3_kd_matchup_interaction")
DECAYS = (0.50, 0.80, 0.90, 0.95)
PRIORS = (100.0, 200.0, 400.0)
SEL_YEARS = tuple(range(2020, 2025))
CONF_YEARS = (2025, 2026)

# Workflow trigger marker: 2026-08-27


def era_bucket(d: pd.Series) -> pd.Series:
    y = pd.to_datetime(d).dt.year
    return pd.cut(y, [-np.inf, 2014, 2019, 2023, np.inf], labels=["pre2015", "2015_19", "2020_23", "2024plus"]).astype(str)


def history_cols(decay: float) -> tuple[str, str, str, str]:
    tag = f"ewm{int(round(decay*100)):02d}"
    return (f"{tag}_kd_scored", f"{tag}_sig_landed", f"opp_{tag}_kd_absorbed", f"opp_{tag}_sig_absorbed")


def group_priors(train: pd.DataFrame) -> pd.DataFrame:
    z = train.copy()
    z["era"] = era_bucket(z.event_date)
    g = z.groupby(["division_cat", "era"], observed=True).agg(
        att_kd=("kd_scored", "sum"), att_exp=("sig_landed", "sum"),
        def_kd=("kd_absorbed", "sum"), def_exp=("sig_absorbed", "sum"),
    ).reset_index()
    glob_att = float(z.kd_scored.sum()/max(z.sig_landed.sum(), 1e-9))
    glob_def = float(z.kd_absorbed.sum()/max(z.sig_absorbed.sum(), 1e-9))
    g["att_prior"] = np.where(g.att_exp.gt(0), g.att_kd/g.att_exp, glob_att)
    g["def_prior"] = np.where(g.def_exp.gt(0), g.def_kd/g.def_exp, glob_def)
    return g[["division_cat", "era", "att_prior", "def_prior"]]


def add_features(train: pd.DataFrame, test: pd.DataFrame, decay: float, strength: float):
    pri = group_priors(train)
    ac, ae, dc, de = history_cols(decay)
    outs=[]
    for src in (train, test):
        x=src.copy(); x["era"] = era_bucket(x.event_date)
        x=x.merge(pri,on=["division_cat","era"],how="left",validate="many_to_one")
        ga=float(train.kd_scored.sum()/max(train.sig_landed.sum(),1e-9)); gd=float(train.kd_absorbed.sum()/max(train.sig_absorbed.sum(),1e-9))
        x["att_prior"]=x.att_prior.fillna(ga); x["def_prior"]=x.def_prior.fillna(gd)
        x["att_kd_create"]=(x[ac].astype(float)+strength*x.att_prior)/(x[ae].astype(float)+strength)
        x["def_kd_allow"]=(x[dc].astype(float)+strength*x.def_prior)/(x[de].astype(float)+strength)
        x["kd_interaction"]=(x.att_kd_create-x.att_prior)*(x.def_kd_allow-x.def_prior)
        x["att_log_exp"]=np.log1p(x[ae].astype(float)); x["def_log_exp"]=np.log1p(x[de].astype(float))
        outs.append(x)
    return outs


def fit_predict(train, test, cols):
    arm=s1.Arm("tmp",tuple(cols),("division_cat",))
    enc=s1.NumericCategoricalEncoder(arm.numeric,arm.categorical).fit(train)
    model=LogisticRegression(C=1.0,max_iter=5000,solver="lbfgs")
    model.fit(enc.transform(train),train.ko_win.astype(int).to_numpy())
    return model.predict_proba(enc.transform(test))[:,1], model, enc


def metrics(df,p):
    y=df.ko_win.astype(int).to_numpy(); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    return dict(n=int(len(y)),actual=float(y.mean()),predicted=float(p.mean()),
        auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),
        log_loss=float(log_loss(y,p,labels=[0,1])))


def calibration(df,p,bins=10):
    z=pd.DataFrame({"y":df.ko_win.astype(int).to_numpy(),"p":p})
    z["bin"]=pd.qcut(z.p,q=min(bins,z.p.nunique()),duplicates="drop")
    return z.groupby("bin",observed=True).agg(n=("y","size"),predicted=("p","mean"),actual=("y","mean")).reset_index().astype({"bin":"str"})


def side_discrimination(frame,p):
    z=frame[["fight_id","fighter_id","ko_win"]].copy(); z["p"]=p
    ko=z.groupby("fight_id").filter(lambda g: g.ko_win.sum()==1)
    if ko.empty: return float("nan")
    wins=0; total=0
    for _,g in ko.groupby("fight_id"):
        w=g[g.ko_win.astype(bool)]; l=g[~g.ko_win.astype(bool)]
        if len(w)==1 and len(l)==1:
            wins += int(float(w.p.iloc[0]) > float(l.p.iloc[0])); total += 1
    return wins/max(total,1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ff,audit=s1.load_raw_fighter_fights(ROUND_STATS_PATH,MASTER_PATH)
    states=s1.build_prefight_states(ff)
    frame=s1.build_matchup_frame(states).copy()
    frame["ko_win"]=frame.ko_win.astype(int)

    rows=[]
    for decay in DECAYS:
      for strength in PRIORS:
        for year in sorted(y for y in frame.test_year.unique() if y>=2020):
            train=frame[frame.event_date < pd.Timestamp(f"{year}-01-01")].copy(); test=frame[frame.test_year.eq(year)].copy()
            if len(train)<1000 or len(test)<100: continue
            tr,te=add_features(train,test,decay,strength)
            base=["att_kd_create","def_kd_allow","att_log_exp","def_log_exp","attacker_age","defender_age"]
            for arm,cols in (("additive",base),("interaction",base+["kd_interaction"])):
                p,_,_=fit_predict(tr,te,cols); m=metrics(te,p); m.update(test_year=int(year),arm=arm,decay=decay,prior_strength=strength,correct_side_ko=side_discrimination(te,p)); rows.append(m)
    by=pd.DataFrame(rows); by.to_csv(OUT/"by_year.csv",index=False)

    pooled=[]
    for period,ys in (("selection",SEL_YEARS),("confirmation",CONF_YEARS)):
      for (decay,strength,arm),g in by[by.test_year.isin(ys)].groupby(["decay","prior_strength","arm"]):
        w=g.n.to_numpy(float)
        pooled.append(dict(period=period,decay=float(decay),prior_strength=float(strength),arm=arm,n=int(w.sum()),
          auc=float(np.average(g.auc,weights=w)),brier=float(np.average(g.brier,weights=w)),log_loss=float(np.average(g.log_loss,weights=w)),
          correct_side_ko=float(np.average(g.correct_side_ko,weights=w))))
    pooled=pd.DataFrame(pooled); pooled.to_csv(OUT/"pooled.csv",index=False)

    sel_i=pooled[(pooled.period=="selection")&(pooled.arm=="interaction")].sort_values("log_loss").iloc[0]
    d=float(sel_i.decay); s=float(sel_i.prior_strength)
    sel_a=pooled[(pooled.period=="selection")&(pooled.arm=="additive")&(pooled.decay==d)&(pooled.prior_strength==s)].iloc[0]
    conf_i=pooled[(pooled.period=="confirmation")&(pooled.arm=="interaction")&(pooled.decay==d)&(pooled.prior_strength==s)].iloc[0]
    conf_a=pooled[(pooled.period=="confirmation")&(pooled.arm=="additive")&(pooled.decay==d)&(pooled.prior_strength==s)].iloc[0]

    train=frame[frame.event_date < pd.Timestamp("2025-01-01")].copy(); test=frame[frame.test_year.isin(CONF_YEARS)].copy()
    tr,te=add_features(train,test,d,s)
    base=["att_kd_create","def_kd_allow","att_log_exp","def_log_exp","attacker_age","defender_age"]
    p_a,_,_=fit_predict(tr,te,base); p_i,_,_=fit_predict(tr,te,base+["kd_interaction"])
    calibration(te,p_a).to_csv(OUT/"calibration_additive.csv",index=False)
    calibration(te,p_i).to_csv(OUT/"calibration_interaction.csv",index=False)
    def extras(p):
        y=te.ko_win.astype(int).to_numpy(); q=np.quantile(p,.9); top=p>=q
        return dict(top_decile_precision=float(y[top].mean()),extreme_fp_ge_050=int(((p>=.5)&(y==0)).sum()),
          actual_ko_winner_mean_p=float(p[y==1].mean()),non_ko_mean_p=float(p[y==0].mean()),mean_pred=float(p.mean()),historical_rate=float(y.mean()))
    report={
      "audit":audit,"selected_decay":d,"selected_prior_strength":s,
      "selection":{"additive":sel_a.to_dict(),"interaction":sel_i.to_dict(),"delta_log_loss":float(sel_i.log_loss-sel_a.log_loss),"delta_auc":float(sel_i.auc-sel_a.auc),"delta_brier":float(sel_i.brier-sel_a.brier)},
      "confirmation":{"additive":conf_a.to_dict(),"interaction":conf_i.to_dict(),"delta_log_loss":float(conf_i.log_loss-conf_a.log_loss),"delta_auc":float(conf_i.auc-conf_a.auc),"delta_brier":float(conf_i.brier-conf_a.brier),"delta_correct_side_ko":float(conf_i.correct_side_ko-conf_a.correct_side_ko)},
      "confirmation_extras":{"additive":extras(p_a),"interaction":extras(p_i)},
      "interaction_supported":bool((conf_i.log_loss < conf_a.log_loss) and (conf_i.brier < conf_a.brier) and (conf_i.auc > conf_a.auc)),
      "uses_fsr_traits":False,"changes_mc":False,
    }
    (OUT/"report.json").write_text(json.dumps(report,indent=2,default=str)+"\n")
    print("KO V3 KD MATCHUP INTERACTION")
    print(json.dumps(report,indent=2,default=str))
    print("\nSELECTION TOP INTERACTION CONFIGS")
    print(pooled[(pooled.period=="selection")&(pooled.arm=="interaction")].sort_values("log_loss").head(8).to_string(index=False))
    print("\nCONFIRMATION SELECTED CONFIG")
    print(pooled[(pooled.period=="confirmation")&(pooled.decay==d)&(pooled.prior_strength==s)].to_string(index=False))

if __name__=="__main__": main()
