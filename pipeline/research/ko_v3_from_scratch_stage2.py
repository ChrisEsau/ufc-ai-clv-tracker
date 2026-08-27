"""KO V3 from scratch — Stage 2 hazard construction and confirmation.

Builds MC-compatible hazards from only raw Stage-1 evidence:
  * KD hazard per landed significant strike;
  * direct KO/TKO proxy hazard per landed significant strike;
  * post-KD finish-sequence hazard per recorded knockdown.

No FSR traits are read. No MC mechanics are changed. Hyperparameters are selected
using expanding-year OOS predictions from 2020-2024 and confirmed on 2025-2026.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.research import ko_v3_from_scratch_stage1 as s1

DEFAULT_OUT = Path("data/research/ko_v3_from_scratch_stage2")
DECAYS = s1.EWM_DECAYS
PRIOR_STRENGTHS = (25.0, 50.0, 100.0, 200.0, 400.0)
SELECTION_YEARS = tuple(range(2020, 2025))
CONFIRMATION_YEARS = (2025, 2026)


def weighted_rows(frame: pd.DataFrame, kcol: str, ncol: str):
    ids=[]; ys=[]; ws=[]
    for idx,row in frame.iterrows():
        k=float(row[kcol]); n=float(row[ncol])
        if not (n>0 and 0<=k<=n): continue
        if k>0: ids.append(idx); ys.append(1); ws.append(k)
        if n-k>0: ids.append(idx); ys.append(0); ws.append(n-k)
    return frame.loc[ids].reset_index(drop=True),np.asarray(ys,int),np.asarray(ws,float)


def fit_logit(train: pd.DataFrame, test: pd.DataFrame, cols, cats, y, weights):
    arm=s1.Arm("candidate",tuple(cols),tuple(cats))
    enc=s1.NumericCategoricalEncoder(arm.numeric,arm.categorical).fit(train)
    model=LogisticRegression(C=1.0,max_iter=5000,solver="lbfgs")
    model.fit(enc.transform(train),y,sample_weight=weights)
    return model.predict_proba(enc.transform(test))[:,1],enc,model


def hazard_metrics(g: pd.DataFrame, p: np.ndarray, kcol: str, ncol: str) -> dict:
    k=g[kcol].to_numpy(float); n=g[ncol].to_numpy(float); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    total=float(n.sum()); any_y=(k>0).astype(int); any_p=1-np.power(1-p,n)
    return {"n_rows":int(len(g)),"opportunities":total,"events":float(k.sum()),"actual_hazard":float(k.sum()/max(total,1.0)),"predicted_hazard":float(np.sum(n*p)/max(total,1.0)),"event_log_loss":float(-np.sum(k*np.log(p)+(n-k)*np.log1p(-p))/max(total,1.0)),"event_brier":float(np.sum(k*(1-p)**2+(n-k)*p**2)/max(total,1.0)),"any_event_auc":float(roc_auc_score(any_y,any_p)) if np.unique(any_y).size==2 else np.nan,"any_event_brier":float(brier_score_loss(any_y,any_p))}


def add_shrunken(frame: pd.DataFrame, *, kind: str, decay: float, strength: float, p0: float) -> pd.DataFrame:
    x=frame.copy(); tag=s1._tag(decay)
    if kind=="kd":
        ak,an=f"{tag}_kd_scored",f"{tag}_sig_landed"; dk,dn=f"opp_{tag}_kd_absorbed",f"opp_{tag}_sig_absorbed"
    elif kind=="direct":
        ak,an=f"{tag}_direct_ko_wins",f"{tag}_sig_landed"; dk,dn=f"opp_{tag}_direct_ko_losses",f"opp_{tag}_sig_absorbed"
    else: raise ValueError(kind)
    x["shr_att"]=(x[ak].astype(float)+p0*strength)/(x[an].astype(float)+strength)
    x["shr_def"]=(x[dk].astype(float)+p0*strength)/(x[dn].astype(float)+strength)
    x["shr_att_log_exp"]=np.log1p(x[an].astype(float)); x["shr_def_log_exp"]=np.log1p(x[dn].astype(float))
    return x


def evaluate_hazard_grid(frame: pd.DataFrame, *, kind: str, kcol: str, ncol: str, first_year: int=2020):
    usable=frame[(frame[ncol]>0)&(frame[kcol]>=0)&(frame[kcol]<=frame[ncol])].copy(); years=sorted(y for y in usable.test_year.unique() if y>=first_year)
    candidates=[("age_division",None,None,False)]
    for decay in DECAYS:
        for strength in PRIOR_STRENGTHS:
            candidates.append((f"shr_{s1._tag(decay)}_s{int(strength)}",decay,strength,False)); candidates.append((f"shr_{s1._tag(decay)}_s{int(strength)}_exp",decay,strength,True))
    preds=[]; by_year=[]; coef=[]
    for year in years:
        train=usable[usable.event_date<pd.Timestamp(f"{year}-01-01")].copy(); test=usable[usable.test_year.eq(year)].copy()
        if len(train)<500 or len(test)<20: continue
        p0=float(train[kcol].sum()/train[ncol].sum()); p=np.full(len(test),np.clip(p0,1e-9,1-1e-9)); m=hazard_metrics(test,p,kcol,ncol); m.update(test_year=year,arm="population"); by_year.append(m)
        d=test[["event_date","fight_id","fighter_id","fighter_name","opponent_id","division",kcol,ncol]].copy(); d["test_year"],d["arm"],d["p_hazard"]=year,"population",p; preds.append(d)
        for name,decay,strength,with_exp in candidates:
            tr=train; te=test
            if decay is not None:
                tr=add_shrunken(train,kind=kind,decay=decay,strength=strength,p0=p0); te=add_shrunken(test,kind=kind,decay=decay,strength=strength,p0=p0)
                cols=["shr_att","shr_def","attacker_age","defender_age"] + (["shr_att_log_exp","shr_def_log_exp"] if with_exp else [])
            else: cols=["attacker_age","defender_age"]
            trw,y,w=weighted_rows(tr,kcol,ncol)
            if np.unique(y).size<2: continue
            try: p,enc,model=fit_logit(trw,te,cols,["division_cat"],y,w)
            except Exception as exc: print(f"{kind} SKIP year={year} arm={name}: {exc}"); continue
            m=hazard_metrics(te,p,kcol,ncol); m.update(test_year=year,arm=name); by_year.append(m)
            for fn,v in [("intercept",model.intercept_[0]),*zip(enc.feature_names,model.coef_[0])]: coef.append({"test_year":year,"arm":name,"feature":fn,"coefficient":float(v)})
            d=te[["event_date","fight_id","fighter_id","fighter_name","opponent_id","division",kcol,ncol]].copy(); d["test_year"],d["arm"],d["p_hazard"]=year,name,p; preds.append(d)
    return pd.concat(preds,ignore_index=True),pd.DataFrame(by_year),pd.DataFrame(coef)


def pooled_hazard(pred: pd.DataFrame, *, kcol: str, ncol: str, years) -> pd.DataFrame:
    rows=[]
    for arm,g in pred[pred.test_year.isin(years)].groupby("arm",sort=False):
        m=hazard_metrics(g,g.p_hazard.to_numpy(float),kcol,ncol); m["arm"]=arm; rows.append(m)
    return pd.DataFrame(rows).sort_values(["event_log_loss","event_brier"]).reset_index(drop=True)


@dataclass
class SequenceModel:
    encoder: s1.NumericCategoricalEncoder
    beta: np.ndarray
    def hazard(self, frame: pd.DataFrame) -> np.ndarray:
        X=self.encoder.transform(frame); eta=self.beta[0]+X@self.beta[1:]; return 1/(1+np.exp(-np.clip(eta,-30,30)))


def fit_sequence_model(train: pd.DataFrame) -> SequenceModel:
    arm=s1.Arm("age_division",("attacker_age","defender_age"),("division_cat",)); enc=s1.NumericCategoricalEncoder(arm.numeric,arm.categorical).fit(train)
    X=enc.transform(train); y=train["post_kd_finish"].to_numpy(float); k=train["kd_scored"].to_numpy(float)
    def objective(beta):
        eta=beta[0]+X@beta[1:]; h=1/(1+np.exp(-np.clip(eta,-30,30))); p=np.clip(1-np.power(1-h,k),1e-9,1-1e-9)
        return float(-np.sum(y*np.log(p)+(1-y)*np.log1p(-p))+0.5*0.01*np.sum(beta[1:]**2))
    init=np.zeros(X.shape[1]+1); init[0]=-1.0; res=minimize(objective,init,method="L-BFGS-B",options={"maxiter":1000})
    if not res.success: raise RuntimeError(f"sequence fit failed: {res.message}")
    return SequenceModel(enc,np.asarray(res.x,float))


def sequence_metrics(g: pd.DataFrame, h: np.ndarray) -> dict:
    y=g.post_kd_finish.to_numpy(int); k=g.kd_scored.to_numpy(float); h=np.clip(h,1e-9,1-1e-9); p=1-np.power(1-h,k)
    return {"n":int(len(g)),"recorded_kds":float(k.sum()),"actual_finish_rate":float(y.mean()),"predicted_finish_rate":float(p.mean()),"mean_per_kd_sequence_hazard":float(h.mean()),"auc":float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan,"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,np.clip(p,1e-9,1-1e-9),labels=[0,1]))}


def evaluate_sequence(frame: pd.DataFrame, first_year=2020):
    usable=frame[frame.post_kd_opportunity.gt(0)&frame.kd_scored.gt(0)].copy(); preds=[]; rows=[]; coefs=[]
    for year in sorted(y for y in usable.test_year.unique() if y>=first_year):
        tr=usable[usable.event_date<pd.Timestamp(f"{year}-01-01")].copy(); te=usable[usable.test_year.eq(year)].copy()
        if len(tr)<200 or len(te)<10: continue
        model=fit_sequence_model(tr); h=model.hazard(te); m=sequence_metrics(te,h); m["test_year"]=year; rows.append(m)
        d=te[["event_date","fight_id","fighter_id","fighter_name","opponent_id","division","kd_scored","post_kd_finish"]].copy(); d["test_year"]=year; d["p_per_kd_sequence"]=h; d["p_finish_given_recorded_kds"]=1-np.power(1-h,te.kd_scored.to_numpy(float)); preds.append(d)
        for n,v in zip(["intercept"]+model.encoder.feature_names,model.beta,strict=True): coefs.append({"test_year":year,"feature":n,"coefficient":float(v)})
    return pd.concat(preds,ignore_index=True),pd.DataFrame(rows),pd.DataFrame(coefs)


def pooled_sequence(pred: pd.DataFrame,years) -> dict:
    g=pred[pred.test_year.isin(years)].copy(); return sequence_metrics(g,g.p_per_kd_sequence.to_numpy(float)) if len(g) else {}


def confirmation_compare(summary: pd.DataFrame, selected: str) -> dict:
    rows={r.arm:r for r in summary.itertuples(index=False)}; s=rows[selected]; b=rows["age_division"]
    return {"selected":selected,"selected_log_loss":float(s.event_log_loss),"age_division_log_loss":float(b.event_log_loss),"delta_log_loss":float(s.event_log_loss-b.event_log_loss),"selected_auc":float(s.any_event_auc),"age_division_auc":float(b.any_event_auc)}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--round-path",type=Path,default=ROUND_STATS_PATH); p.add_argument("--master-path",type=Path,default=MASTER_PATH); p.add_argument("--out-dir",type=Path,default=DEFAULT_OUT); return p.parse_args()


def main():
    args=parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    ff,audit=s1.load_raw_fighter_fights(args.round_path,args.master_path); frame=s1.build_matchup_frame(s1.build_prefight_states(ff))
    kd_pred,kd_year,kd_coef=evaluate_hazard_grid(frame,kind="kd",kcol="kd_scored",ncol="sig_landed"); kd_sel=pooled_hazard(kd_pred,kcol="kd_scored",ncol="sig_landed",years=SELECTION_YEARS); kd_confirm=pooled_hazard(kd_pred,kcol="kd_scored",ncol="sig_landed",years=CONFIRMATION_YEARS); kd_choice=str(kd_sel.iloc[0].arm)
    direct_frame=frame[(frame.sig_landed>0)&(frame.direct_ko_win<=frame.sig_landed)].copy(); direct_pred,direct_year,direct_coef=evaluate_hazard_grid(direct_frame,kind="direct",kcol="direct_ko_win",ncol="sig_landed"); direct_sel=pooled_hazard(direct_pred,kcol="direct_ko_win",ncol="sig_landed",years=SELECTION_YEARS); direct_confirm=pooled_hazard(direct_pred,kcol="direct_ko_win",ncol="sig_landed",years=CONFIRMATION_YEARS); direct_choice=str(direct_sel.iloc[0].arm)
    seq_pred,seq_year,seq_coef=evaluate_sequence(frame); seq_select=pooled_sequence(seq_pred,SELECTION_YEARS); seq_confirm=pooled_sequence(seq_pred,CONFIRMATION_YEARS)
    for name,df in {"kd_hazard_predictions.csv":kd_pred,"kd_hazard_by_year.csv":kd_year,"kd_hazard_coefficients.csv":kd_coef,"kd_selection_summary.csv":kd_sel,"kd_confirmation_summary.csv":kd_confirm,"direct_hazard_predictions.csv":direct_pred,"direct_hazard_by_year.csv":direct_year,"direct_hazard_coefficients.csv":direct_coef,"direct_selection_summary.csv":direct_sel,"direct_confirmation_summary.csv":direct_confirm,"post_kd_sequence_predictions.csv":seq_pred,"post_kd_sequence_by_year.csv":seq_year,"post_kd_sequence_coefficients.csv":seq_coef}.items(): df.to_csv(args.out_dir/name,index=False)
    report={"stage":"KO V3 from scratch — Stage 2 MC-compatible hazards","uses_fsr_traits":False,"changes_mc_mechanics":False,"same_date_delayed":True,"selection_years":list(SELECTION_YEARS),"confirmation_years":list(CONFIRMATION_YEARS),"raw_data_audit":audit,"kd_selected_on_2020_2024":kd_choice,"kd_confirmation":confirmation_compare(kd_confirm,kd_choice),"direct_selected_on_2020_2024":direct_choice,"direct_confirmation":confirmation_compare(direct_confirm,direct_choice),"post_kd_sequence_selection":seq_select,"post_kd_sequence_confirmation":seq_confirm,"promotion_rules":{"kd":"promote only if selected raw shrunken history beats age+division on untouched 2025-2026 confirmation","direct":"same confirmation rule; target is zero-recorded-KD KO/TKO proxy per landed sig strike","post_kd":"age+division only; fighter conversion/recovery history failed Stage-1b ablation","hurt_decay":"not estimated here because aggregate round stats do not contain KD timestamps"}}
    (args.out_dir/"stage2_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KO V3 FROM SCRATCH — STAGE 2"); print(json.dumps(report,indent=2,sort_keys=True)); print("\nKD SELECTION TOP 10"); print(kd_sel.head(10).to_string(index=False,float_format=lambda x:f"{x:.6f}")); print("\nKD CONFIRMATION"); print(kd_confirm[kd_confirm.arm.isin([kd_choice,"age_division","population"])].to_string(index=False,float_format=lambda x:f"{x:.6f}")); print("\nDIRECT SELECTION TOP 10"); print(direct_sel.head(10).to_string(index=False,float_format=lambda x:f"{x:.6f}")); print("\nDIRECT CONFIRMATION"); print(direct_confirm[direct_confirm.arm.isin([direct_choice,"age_division","population"])].to_string(index=False,float_format=lambda x:f"{x:.6f}"))


if __name__=="__main__": main()
