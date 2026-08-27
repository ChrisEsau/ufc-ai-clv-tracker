"""Validate shrinkage + neutral-preserving matchup blends for all-KO-per-sig.

Research only. Production unchanged. Market never used.

The previous literal union blend, 1-(1-attacker)*(1-defender), double-counts the
population prior: when attacker=defender=p0 it returns about 2*p0. This study keeps
the same chronological attacker/defender histories and shrinkage, but tests only
symmetric blends that return p0 when both inputs equal p0.

Selection: 2020-2024. Confirmation: 2025-2026.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.research import ko_v3_from_scratch_stage1 as s1

OUT=Path("data/research/ko_v3_all_ko_per_sig_validation")
SELECTION_YEARS=tuple(range(2020,2025)); CONFIRMATION_YEARS=(2025,2026)
PRIOR_STRENGTHS=(25.0,50.0,100.0,200.0,400.0)
BLENDS=("arithmetic_mean","logit_mean","logit_deviation_sum")

def sigmoid(z): return 1/(1+np.exp(-np.clip(np.asarray(z,float),-30,30)))
def logit(p):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9); return np.log(p/(1-p))

def blend(att,deff,p0,kind):
    att=np.asarray(att,float); deff=np.asarray(deff,float); p0=np.asarray(p0,float)
    if kind=="arithmetic_mean": return 0.5*(att+deff)
    if kind=="logit_mean": return sigmoid(0.5*(logit(att)+logit(deff)))
    if kind=="logit_deviation_sum": return sigmoid(logit(p0)+(logit(att)-logit(p0))+(logit(deff)-logit(p0)))
    if kind=="literal_union": return 1-(1-att)*(1-deff)
    raise ValueError(kind)

def add_strict_population_prior(frame):
    x=frame.copy(); x["event_date"]=pd.to_datetime(x.event_date).dt.normalize()
    d=x.groupby("event_date",as_index=False).agg(day_ko=("ko_win","sum"),day_sig=("sig_landed","sum")).sort_values("event_date")
    d["pk"]=d.day_ko.cumsum().shift(1,fill_value=0.0); d["pn"]=d.day_sig.cumsum().shift(1,fill_value=0.0)
    d["population_ko_per_sig"]=np.divide(d.pk,d.pn,out=np.full(len(d),np.nan),where=d.pn>0)
    return x.merge(d[["event_date","population_ko_per_sig","pk","pn"]],on="event_date",how="left",validate="many_to_one")

def shrunk(k,n,p0,s):
    return (np.asarray(k,float)+s*np.asarray(p0,float))/(np.asarray(n,float)+s)

def metrics(g,p):
    y=g.ko_win.astype(int).to_numpy(); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    return {"rows":int(len(g)),"ko_wins":int(y.sum()),"actual_ko_win_rate":float(y.mean()),"mean_predicted_ko_probability":float(p.mean()),"calibration_bias":float(p.mean()-y.mean()),"auc":float(roc_auc_score(y,p)),"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),"extreme_false_positives_ge_50":int(((p>=.5)&(y==0)).sum()),"top_decile_precision":float(y[p>=np.quantile(p,.9)].mean()),"mean_p_actual_ko_winners":float(p[y==1].mean()),"mean_p_non_ko":float(p[y==0].mean())}

def candidate(frame,s,kind):
    x=frame.copy(); p0=x.population_ko_per_sig.to_numpy(float)
    a=shrunk(x.prior_ko_wins,x.prior_sig_landed,p0,s); d=shrunk(x.opp_prior_ko_losses,x.opp_prior_sig_absorbed,p0,s)
    valid=np.isfinite(a)&np.isfinite(d)&np.isfinite(p0)&x.sig_landed.gt(0).to_numpy(); x=x.loc[valid].copy(); a=a[valid]; d=d[valid]; p0=p0[valid]
    ps=np.clip(blend(a,d,p0,kind),0,1); pf=1-np.power(1-ps,x.sig_landed.to_numpy(float))
    x["att_ko_per_sig"]=a; x["def_ko_loss_per_sig"]=d; x["combined_per_sig"]=ps; x["p_fight"]=pf; x["prior_strength"]=s; x["blend"]=kind
    return x

def score(x,years):
    g=x[x.test_year.isin(years)]; return metrics(g,g.p_fight)
def correct_side(x,years):
    g=x[x.test_year.isin(years)]; vals=[]
    for _,b in g.groupby("fight_id"):
        if len(b)==2 and bool(b.ko_win.any()):
            w=b[b.ko_win].iloc[0]; l=b[~b.ko_win].iloc[0]; vals.append(float(w.p_fight)>float(l.p_fight))
    return {"ko_fights":len(vals),"correct_side_rate":float(np.mean(vals))}
def zeros(x,years):
    g=x[x.test_year.isin(years)]; return {"rows":int(len(g)),"zero_per_sig_hazards":int((g.combined_per_sig<=0).sum()),"zero_fight_probabilities":int((g.p_fight<=0).sum())}
def population(frame,years):
    g=frame[frame.test_year.isin(years)&frame.sig_landed.gt(0)&frame.population_ko_per_sig.notna()]; p=1-np.power(1-g.population_ko_per_sig.to_numpy(float),g.sig_landed.to_numpy(float)); return metrics(g,p)
def literal_raw(frame,years):
    x=frame.copy(); p0=x.population_ko_per_sig.to_numpy(float)
    a=np.divide(x.prior_ko_wins,x.prior_sig_landed,out=p0.copy(),where=x.prior_sig_landed.to_numpy(float)>0)
    d=np.divide(x.opp_prior_ko_losses,x.opp_prior_sig_absorbed,out=p0.copy(),where=x.opp_prior_sig_absorbed.to_numpy(float)>0)
    valid=np.isfinite(a)&np.isfinite(d)&x.sig_landed.gt(0).to_numpy(); x=x.loc[valid].copy(); ps=blend(a[valid],d[valid],p0[valid],"literal_union"); x["combined_per_sig"]=ps; x["p_fight"]=1-np.power(1-ps,x.sig_landed.to_numpy(float)); return {"metrics":score(x,years),"correct_side":correct_side(x,years),"zero_audit":zeros(x,years)}

def main():
    OUT.mkdir(parents=True,exist_ok=True); ff,audit=s1.load_raw_fighter_fights(ROUND_STATS_PATH,MASTER_PATH); frame=add_strict_population_prior(s1.build_matchup_frame(s1.build_prefight_states(ff)))
    variants={}; preds=[]
    for kind in BLENDS:
        for s in PRIOR_STRENGTHS:
            x=candidate(frame,s,kind); key=f"{kind}_s{int(s)}"; variants[key]={"blend":kind,"prior_strength_sig_strikes":s,"selection":score(x,SELECTION_YEARS),"confirmation":score(x,CONFIRMATION_YEARS),"selection_correct_side":correct_side(x,SELECTION_YEARS),"confirmation_correct_side":correct_side(x,CONFIRMATION_YEARS),"selection_zero_hazard_audit":zeros(x,SELECTION_YEARS),"confirmation_zero_hazard_audit":zeros(x,CONFIRMATION_YEARS)}; preds.append(x[["event_date","fight_id","fighter_id","fighter_name","opponent_id","ko_win","sig_landed","population_ko_per_sig","prior_strength","blend","att_ko_per_sig","def_ko_loss_per_sig","combined_per_sig","p_fight"]])
    selkey=min(variants,key=lambda k:variants[k]["selection"]["log_loss"]); sel=variants[selkey]
    report={"study":"neutral-preserving shrinkage/blend for all KO/TKO per sig","market_used":False,"changes_mc":False,"same_date_delayed_fighter_histories":True,"population_prior_strictly_before_event_date":True,"selection_years":list(SELECTION_YEARS),"confirmation_years":list(CONFIRMATION_YEARS),"prior_strength_grid_sig_strikes":list(PRIOR_STRENGTHS),"blend_grid":list(BLENDS),"selection_rule":"minimum 2020-2024 log_loss","selected":{"key":selkey,**sel},"literal_union_raw":{"selection":literal_raw(frame,SELECTION_YEARS),"confirmation":literal_raw(frame,CONFIRMATION_YEARS)},"population":{"selection":population(frame,SELECTION_YEARS),"confirmation":population(frame,CONFIRMATION_YEARS)},"variants":variants,"raw_audit":audit}
    (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); pd.concat(preds,ignore_index=True).to_csv(OUT/"shrinkage_predictions.csv",index=False); print("KO V3 NEUTRAL-PRESERVING SHRINKAGE VALIDATION"); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
