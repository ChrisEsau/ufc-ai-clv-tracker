from __future__ import annotations

import json
import numpy as np
import pandas as pd

from pipeline.research.xgboost_method_market_offset import ROOT
from pipeline.research.xgboost_method_bet_gate_oof_diagnostic import _long_ledger

OUT_SIGNAL = ROOT / "xgboost_method_market_offset__2021_2024_oof_method_family_residual_signal.csv"
OUT_CROWD = ROOT / "xgboost_method_market_offset__2021_2024_oof_method_family_crowding.csv"
OUT_SUMMARY = ROOT / "xgboost_method_market_offset__2021_2024_oof_method_family_residual_signal_summary.json"

THRESHOLDS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
EPS = 1e-12


def family(class_name: str) -> str:
    if class_name.endswith("DEC"):
        return "DEC"
    if "KO_TKO" in class_name:
        return "KO_TKO"
    return "SUB"


def binary_log_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(np.mean(-(y * np.log(p) + (1-y) * np.log(1-p))))


def brier(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((p-y)**2))


def metrics(g: pd.DataFrame) -> dict:
    n = len(g)
    if not n:
        return {"rows":0,"unique_fights":0,"hit_rate":None,"avg_model_prob":None,"model_calibration_gap":None,
                "avg_fair_market_prob":None,"market_calibration_gap":None,"avg_raw_implied_prob":None,
                "avg_signed_logit_residual":None,"avg_price_edge":None,"model_binary_log_loss":None,
                "market_binary_log_loss":None,"log_loss_delta_model_minus_market":None,"model_brier":None,
                "market_brier":None,"brier_delta_model_minus_market":None}
    y=g.won.to_numpy(float); mp=g.model_prob.to_numpy(float); fp=g.fair_market_prob.to_numpy(float)
    hit=float(y.mean()); am=float(mp.mean()); af=float(fp.mean())
    mll=binary_log_loss(y,mp); fll=binary_log_loss(y,fp); mb=brier(y,mp); fb=brier(y,fp)
    return {"rows":n,"unique_fights":int(g.fight_id.nunique()),"hit_rate":hit,"avg_model_prob":am,
            "model_calibration_gap":hit-am,"avg_fair_market_prob":af,"market_calibration_gap":hit-af,
            "avg_raw_implied_prob":float(g.raw_implied_prob.mean()),"avg_signed_logit_residual":float(g.signed_logit_residual.mean()),
            "avg_price_edge":float(g.price_edge.mean()),"model_binary_log_loss":mll,"market_binary_log_loss":fll,
            "log_loss_delta_model_minus_market":mll-fll,"model_brier":mb,"market_brier":fb,
            "brier_delta_model_minus_market":mb-fb}


def main():
    df=_long_ledger().copy()
    if df.fight_id.nunique()!=1604:
        raise RuntimeError("authoritative OOF fight count changed")
    if df.date.max() > pd.Timestamp("2024-12-31"):
        raise RuntimeError("2025+ data present; abort")
    df["method_family"] = df.class_name.map(family)

    rows=[]
    for view in ["residual_only","raw_break_even_eligible"]:
        for t in THRESHOLDS:
            g0=df[df.signed_logit_residual >= t]
            if view=="raw_break_even_eligible":
                g0=g0[g0.price_edge > 0]
            for fam in ["DEC","KO_TKO","SUB"]:
                g=g0[g0.method_family==fam]
                rows.append({"view":view,"logit_threshold":t,"method_family":fam,**metrics(g)})
    signal=pd.DataFrame(rows)
    signal.to_csv(OUT_SIGNAL,index=False)

    # Crowding at frozen 0.30 + raw-break-even eligibility, before one-bet-per-fight pruning.
    q=df[(df.signed_logit_residual>=0.30)&(df.price_edge>0)].copy()
    top=(q.sort_values(["fight_id","signed_logit_residual","price_edge","class_slug"],ascending=[True,False,False,True])
           .drop_duplicates("fight_id",keep="first")[["fight_id","method_family","class_name","signed_logit_residual"]]
           .rename(columns={"method_family":"winner_family","class_name":"winner_class","signed_logit_residual":"winner_logit"}))
    crowd=[]
    for fid,g in q.groupby("fight_id"):
        rec={"fight_id":fid,
             "qualifying_total":len(g),
             "qualifying_dec":int((g.method_family=="DEC").sum()),
             "qualifying_ko":int((g.method_family=="KO_TKO").sum()),
             "qualifying_sub":int((g.method_family=="SUB").sum()),
             "has_dec":bool((g.method_family=="DEC").any()),
             "has_ko":bool((g.method_family=="KO_TKO").any()),
             "has_sub":bool((g.method_family=="SUB").any())}
        crowd.append(rec)
    crowd=pd.DataFrame(crowd).merge(top,on="fight_id",how="left")
    crowd.to_csv(OUT_CROWD,index=False)

    summary={
        "experiment":"six_way_method_oof_family_residual_signal_v1",
        "period":"chronological 2021-2024 OOF only",
        "source":"authoritative persisted frozen OOF ledger via _long_ledger()",
        "oof_fights":1604,
        "class_rows":int(len(df)),
        "no_refit":True,"no_roi_or_profit":True,"no_one_bet_pruning_in_signal_tables":True,"reads_2025_plus":False,
        "crowding_gate":{"signed_logit_residual":0.30,"price_edge":"model_prob > raw_implied_prob"},
        "crowding":{"qualifying_fights":int(len(crowd)),"qualifying_class_rows":int(len(q)),
                    "fights_with_dec":int(crowd.has_dec.sum()),"fights_with_ko":int(crowd.has_ko.sum()),"fights_with_sub":int(crowd.has_sub.sum()),
                    "top_family_counts":{k:int(v) for k,v in crowd.winner_family.value_counts().to_dict().items()},
                    "ko_present_but_dec_wins":int(((crowd.has_ko)&(crowd.winner_family=="DEC")).sum()),
                    "sub_present_but_dec_wins":int(((crowd.has_sub)&(crowd.winner_family=="DEC")).sum())},
        "threshold_030": signal[signal.logit_threshold.eq(0.30)].to_dict(orient="records")
    }
    OUT_SUMMARY.write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
    print("\nSIGNAL\n",signal.to_string(index=False))

if __name__=="__main__":
    main()
