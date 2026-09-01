from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research import xgboost_method_market_offset as method
from pipeline.research import xgboost_ko_conditional_ml_stack_oof as pure

OUT = Path("data/research/prop_mispricing")
PREDICTIONS = OUT / "xgboost_ko_projected_winner_edge_oof_predictions.csv"
METRICS = OUT / "xgboost_ko_projected_winner_edge_metrics.csv"
RULES = OUT / "xgboost_ko_projected_winner_edge_rules.csv"
ROBUSTNESS = OUT / "xgboost_ko_projected_winner_edge_robustness.csv"
SUMMARY = OUT / "xgboost_ko_projected_winner_edge_summary.json"
HIER_OOF = OUT / "xgboost_method_hierarchical_v5_oof_predictions.csv"

PARAMS = {
    "max_depth": 1,
    "eta": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "min_child_weight": 10,
    "lambda": 8.0,
    "alpha": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": 42,
    "nthread": 2,
}
ROUNDS = 300
EPS = 1e-9


def q_market(frame: pd.DataFrame, side: str) -> np.ndarray:
    cols = [f"market_{side}_ko", f"market_{side}_sub", f"market_{side}_dec"]
    total = frame[cols].sum(axis=1).to_numpy(float)
    return frame[f"market_{side}_ko"].to_numpy(float) / np.clip(total, EPS, None)


def project(frame: pd.DataFrame, features: list[str]):
    red = frame["model_p_red"].to_numpy(float) >= 0.5
    sign = np.where(red, 1.0, -1.0)
    x = frame[features].replace([np.inf, -np.inf], np.nan).mul(sign, axis=0)
    ml = np.where(red, frame["model_p_red"].to_numpy(float), 1.0 - frame["model_p_red"].to_numpy(float))
    q = np.where(red, q_market(frame, "red"), q_market(frame, "blue"))
    fused = pure.clip_p(ml * q)
    fair_method = np.where(red, frame["market_red_ko"].to_numpy(float), frame["market_blue_ko"].to_numpy(float))
    target = np.where(red, frame["target"].to_numpy(int) == 0, frame["target"].to_numpy(int) == 3).astype(int)
    side = np.where(red, "red", "blue")
    return x, ml, fused, fair_method, target, side


def auc(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float)
    n1 = int(y.sum()); n0 = int(len(y)-n1)
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy(float)
    return float((ranks[y == 1].sum() - n1*(n1+1)/2.0)/(n1*n0))


def metrics(y, p):
    d = pure.binary_metrics(y, p)
    d["auc"] = auc(y, p)
    return d


def fit(train: pd.DataFrame, val: pd.DataFrame, features: list[str], include_ml: bool):
    a, ml_a, base_a, _, y, _ = project(train, features)
    b, ml_b, base_b, _, _, _ = project(val, features)
    valid = [c for c in features if a[c].notna().any()]
    med = a[valid].median(numeric_only=True)
    a = a[valid].fillna(med).fillna(0.0)
    b = b[valid].fillna(med).fillna(0.0)
    if include_ml:
        a = a.copy(); b = b.copy()
        a["v5_ml_p_projected_winner"] = ml_a
        b["v5_ml_p_projected_winner"] = ml_b
    dtr = xgb.DMatrix(a, label=y, base_margin=pure.logit(base_a), feature_names=list(a.columns))
    dva = xgb.DMatrix(b, base_margin=pure.logit(base_b), feature_names=list(b.columns))
    model = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    p = pure.clip_p(np.asarray(model.predict(dva), float))
    gain = model.get_score(importance_type="gain")
    return p, gain, len(valid) + int(include_ml)


def load_hier():
    h = pd.read_csv(HIER_OOF)
    h["fight_id"] = h["fight_id"].astype(str)
    if h["fight_id"].duplicated().any():
        raise RuntimeError("duplicate hierarchical V5 OOF fight_id")
    return h[["fight_id", "hier_red_ko", "hier_blue_ko"]]


def load_raw_projected_prices(pred: pd.DataFrame):
    raw = pure.load_raw_ko_market()
    out = pred.merge(raw, left_on=["fight_id", "projected_side"], right_on=["fight_id", "outcome_side"], how="inner", validate="one_to_one")
    out["market_raw_p"] = out["implied_probability"].astype(float)
    out["win_profit_units"] = out["profit_per_100"].astype(float)/100.0
    out["decimal_odds"] = 1.0 + out["win_profit_units"]
    out["american_odds"] = pure.american_from_profit_per_100(out["profit_per_100"])
    out["year"] = pd.to_datetime(out["date"]).dt.year.astype(int)
    out["winner_agreement"] = ((out["model_p_red"] >= 0.5) == (out["market_p_red"] >= 0.5))
    return out


def edge_frame(priced: pd.DataFrame, variant: str, pcol: str):
    x = priced.copy()
    x["variant"] = variant
    x["model_p_ko"] = x[pcol].to_numpy(float)
    x["actual_ko_win"] = x["projected_winner_ko"].astype(int)
    x["ev"] = x["model_p_ko"] * x["decimal_odds"] - 1.0
    x["prob_diff"] = x["model_p_ko"] - x["market_raw_p"]
    x["logit_residual"] = pure.logit(x["model_p_ko"]) - pure.logit(x["market_raw_p"])
    x["profit_units"] = np.where(x["actual_ko_win"].eq(1), x["win_profit_units"], -1.0)
    return x


def rule_sets(x: pd.DataFrame):
    return {
        "positive_ev": x[x["ev"] > 0].copy(),
        "positive_ev_agree": x[(x["ev"] > 0) & x["winner_agreement"]].copy(),
        "residual_020": x[x["logit_residual"] >= 0.20].copy(),
        "residual_030": x[x["logit_residual"] >= 0.30].copy(),
        "residual_030_agree": x[(x["logit_residual"] >= 0.30) & x["winner_agreement"]].copy(),
    }


def edge_stats(bets):
    return pure.bet_stats(bets)


def evaluate_edges(cand: pd.DataFrame):
    rows=[]; robust=[]
    for variant in cand["variant"].drop_duplicates():
        x=cand[cand["variant"].eq(variant)]
        for name,bets in rule_sets(x).items():
            rows.append({"variant":variant,"rule":name,"scope":"pooled",**edge_stats(bets)})
            for year in [2021,2022,2023,2024]:
                rows.append({"variant":variant,"rule":name,"scope":f"year_{year}",**edge_stats(bets[bets["year"].eq(year)])})
            for omit in [2021,2022,2023,2024]:
                robust.append({"variant":variant,"rule":name,"check":f"leave_out_{omit}",**edge_stats(bets[~bets["year"].eq(omit)])})
            if not bets.empty:
                robust.append({"variant":variant,"rule":name,"check":"remove_largest_winner",**edge_stats(bets.drop(index=bets["profit_units"].idxmax()))})
            for cap in [500,750,1000]:
                capped=bets[(bets["american_odds"]<0)|(bets["american_odds"]<=cap)]
                robust.append({"variant":variant,"rule":name,"check":f"odds_cap_plus_{cap}",**edge_stats(capped)})
    return pd.DataFrame(rows),pd.DataFrame(robust)


def run(v5_market_path: str, v5_feature_path: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df, features, excluded = method._build_rows(True, True)
    df["date"] = pd.to_datetime(df["date"])
    if (df["date"] > "2024-12-31").any():
        raise RuntimeError("2025+ entered projected-winner KO development")
    ml_stack, _, v5_ll = pure.build_honest_v5_stack(v5_market_path, v5_feature_path)
    ml_stack["fight_id"] = ml_stack["fight_id"].astype(str)
    df["fight_id"] = df["fight_id"].astype(str)
    df = df.merge(ml_stack[["fight_id", "model_p_red", "market_p_red"]], on="fight_id", how="left")
    h=load_hier()

    parts=[]; metric_rows=[]; feature_use=[]
    for fold,train_end,val_start,val_end in method.FOLDS:
        train=df[(df["date"]<=train_end)&df["model_p_red"].notna()].copy()
        val=df[(df["date"]>=val_start)&(df["date"]<=val_end)&df["model_p_red"].notna()].copy()
        val=val.merge(h,on="fight_id",how="left",validate="one_to_one")
        if train.empty or val.empty or val[["hier_red_ko","hier_blue_ko"]].isna().any(axis=None):
            raise RuntimeError(f"incomplete fold {fold}")
        _,ml,base,fair,y,side=project(val,features)
        hier=np.where(side=="red",val["hier_red_ko"].to_numpy(float),val["hier_blue_ko"].to_numpy(float))
        p,gain,fc=fit(train,val,features,False)
        pm,gainm,fcm=fit(train,val,features,True)
        for name,prob in [("method_market_fair_exact_ko",fair),("fused_base",base),("existing_hierarchical_v5",hier),("xgb_residual",p),("xgb_residual_plus_ml",pm)]:
            metric_rows.append({"fold":fold,"variant":name,**metrics(y,prob)})
        feature_use.append({"fold":fold,"v5_ml_used":bool(gainm.get("v5_ml_p_projected_winner",0)>0),"v5_ml_gain":float(gainm.get("v5_ml_p_projected_winner",0)),"feature_count":fcm})
        out=val[["fight_id","date","event_name","red_fighter","blue_fighter","target","betting_eligible","model_p_red","market_p_red","hier_red_ko","hier_blue_ko"]].copy()
        out["fold"]=fold
        out["projected_side"]=side
        out["v5_ml_p_projected_winner"]=ml
        out["projected_winner_ko"]=y
        out["market_fair_exact_ko"]=fair
        out["fused_base_p_ko"]=base
        out["hier_projected_p_ko"]=hier
        out["xgb_p_ko"]=p
        out["xgb_ml_p_ko"]=pm
        parts.append(out)

    pred=pd.concat(parts,ignore_index=True).sort_values(["date","fight_id"]).reset_index(drop=True)
    y=pred["projected_winner_ko"].to_numpy(int)
    pooled={
        "method_market_fair_exact_ko":metrics(y,pred["market_fair_exact_ko"]),
        "fused_base":metrics(y,pred["fused_base_p_ko"]),
        "existing_hierarchical_v5":metrics(y,pred["hier_projected_p_ko"]),
        "xgb_residual":metrics(y,pred["xgb_p_ko"]),
        "xgb_residual_plus_ml":metrics(y,pred["xgb_ml_p_ko"]),
    }
    for name,vals in pooled.items(): metric_rows.append({"fold":"pooled_2021_2024","variant":name,**vals})
    selected=min(["xgb_residual","xgb_residual_plus_ml"],key=lambda z:(pooled[z]["log_loss"],pooled[z]["brier"]))

    priced=load_raw_projected_prices(pred)
    cand=pd.concat([
        edge_frame(priced,"fused_base","fused_base_p_ko"),
        edge_frame(priced,"existing_hierarchical_v5","hier_projected_p_ko"),
        edge_frame(priced,"xgb_residual","xgb_p_ko"),
        edge_frame(priced,"xgb_residual_plus_ml","xgb_ml_p_ko"),
    ],ignore_index=True)
    rule_df,robust_df=evaluate_edges(cand)

    pred.to_csv(PREDICTIONS,index=False)
    pd.DataFrame(metric_rows).to_csv(METRICS,index=False)
    rule_df.to_csv(RULES,index=False)
    robust_df.to_csv(ROBUSTNESS,index=False)
    summary={
        "experiment":"projected_winner_exact_ko_market_offset_v1",
        "design":"one row per fight on the side frozen V5 projects to win; binary target is projected fighter wins by KO; base margin = V5 P(win) * sportsbook P(KO|win); XGBoost learns residual",
        "development_window":"chronological 2021-2024 OOF only",
        "reads_2025_plus":False,
        "roi_used_for_model_selection":False,
        "feature_count":len(features),
        "features":features,
        "excluded_leakage_features":excluded,
        "hyperparameters":{**PARAMS,"num_boost_round":ROUNDS},
        "v5_canonical_oof_log_loss":v5_ll,
        "selected_xgb_variant":selected,
        "selection_metric":"pooled projected-winner exact-KO binary log loss; Brier tiebreak",
        "pooled_probability_metrics":pooled,
        "delta_selected_vs_fused_base_log_loss":pooled[selected]["log_loss"]-pooled["fused_base"]["log_loss"],
        "delta_selected_vs_existing_hierarchical_v5_log_loss":pooled[selected]["log_loss"]-pooled["existing_hierarchical_v5"]["log_loss"],
        "v5_ml_split_feature_use":feature_use,
        "oof_fights":int(len(pred)),
        "edge_rules_diagnostic_only":True,
        "artifacts":[str(PREDICTIONS),str(METRICS),str(RULES),str(ROBUSTNESS)],
    }
    SUMMARY.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--v5-market",required=True)
    ap.add_argument("--v5-features",required=True)
    args=ap.parse_args()
    run(args.v5_market,args.v5_features)
