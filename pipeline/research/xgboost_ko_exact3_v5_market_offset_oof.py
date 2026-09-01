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
PREDICTIONS = OUT / "xgboost_ko_exact3_v5_market_offset_oof_predictions.csv"
CANDIDATES = OUT / "xgboost_ko_exact3_v5_market_offset_edge_candidates.csv"
METRICS = OUT / "xgboost_ko_exact3_v5_market_offset_metrics.csv"
RULES = OUT / "xgboost_ko_exact3_v5_market_offset_edge_rules.csv"
ROBUSTNESS = OUT / "xgboost_ko_exact3_v5_market_offset_edge_robustness.csv"
SUMMARY = OUT / "xgboost_ko_exact3_v5_market_offset_summary.json"
HIER_OOF = OUT / "xgboost_method_hierarchical_v5_oof_predictions.csv"

PARAMS = {
    "max_depth": 1,
    "eta": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "min_child_weight": 10,
    "lambda": 8.0,
    "alpha": 1.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "seed": 42,
    "nthread": 2,
}
ROUNDS = 300
EPS = 1e-9


def normalize3(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def side_market_q(frame: pd.DataFrame, side: str) -> np.ndarray:
    cols = [f"market_{side}_ko", f"market_{side}_sub", f"market_{side}_dec"]
    total = frame[cols].sum(axis=1).to_numpy(float)
    return frame[f"market_{side}_ko"].to_numpy(float) / np.clip(total, EPS, None)


def fused_base3(frame: pd.DataFrame) -> np.ndarray:
    p_red = frame["model_p_red"].to_numpy(float)
    q_red = side_market_q(frame, "red")
    q_blue = side_market_q(frame, "blue")
    p = np.column_stack([
        p_red * q_red,
        (1.0 - p_red) * q_blue,
        1.0 - p_red * q_red - (1.0 - p_red) * q_blue,
    ])
    return normalize3(p)


def method_market3(frame: pd.DataFrame) -> np.ndarray:
    p = np.column_stack([
        frame["market_red_ko"].to_numpy(float),
        frame["market_blue_ko"].to_numpy(float),
        1.0 - frame["market_red_ko"].to_numpy(float) - frame["market_blue_ko"].to_numpy(float),
    ])
    return normalize3(p)


def target3(frame: pd.DataFrame) -> np.ndarray:
    t = frame["target"].to_numpy(int)
    return np.where(t == 0, 0, np.where(t == 3, 1, 2))


def prep(train: pd.DataFrame, score: pd.DataFrame, features: list[str], include_ml: bool):
    a = train[features].replace([np.inf, -np.inf], np.nan)
    b = score[features].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in features if a[c].notna().any()]
    med = a[valid].median(numeric_only=True)
    a = a[valid].fillna(med).fillna(0.0)
    b = b[valid].fillna(med).fillna(0.0)
    if include_ml:
        a = a.copy(); b = b.copy()
        a["v5_model_p_red"] = train["model_p_red"].to_numpy(float)
        b["v5_model_p_red"] = score["model_p_red"].to_numpy(float)
    return a, b, valid


def fit_predict(train: pd.DataFrame, score: pd.DataFrame, features: list[str], include_ml: bool):
    a, b, valid = prep(train, score, features, include_ml)
    y = target3(train)
    tr_base = fused_base3(train)
    sc_base = fused_base3(score)
    dtr = xgb.DMatrix(a, label=y, feature_names=list(a.columns))
    dsc = xgb.DMatrix(b, feature_names=list(b.columns))
    dtr.set_base_margin(np.log(tr_base).reshape(-1))
    dsc.set_base_margin(np.log(sc_base).reshape(-1))
    model = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    pred = normalize3(np.asarray(model.predict(dsc), float))
    gain = model.get_score(importance_type="gain")
    return pred, gain, len(valid) + int(include_ml)


def load_hier() -> pd.DataFrame:
    h = pd.read_csv(HIER_OOF)
    h["fight_id"] = h["fight_id"].astype(str)
    if h["fight_id"].duplicated().any():
        raise RuntimeError("duplicate hierarchical V5 OOF fight_id")
    return h[["fight_id", "hier_red_ko", "hier_blue_ko"]]


def hier3(frame: pd.DataFrame) -> np.ndarray:
    return normalize3(np.column_stack([
        frame["hier_red_ko"].to_numpy(float),
        frame["hier_blue_ko"].to_numpy(float),
        1.0 - frame["hier_red_ko"].to_numpy(float) - frame["hier_blue_ko"].to_numpy(float),
    ]))


def build_candidates(pred: pd.DataFrame) -> pd.DataFrame:
    raw = pure.load_raw_ko_market()
    rows = []
    variants = {
        "fused_base": ("fused_red_ko", "fused_blue_ko"),
        "xgb_residual": ("xgb_red_ko", "xgb_blue_ko"),
        "xgb_residual_plus_ml": ("xgb_ml_red_ko", "xgb_ml_blue_ko"),
        "existing_hierarchical_v5": ("hier_red_ko", "hier_blue_ko"),
    }
    for variant, (rcol, bcol) in variants.items():
        for side, pcol in [("red", rcol), ("blue", bcol)]:
            part = pred[["fight_id", "date", "fold", "event_name", "red_fighter", "blue_fighter", "target", "betting_eligible", "model_p_red", "market_p_red", pcol]].copy()
            part["variant"] = variant
            part["side"] = side
            part["fighter"] = np.where(side == "red", part["red_fighter"], part["blue_fighter"])
            part["model_p_ko"] = part[pcol].to_numpy(float)
            part["actual_ko_win"] = np.where(side == "red", part["target"].to_numpy(int) == 0, part["target"].to_numpy(int) == 3).astype(int)
            part["v5_ml_p_side"] = np.where(side == "red", part["model_p_red"], 1.0 - part["model_p_red"])
            part["market_ml_p_side"] = np.where(side == "red", part["market_p_red"], 1.0 - part["market_p_red"])
            part["v5_projected_winner"] = part["v5_ml_p_side"] >= 0.5
            part["winner_agreement"] = ((part["model_p_red"] >= 0.5) == (part["market_p_red"] >= 0.5))
            rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(raw, left_on=["fight_id", "side"], right_on=["fight_id", "outcome_side"], how="inner")
    out["market_raw_p"] = out["implied_probability"].astype(float)
    out["win_profit_units"] = out["profit_per_100"].astype(float) / 100.0
    out["decimal_odds"] = 1.0 + out["win_profit_units"]
    out["american_odds"] = pure.american_from_profit_per_100(out["profit_per_100"])
    out["ev"] = out["model_p_ko"] * out["decimal_odds"] - 1.0
    out["prob_diff"] = out["model_p_ko"] - out["market_raw_p"]
    out["logit_residual"] = pure.logit(out["model_p_ko"]) - pure.logit(out["market_raw_p"])
    out["profit_units"] = np.where(out["actual_ko_win"].eq(1), out["win_profit_units"], -1.0)
    out["year"] = pd.to_datetime(out["date"]).dt.year.astype(int)
    return out.sort_values(["date", "fight_id", "variant", "side"]).reset_index(drop=True)


def choose_one(frame: pd.DataFrame, rank: str) -> pd.DataFrame:
    return frame.sort_values(["date", "fight_id", rank, "model_p_ko"], ascending=[True, True, False, False]).drop_duplicates("fight_id", keep="first")


def rule_sets(cand: pd.DataFrame, variant: str) -> dict[str, pd.DataFrame]:
    x = cand[cand["variant"].eq(variant)].copy()
    return {
        "positive_ev_one_best_ev": choose_one(x[x["ev"] > 0], "ev"),
        "residual_030_one": choose_one(x[x["logit_residual"] >= 0.30], "logit_residual"),
        "v5_winner_positive_ev_one": choose_one(x[(x["v5_projected_winner"]) & (x["ev"] > 0)], "ev"),
        "v5_winner_agree_positive_ev_one": choose_one(x[(x["v5_projected_winner"]) & (x["winner_agreement"]) & (x["ev"] > 0)], "ev"),
        "v5_winner_residual030_one": choose_one(x[(x["v5_projected_winner"]) & (x["logit_residual"] >= 0.30)], "logit_residual"),
        "v5_winner_agree_residual030_one": choose_one(x[(x["v5_projected_winner"]) & (x["winner_agreement"]) & (x["logit_residual"] >= 0.30)], "logit_residual"),
    }


def evaluate_edges(cand: pd.DataFrame):
    rows, robust = [], []
    for variant in cand["variant"].drop_duplicates():
        for name, bets in rule_sets(cand, variant).items():
            rows.append({"variant": variant, "rule": name, "scope": "pooled", **pure.bet_stats(bets)})
            for year in [2021, 2022, 2023, 2024]:
                rows.append({"variant": variant, "rule": name, "scope": f"year_{year}", **pure.bet_stats(bets[bets["year"].eq(year)])})
            for omit in [2021, 2022, 2023, 2024]:
                robust.append({"variant": variant, "rule": name, "check": f"leave_out_{omit}", **pure.bet_stats(bets[~bets["year"].eq(omit)])})
            if not bets.empty:
                robust.append({"variant": variant, "rule": name, "check": "remove_largest_winner", **pure.bet_stats(bets.drop(index=bets["profit_units"].idxmax()))})
            for cap in [500, 750, 1000]:
                capped = bets[(bets["american_odds"] < 0) | (bets["american_odds"] <= cap)]
                robust.append({"variant": variant, "rule": name, "check": f"odds_cap_plus_{cap}", **pure.bet_stats(capped)})
    return pd.DataFrame(rows), pd.DataFrame(robust)


def run(v5_market_path: str, v5_feature_path: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df, features, excluded = method._build_rows(True, True)
    df["date"] = pd.to_datetime(df["date"])
    if (df["date"] > "2024-12-31").any():
        raise RuntimeError("2025+ entered exact-KO development")

    ml_stack, _, v5_ll = pure.build_honest_v5_stack(v5_market_path, v5_feature_path)
    ml_stack["fight_id"] = ml_stack["fight_id"].astype(str)
    df["fight_id"] = df["fight_id"].astype(str)
    df = df.merge(ml_stack[["fight_id", "model_p_red", "market_p_red"]], on="fight_id", how="left")
    h = load_hier()

    parts = []
    metrics = []
    feature_use = []
    for fold, train_end, val_start, val_end in method.FOLDS:
        train = df[(df["date"] <= train_end) & df["model_p_red"].notna()].copy()
        val = df[(df["date"] >= val_start) & (df["date"] <= val_end) & df["model_p_red"].notna()].copy()
        val = val.merge(h, on="fight_id", how="left", validate="one_to_one")
        if train.empty or val.empty or val[["hier_red_ko", "hier_blue_ko"]].isna().any(axis=None):
            raise RuntimeError(f"incomplete exact-KO fold {fold}")
        y = target3(val)
        bas = fused_base3(val)
        mkt = method_market3(val)
        hp = hier3(val)
        p, gain, fc = fit_predict(train, val, features, False)
        pm, gainm, fcm = fit_predict(train, val, features, True)
        for name, probs in [("method_market", mkt), ("fused_base", bas), ("existing_hierarchical_v5", hp), ("xgb_residual", p), ("xgb_residual_plus_ml", pm)]:
            metrics.append({"fold": fold, "variant": name, **pure.multiclass3_metrics(y, probs)})
        feature_use.append({"fold": fold, "variant": "xgb_residual_plus_ml", "v5_ml_used": bool(gainm.get("v5_model_p_red", 0.0) > 0), "v5_ml_gain": float(gainm.get("v5_model_p_red", 0.0)), "feature_count": fcm})

        out = val[["fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target", "betting_eligible", "cold_start", "model_p_red", "market_p_red", "hier_red_ko", "hier_blue_ko"]].copy()
        out["fold"] = fold
        out["fused_red_ko"], out["fused_blue_ko"], out["fused_no_ko"] = bas[:,0], bas[:,1], bas[:,2]
        out["xgb_red_ko"], out["xgb_blue_ko"], out["xgb_no_ko"] = p[:,0], p[:,1], p[:,2]
        out["xgb_ml_red_ko"], out["xgb_ml_blue_ko"], out["xgb_ml_no_ko"] = pm[:,0], pm[:,1], pm[:,2]
        parts.append(out)

    pred = pd.concat(parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
    eval_df = pred.merge(df[["fight_id", "market_red_ko", "market_blue_ko", "market_red_sub", "market_red_dec", "market_blue_sub", "market_blue_dec"]], on="fight_id", how="left")
    y = target3(pred)
    pooled_probs = {
        "method_market": method_market3(eval_df),
        "fused_base": pred[["fused_red_ko", "fused_blue_ko", "fused_no_ko"]].to_numpy(float),
        "existing_hierarchical_v5": hier3(pred),
        "xgb_residual": pred[["xgb_red_ko", "xgb_blue_ko", "xgb_no_ko"]].to_numpy(float),
        "xgb_residual_plus_ml": pred[["xgb_ml_red_ko", "xgb_ml_blue_ko", "xgb_ml_no_ko"]].to_numpy(float),
    }
    pooled = {k: pure.multiclass3_metrics(y, v) for k, v in pooled_probs.items()}
    for name, vals in pooled.items():
        metrics.append({"fold": "pooled_2021_2024", "variant": name, **vals})

    selected = min(["xgb_residual", "xgb_residual_plus_ml"], key=lambda z: (pooled[z]["log_loss"], pooled[z]["brier"]))
    cand = build_candidates(pred)
    rule_df, robust_df = evaluate_edges(cand)

    pred.to_csv(PREDICTIONS, index=False)
    cand.to_csv(CANDIDATES, index=False)
    pd.DataFrame(metrics).to_csv(METRICS, index=False)
    rule_df.to_csv(RULES, index=False)
    robust_df.to_csv(ROBUSTNESS, index=False)

    summary = {
        "experiment": "exact3_ko_xgboost_residual_on_v5_ml_times_market_conditional_v1",
        "design": "base margin is coherent 3-class [red KO, blue KO, no KO] formed from frozen V5 P(win) multiplied by sportsbook conditional KO distribution; XGBoost learns residual correction",
        "development_window": "chronological 2021-2024 OOF only",
        "reads_2025_plus": False,
        "roi_used_for_model_selection": False,
        "features": features,
        "feature_count": len(features),
        "excluded_leakage_features": excluded,
        "hyperparameters": {**PARAMS, "num_boost_round": ROUNDS},
        "v5_canonical_oof_log_loss": v5_ll,
        "selected_xgb_variant": selected,
        "selection_metric": "pooled 2021-2024 exact 3-class log loss; Brier tiebreak",
        "pooled_probability_metrics": pooled,
        "delta_selected_vs_fused_base_log_loss": pooled[selected]["log_loss"] - pooled["fused_base"]["log_loss"],
        "delta_selected_vs_existing_hierarchical_v5_log_loss": pooled[selected]["log_loss"] - pooled["existing_hierarchical_v5"]["log_loss"],
        "v5_ml_split_feature_use": feature_use,
        "oof_fights": int(len(pred)),
        "edge_rules_diagnostic_only": True,
        "artifacts": [str(PREDICTIONS), str(CANDIDATES), str(METRICS), str(RULES), str(ROBUSTNESS)],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-market", required=True)
    ap.add_argument("--v5-features", required=True)
    args = ap.parse_args()
    run(args.v5_market, args.v5_features)
