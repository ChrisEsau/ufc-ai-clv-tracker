#!/usr/bin/env python3
"""Test canonical leakage-safe WHR as one additional frozen-V5 feature.

WHR snapshots are generated separately by the exact prior UFC WHR research
implementation. Selection here uses chronological 2021-2024 OOF log loss only.
ROI is not used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

OUT = Path("data/research/prop_mispricing")
WHR_PATH = Path("/tmp/v5_whr/fight_whr_holdout.csv")
BASE_SUMMARY = OUT / "v5_depth1_vs_depth2_summary.json"
EXPECTED_V5_LL = 0.600822510744624


def clip_p(p): return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
def logit(p):
    p = clip_p(p); return np.log(p / (1 - p))
def sigmoid(z):
    z = np.clip(np.asarray(z, float), -30, 30); return 1 / (1 + np.exp(-z))
def metrics(y, p):
    y = np.asarray(y, int); p = clip_p(p)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }
def norm_name(s): return " ".join(str(s).strip().lower().split())


def main():
    base = json.loads(BASE_SUMMARY.read_text())
    base_features = list(base["features"])
    assert len(base_features) == 51 and base_features[-1] == "market_overround"
    signed = base_features[:-1]

    market = pd.read_parquet("data/market/historical_market_outcomes.parquet").copy()
    market = market[(market["bookmaker"] == "legacy_consensus") & (market["result_status"] == "graded") & market["won"].notna()].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market["won"] = market["won"].astype(bool).astype(int)
    market["implied_probability"] = pd.to_numeric(market["implied_probability"], errors="coerce")
    market["profit_per_100"] = pd.to_numeric(market["profit_per_100"], errors="coerce")
    market = market.dropna(subset=["date", "implied_probability", "profit_per_100"]).copy()
    ml = market[market["market_key"] == "moneyline"].copy()
    good = ml.groupby("fight_id").size(); good = good[good == 2].index
    ml = ml[ml["fight_id"].isin(good)].copy()
    ml["market_overround"] = ml.groupby("fight_id")["implied_probability"].transform("sum")
    ml["fair_market_p"] = ml["implied_probability"] / ml["market_overround"]
    red = ml[ml["outcome_side"].astype(str).eq("red")].copy()

    fv = pd.read_parquet("data/features/moneyline_feature_view.parquet").copy()
    missing = [c for c in signed if c not in fv.columns]
    if missing: raise RuntimeError(f"Missing frozen V5 features: {missing}")
    df = red.merge(fv[["fight_id"] + signed], on="fight_id", how="inner")

    whr = pd.read_csv(WHR_PATH)
    whr["bout_id"] = whr["bout_id"].astype(str)
    whr["whr_diff"] = pd.to_numeric(whr["whr_red_pre_rating"], errors="coerce") - pd.to_numeric(whr["whr_blue_pre_rating"], errors="coerce")
    whr_small = whr[["bout_id", "red_fighter", "blue_fighter", "whr_diff", "whr_red_win_prob", "solver_success"]].drop_duplicates("bout_id")
    df["_fid"] = df["fight_id"].astype(str)
    df = df.merge(whr_small, left_on="_fid", right_on="bout_id", how="left", validate="one_to_one")
    df = df.sort_values(["date", "fight_id"]).copy()

    orientation = None
    if "outcome_label" in df.columns:
        chk = df[df["red_fighter"].notna() & df["outcome_label"].notna()].copy()
        if len(chk):
            orientation = float(np.mean([norm_name(a) == norm_name(b) for a, b in zip(chk["outcome_label"], chk["red_fighter"])]))
            if orientation < 0.98: raise RuntimeError(f"Red-corner orientation match too low: {orientation:.4f}")

    Xbase = df[base_features].replace([np.inf, -np.inf], np.nan)
    Xwhr = df[base_features + ["whr_diff"]].replace([np.inf, -np.inf], np.nan)
    params = {
        "max_depth": 1, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.7,
        "min_child_weight": 10, "lambda": 8.0, "alpha": 1.0,
        "objective": "binary:logistic", "eval_metric": "logloss", "seed": 42, "nthread": 2,
    }
    rounds = 300
    folds = [
        ("2021", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2022", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2023", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2024", "2023-12-31", "2024-01-01", "2024-12-31"),
    ]

    results, stores = {}, {}
    for model_name, Xraw in [("v5", Xbase), ("v5_plus_whr", Xwhr)]:
        parts, folds_out = [], []
        for fold_name, train_end, val_start, val_end in folds:
            tr = df["date"] <= train_end
            va = (df["date"] >= val_start) & (df["date"] <= val_end)
            valid = [c for c in Xraw.columns if Xraw.loc[tr, c].notna().any()]
            med = Xraw.loc[tr, valid].median(numeric_only=True)
            Xtr = Xraw.loc[tr, valid].fillna(med).fillna(0.0)
            Xva = Xraw.loc[va, valid].fillna(med).fillna(0.0)
            ytr = df.loc[tr, "won"].astype(int).to_numpy(); yva = df.loc[va, "won"].astype(int).to_numpy()
            mtr = logit(df.loc[tr, "fair_market_p"]); mva = logit(df.loc[va, "fair_market_p"])
            dtr = xgb.DMatrix(Xtr, label=ytr, base_margin=mtr, feature_names=valid)
            dva = xgb.DMatrix(Xva, label=yva, base_margin=mva, feature_names=valid)
            model = xgb.train(params, dtr, num_boost_round=rounds, verbose_eval=False)
            p = sigmoid(model.predict(dva, output_margin=True))
            mm, mx = metrics(yva, sigmoid(mva)), metrics(yva, p)
            folds_out.append({
                "fold": fold_name, "train_n": int(tr.sum()), "validation_n": int(va.sum()),
                "feature_count": int(len(valid)),
                "whr_train_nonnull": int(Xraw.loc[tr, "whr_diff"].notna().sum()) if "whr_diff" in Xraw else None,
                "whr_validation_nonnull": int(Xraw.loc[va, "whr_diff"].notna().sum()) if "whr_diff" in Xraw else None,
                "market": mm, "model": mx,
                "delta_log_loss_vs_market": float(mx["log_loss"] - mm["log_loss"]),
            })
            parts.append(pd.DataFrame({
                "fight_id": df.loc[va, "fight_id"].to_numpy(), "date": df.loc[va, "date"].to_numpy(),
                "fold": fold_name, "won": yva, "market_p": sigmoid(mva), "model_p": p,
                "whr_diff": df.loc[va, "whr_diff"].to_numpy(),
                "whr_p_red": df.loc[va, "whr_red_win_prob"].to_numpy(),
            }))
        odf = pd.concat(parts, ignore_index=True)
        results[model_name] = {"folds": folds_out, "oof": metrics(odf["won"], odf["model_p"])}
        stores[model_name] = odf

    v5_ll = results["v5"]["oof"]["log_loss"]
    if abs(v5_ll - EXPECTED_V5_LL) > 1e-12:
        raise RuntimeError(f"V5 reproduction gate failed: {v5_ll} vs {EXPECTED_V5_LL}")
    whr_ll = results["v5_plus_whr"]["oof"]["log_loss"]
    oof = stores["v5_plus_whr"]
    summary = {
        "experiment": "frozen_v5_plus_canonical_whr_diff_v1",
        "selection_objective": "2021-2024 chronological OOF log loss only; ROI not used",
        "v5_source_commit": "7df1b61126be1f4e036b256d1c774c531b8a281f",
        "whr_source_commit": "8f18b071a0d4913b165471a3133a5464776e680c",
        "whr_spec": {
            "type": "leakage-safe FightMatrix-style dynamic Bradley-Terry whole-history MAP",
            "w": 2.75, "w2_elo2_per_day": 7.5625, "starter_rating": 0.0,
            "split": [0.55, 0.45], "majority": [0.61, 0.39], "unanimous": [0.91, 0.09], "other_win": [1.0, 0.0],
            "same_day_rule": "fit only bouts strictly before target event date", "convergence_gtol": 1e-5,
        },
        "feature_tested": "whr_diff = prefight red WHR - prefight blue WHR",
        "rows": int(len(df)), "whr_nonnull_rows": int(df["whr_diff"].notna().sum()),
        "whr_oof_nonnull": int(oof["whr_diff"].notna().sum()), "red_corner_orientation_match": orientation,
        "standalone_whr_oof": metrics(oof["won"], oof["whr_p_red"]),
        "models": results,
        "comparison": {
            "v5_oof_log_loss": float(v5_ll), "v5_plus_whr_oof_log_loss": float(whr_ll),
            "whr_minus_v5_log_loss": float(whr_ll - v5_ll), "winner": "v5_plus_whr" if whr_ll < v5_ll else "v5",
        },
    }
    (OUT / "v5_whr_feature_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    oof.to_csv(OUT / "v5_whr_feature_test_oof.csv", index=False)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
