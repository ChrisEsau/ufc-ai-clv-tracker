"""Standalone scorer for the canonical frozen V5 moneyline market-offset XGBoost.

Canonical source:
  .github/workflows/research-xgboost-market-offset-v5-feature-reduction.yml
Canonical snapshot:
  7df1b61126be1f4e036b256d1c774c531b8a281f

This module intentionally preserves V5 feature order, chronological row ordering,
train-fold median imputation, XGBoost parameters, and market-logit base-margin
semantics.  The OOF reproduction oracle is 0.600822510744624 log loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import xgboost as xgb

CANONICAL_SNAPSHOT = "7df1b61126be1f4e036b256d1c774c531b8a281f"
EXPECTED_OOF_LOG_LOSS = 0.600822510744624

# Exact selected top-50 pre-2021 gain order from canonical V5, followed by
# market_overround exactly as in the historical workflow.
SELECTED_SIGNED_FEATURES = [
    "reach_diff",
    "recent_form_recent_avg_fight_time_diff",
    "age_diff",
    "ewm_sapm_diff",
    "ewm_recent_sapm_diff",
    "style_ko_finisher_score_diff",
    "ewm_td_acc_diff",
    "recent_finish_rate_diff",
    "chin_risk_diff",
    "recent_form_avg_opponent_elo_diff",
    "recent_avg_fight_time_diff",
    "aggression_index_diff",
    "age_squared_diff",
    "sapm_diff",
    "ewm_kd_avg_diff",
    "style_all_round_finisher_score_diff",
    "recent_form_kd_absorbed_avg_diff",
    "ewm_recent_splm_diff",
    "elo_diff",
    "ewm_elo_diff",
    "ewm_recent_td_avg_diff",
    "days_since_last_fight_diff",
    "td_avg_diff",
    "style_score_spread_diff",
    "ko_dependency_diff",
    "recent_form_avg_fight_time_diff",
    "wrestling_mismatch_diff",
    "win_pct_diff",
    "recent_form_ko_rate_diff",
    "recent_form_worst_loss_elo_diff",
    "age_x_career_ko_losses_diff",
    "ewm_str_def_diff",
    "losses_diff",
    "ewm_recent_win_pct_diff",
    "avg_opponent_elo_diff",
    "ewm_td_avg_diff",
    "avg_fight_time_diff",
    "ewm_days_since_last_fight_diff",
    "pressure_striking_adv_diff",
    "weight_diff",
    "ctrl_against_per_min_diff",
    "ewm_finish_loss_rate_diff",
    "ewm_win_pct_diff",
    "victory_concentration_index_diff",
    "recent_form_td_acc_diff",
    "sub_avg_diff",
    "recent_form_best_win_elo_diff",
    "ewm_best_win_elo_diff",
    "style_primary_score_diff",
    "recent_form_recent_finish_rate_diff",
]
FEATURES = SELECTED_SIGNED_FEATURES + ["market_overround"]

PARAMS = {
    "max_depth": 1,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "lambda": 8.0,
    "alpha": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": 42,
    "nthread": 2,
}
NUM_BOOST_ROUND = 300
FOLDS = [
    ("2021", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2023-12-31", "2024-01-01", "2024-12-31"),
]


def clip_p(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def logit(p):
    p = clip_p(p)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.clip(np.asarray(z, float), -30, 30)
    return 1 / (1 + np.exp(-z))


def metrics(y, p):
    y = np.asarray(y, int)
    p = clip_p(p)
    return {
        "n": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def load_canonical_frame(
    market_path="data/market/historical_market_outcomes.parquet",
    feature_path="data/features/moneyline_feature_view.parquet",
):
    market = pd.read_parquet(market_path).copy()
    market = market[
        (market["bookmaker"] == "legacy_consensus")
        & (market["result_status"] == "graded")
        & market["won"].notna()
    ].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market["won"] = market["won"].astype(bool).astype(int)
    market["implied_probability"] = pd.to_numeric(market["implied_probability"], errors="coerce")
    market["profit_per_100"] = pd.to_numeric(market["profit_per_100"], errors="coerce")
    market = market.dropna(subset=["date", "implied_probability", "profit_per_100"]).copy()

    ml = market[market["market_key"] == "moneyline"].copy()
    good = ml.groupby("fight_id").size()
    good = good[good == 2].index
    ml = ml[ml["fight_id"].isin(good)].copy()
    ml["market_overround"] = ml.groupby("fight_id")["implied_probability"].transform("sum")
    ml["fair_market_p"] = ml["implied_probability"] / ml["market_overround"]

    red = ml[ml["outcome_side"].astype(str).eq("red")].copy()
    fv = pd.read_parquet(feature_path).copy()
    missing = [c for c in SELECTED_SIGNED_FEATURES if c not in fv.columns]
    if missing:
        raise KeyError(f"Frozen V5 features missing: {missing}")

    # Exact historical merge + chronological row ordering.
    df = red.merge(fv[["fight_id"] + SELECTED_SIGNED_FEATURES], on="fight_id", how="inner").sort_values(
        ["date", "fight_id"]
    ).copy()
    xraw = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df, xraw


def fit_fold(df, xraw, train_mask, score_mask):
    valid = [c for c in FEATURES if xraw.loc[train_mask, c].notna().any()]
    med = xraw.loc[train_mask, valid].median(numeric_only=True)
    xtr = xraw.loc[train_mask, valid].fillna(med).fillna(0.0)
    xsc = xraw.loc[score_mask, valid].fillna(med).fillna(0.0)
    ytr = df.loc[train_mask, "won"].astype(int).to_numpy()
    mtr = logit(df.loc[train_mask, "fair_market_p"])
    msc = logit(df.loc[score_mask, "fair_market_p"])
    dtr = xgb.DMatrix(xtr, label=ytr, base_margin=mtr, feature_names=valid)
    dsc = xgb.DMatrix(xsc, base_margin=msc, feature_names=valid)
    model = xgb.train(PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND, verbose_eval=False)
    full_margin = model.predict(dsc, output_margin=True)
    # Canonical V5 alpha=1 correction semantics.
    p = sigmoid(msc + (full_margin - msc))
    return model, p, sigmoid(msc), valid, med


def reproduce_oof(market_path, feature_path):
    df, xraw = load_canonical_frame(market_path, feature_path)
    parts = []
    fold_metrics = []
    for fold_name, train_end, val_start, val_end in FOLDS:
        tr = df["date"] <= train_end
        va = (df["date"] >= val_start) & (df["date"] <= val_end)
        _, p, pm, valid, _ = fit_fold(df, xraw, tr, va)
        y = df.loc[va, "won"].astype(int).to_numpy()
        mm = metrics(y, pm)
        mx = metrics(y, p)
        fold_metrics.append({
            "fold": fold_name,
            "train_n": int(tr.sum()),
            "validation_n": int(va.sum()),
            "feature_count": len(valid),
            "market": mm,
            "model": mx,
            "delta_log_loss_vs_market": float(mx["log_loss"] - mm["log_loss"]),
        })
        parts.append(pd.DataFrame({
            "fight_id": df.loc[va, "fight_id"].to_numpy(),
            "date": df.loc[va, "date"].to_numpy(),
            "fold": fold_name,
            "won": y,
            "model_p": p,
            "market_p": pm,
        }))
    oof = pd.concat(parts, ignore_index=True)
    overall = metrics(oof["won"], oof["model_p"])
    market_overall = metrics(oof["won"], oof["market_p"])
    return oof, {
        "canonical_snapshot": CANONICAL_SNAPSHOT,
        "selected_candidate": "top_50_pre2021_gain",
        "selected_features": SELECTED_SIGNED_FEATURES,
        "feature_order": FEATURES,
        "params": PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "folds": fold_metrics,
        "market_oof": market_overall,
        "v5_oof": overall,
        "expected_v5_oof_log_loss": EXPECTED_OOF_LOG_LOSS,
        "abs_error": abs(overall["log_loss"] - EXPECTED_OOF_LOG_LOSS),
        "exact_within_1e_12": abs(overall["log_loss"] - EXPECTED_OOF_LOG_LOSS) <= 1e-12,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="data/market/historical_market_outcomes.parquet")
    ap.add_argument("--features", default="data/features/moneyline_feature_view.parquet")
    ap.add_argument("--out-dir", default="data/research/prop_mispricing")
    ap.add_argument("--prefix", default="xgboost_v5_standalone_reproduction")
    ap.add_argument("--assert-exact", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof, summary = reproduce_oof(args.market, args.features)
    oof.to_csv(out / f"{args.prefix}_selected_oof.csv", index=False)
    with open(out / f"{args.prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if args.assert_exact:
        actual = float(summary["v5_oof"]["log_loss"])
        assert abs(actual - EXPECTED_OOF_LOG_LOSS) <= 1e-12, (
            f"Frozen V5 standalone reproduction mismatch: {actual} vs {EXPECTED_OOF_LOG_LOSS}"
        )


if __name__ == "__main__":
    main()
