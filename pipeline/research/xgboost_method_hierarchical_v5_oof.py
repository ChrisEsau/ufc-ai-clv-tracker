from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research.xgboost_market_offset_v5_frozen import (
    EXPECTED_OOF_LOG_LOSS,
    generate_oof as generate_v5_oof,
)
from pipeline.research import xgboost_method_market_offset as method

OUT = Path("data/research/prop_mispricing")
SUMMARY = OUT / "xgboost_method_hierarchical_v5_oof_summary.json"
PREDICTIONS = OUT / "xgboost_method_hierarchical_v5_oof_predictions.csv"
V5_OOF = OUT / "xgboost_market_offset_v5_frozen_standalone_oof.csv"

FOLDS = method.FOLDS
EPS = 1e-12
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


def _normalize3(p):
    p = np.asarray(p, float)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def _dmatrix(x, market3, y=None):
    d = xgb.DMatrix(x, label=y, feature_names=list(x.columns)) if y is not None else xgb.DMatrix(x, feature_names=list(x.columns))
    d.set_base_margin(np.log(_normalize3(market3)).reshape(-1))
    return d


def _prep(train, score, features):
    a = train[features].replace([np.inf, -np.inf], np.nan)
    b = score[features].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in features if a[c].notna().any()]
    med = a[valid].median(numeric_only=True)
    return a[valid].fillna(med).fillna(0.0), b[valid].fillna(med).fillna(0.0), valid


def _fit_conditional(train, score, features, side):
    if side == "red":
        class_idx = [0, 1, 2]
        winner_rows = train[train["target"].isin(class_idx)].copy()
        y = winner_rows["target"].to_numpy(int)
        mcols = ["market_red_ko", "market_red_sub", "market_red_dec"]
    else:
        class_idx = [3, 4, 5]
        winner_rows = train[train["target"].isin(class_idx)].copy()
        y = winner_rows["target"].to_numpy(int) - 3
        mcols = ["market_blue_ko", "market_blue_sub", "market_blue_dec"]
    if winner_rows.empty:
        raise RuntimeError(f"no {side} winner rows")
    xtr, xsc, valid = _prep(winner_rows, score, features)
    dtr = _dmatrix(xtr, winner_rows[mcols].to_numpy(float), y)
    dsc = _dmatrix(xsc, score[mcols].to_numpy(float))
    booster = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    pred = np.asarray(booster.predict(dsc), float)
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise RuntimeError(f"bad {side} conditional prediction shape {pred.shape}")
    pred = _normalize3(pred)
    return pred, len(winner_rows), len(valid)


def _metrics(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    p = p / p.sum(axis=1, keepdims=True)
    onehot = np.eye(6)[y]
    pred = p.argmax(axis=1)
    side = np.array(["red", "red", "red", "blue", "blue", "blue"])
    meth = np.array(["ko", "sub", "dec", "ko", "sub", "dec"])
    return {
        "n": int(len(y)),
        "log_loss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)))),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "top1_accuracy": float(np.mean(pred == y)),
        "winner_accuracy": float(np.mean(side[pred] == side[y])),
        "method_accuracy": float(np.mean(meth[pred] == meth[y])),
    }


def run(v5_market_path, v5_feature_path):
    OUT.mkdir(parents=True, exist_ok=True)

    # Exact frozen V5 moneyline OOF from its canonical snapshot.
    ml, v5_features, v5_ll = generate_v5_oof(v5_market_path, v5_feature_path)
    ml.to_csv(V5_OOF, index=False)
    if abs(v5_ll - EXPECTED_OOF_LOG_LOSS) > 1e-12:
        raise RuntimeError(f"standalone V5 reproduction mismatch: {v5_ll} vs {EXPECTED_OOF_LOG_LOSS}")

    df, features, _ = method._build_rows(True, True)
    # User explicitly disabled cold-start exclusion for exact-method research.
    six_oof = pd.read_csv(method.OOF_PATH)
    six_oof["fight_id"] = six_oof["fight_id"].astype(str)
    six_oof["date"] = pd.to_datetime(six_oof["date"])
    ml["fight_id"] = ml["fight_id"].astype(str)
    ml["date"] = pd.to_datetime(ml["date"])

    parts = []
    fold_summary = []
    for fold, train_end, val_start, val_end in FOLDS:
        train = df[df["date"] <= train_end].copy()
        val = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
        ml_fold = ml[ml["fold"].astype(str).eq(str(fold))][["fight_id", "model_p_red", "market_p_red"]].copy()
        val = val.merge(ml_fold, on="fight_id", how="inner")
        if val.empty:
            raise RuntimeError(f"no V5/method overlap for {fold}")

        red_cond, red_train_n, red_fc = _fit_conditional(train, val, features, "red")
        blue_cond, blue_train_n, blue_fc = _fit_conditional(train, val, features, "blue")
        p_red = np.clip(val["model_p_red"].to_numpy(float), EPS, 1 - EPS)
        p = np.column_stack([
            p_red[:, None] * red_cond,
            (1 - p_red)[:, None] * blue_cond,
        ]).reshape(len(val), 6)
        # column_stack above interleaves incorrectly for matrices; construct explicitly.
        p = np.concatenate([p_red[:, None] * red_cond, (1 - p_red)[:, None] * blue_cond], axis=1)
        p = p / p.sum(axis=1, keepdims=True)
        y = val["target"].to_numpy(int)

        base = six_oof[six_oof["fold"].astype(str).eq(str(fold))].copy()
        base = val[["fight_id"]].merge(base, on="fight_id", how="left")
        if base[[f"model_{s}" for s in method.SLUGS]].isna().any(axis=None):
            raise RuntimeError(f"missing frozen six-way baseline rows in {fold}")
        market_p = base[[f"market_{s}" for s in method.SLUGS]].to_numpy(float)
        six_p = base[[f"model_{s}" for s in method.SLUGS]].to_numpy(float)

        hm = _metrics(y, p)
        mm = _metrics(y, market_p)
        sm = _metrics(y, six_p)
        fold_summary.append({
            "fold": fold,
            "n": int(len(val)),
            "red_conditional_train_n": red_train_n,
            "blue_conditional_train_n": blue_train_n,
            "red_feature_count": red_fc,
            "blue_feature_count": blue_fc,
            "market": mm,
            "frozen_six_way": sm,
            "hierarchical_v5": hm,
            "delta_ll_hier_vs_market": hm["log_loss"] - mm["log_loss"],
            "delta_ll_hier_vs_frozen_six_way": hm["log_loss"] - sm["log_loss"],
        })
        out = val[["fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target"]].copy()
        out["fold"] = fold
        out["v5_model_p_red"] = p_red
        for j, slug in enumerate(method.SLUGS):
            out[f"hier_{slug}"] = p[:, j]
            out[f"market_{slug}"] = market_p[:, j]
            out[f"frozen_six_{slug}"] = six_p[:, j]
        parts.append(out)

    pred = pd.concat(parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
    y = pred["target"].to_numpy(int)
    hp = pred[[f"hier_{s}" for s in method.SLUGS]].to_numpy(float)
    mp = pred[[f"market_{s}" for s in method.SLUGS]].to_numpy(float)
    sp = pred[[f"frozen_six_{s}" for s in method.SLUGS]].to_numpy(float)
    overall = {
        "market": _metrics(y, mp),
        "frozen_six_way": _metrics(y, sp),
        "hierarchical_v5": _metrics(y, hp),
    }
    overall["delta_ll_hier_vs_market"] = overall["hierarchical_v5"]["log_loss"] - overall["market"]["log_loss"]
    overall["delta_ll_hier_vs_frozen_six_way"] = overall["hierarchical_v5"]["log_loss"] - overall["frozen_six_way"]["log_loss"]

    summary = {
        "experiment": "frozen_v5_moneyline_plus_conditional_method_oof_v1",
        "design": "P(side wins) from exact frozen V5 moneyline; separate red/blue 3-class XGBoost predicts KO/SUB/DEC conditional on that side winning; multiply into six-way probabilities",
        "v5_source_snapshot": "7df1b61126be1f4e036b256d1c774c531b8a281f",
        "v5_expected_oof_log_loss": EXPECTED_OOF_LOG_LOSS,
        "v5_standalone_oof_log_loss": v5_ll,
        "v5_exact_within_1e_12": abs(v5_ll - EXPECTED_OOF_LOG_LOSS) <= 1e-12,
        "method_feature_count": len(features),
        "method_hyperparameters": {**PARAMS, "num_boost_round": ROUNDS},
        "selection": "none; fixed first-pass architecture, chronological 2021-2024 OOF only",
        "roi_used": False,
        "reads_2025_plus_for_selection": False,
        "folds": fold_summary,
        "pooled": overall,
    }
    pred.to_csv(PREDICTIONS, index=False)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-market", required=True)
    ap.add_argument("--v5-features", required=True)
    args = ap.parse_args()
    run(args.v5_market, args.v5_features)
