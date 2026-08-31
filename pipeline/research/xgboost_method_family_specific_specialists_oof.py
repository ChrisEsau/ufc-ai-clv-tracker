from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research.xgboost_method_market_offset import (
    ROOT, FOLDS, PARAMS, EPS, CLASS_ORDER, MARKET_COLS,
    _build_rows, _metrics, _calibration,
)

FEATURE_LIST_PATH = ROOT / "xgboost_method_market_offset__feature_list.json"
OUT_GRID = ROOT / "xgboost_method_market_offset__family_specific_specialists_grid.csv"
OUT_PRED = ROOT / "xgboost_method_market_offset__family_specific_specialists_oof_predictions.csv"
OUT_SUMMARY = ROOT / "xgboost_method_market_offset__family_specific_specialists_oof_summary.json"
OUT_FEATURES = ROOT / "xgboost_method_market_offset__family_specific_specialists_features.json"

FAMILIES = {
    "KO_TKO": [0, 3],
    "SUB": [1, 4],
    "DEC": [2, 5],
}

# Small development grid only. Family choice is made from chronological 2021-2024
# OOF binary log loss, never ROI and never 2025+.
FEATURE_COUNTS = [140, 75, 35]
CAPACITY = [
    {"name": "d1_r150", "max_depth": 1, "rounds": 150},
    {"name": "d1_r300", "max_depth": 1, "rounds": 300},
    {"name": "d2_r150", "max_depth": 2, "rounds": 150},
    {"name": "d2_r300", "max_depth": 2, "rounds": 300},
]


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def binary_ll(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def load_features():
    obj = json.loads(FEATURE_LIST_PATH.read_text())
    return list(obj["features"] if isinstance(obj, dict) else obj)


def prep(train, val, features):
    a = train[features].replace([np.inf, -np.inf], np.nan)
    b = val[features].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in features if a[c].notna().any()]
    med = a[valid].median(numeric_only=True)
    return a[valid].fillna(med).fillna(0.0), b[valid].fillna(med).fillna(0.0), valid


def fit_binary(train, val, features, class_idx, depth, rounds):
    xtr, xva, valid = prep(train, val, features)
    y = (train["target"].to_numpy(int) == class_idx).astype(float)
    mp_tr = train[MARKET_COLS[class_idx]].to_numpy(float)
    mp_va = val[MARKET_COLS[class_idx]].to_numpy(float)
    dtr = xgb.DMatrix(xtr, label=y, feature_names=valid)
    dva = xgb.DMatrix(xva, feature_names=valid)
    dtr.set_base_margin(logit(mp_tr))
    dva.set_base_margin(logit(mp_va))
    params = dict(PARAMS)
    params.update({"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": depth})
    params.pop("num_class", None)
    booster = xgb.train(params, dtr, num_boost_round=rounds, verbose_eval=False)
    p = np.asarray(booster.predict(dva), float)
    residual = logit(p) - logit(mp_va)
    return p, residual, booster, valid


def rank_family_features(df, all_features, class_indices):
    train = df[df["date"] <= pd.Timestamp("2020-12-31")].copy()
    if train.empty:
        raise RuntimeError("no pre-2021 rows for family feature ranking")
    gains = {c: 0.0 for c in all_features}
    # Aggregate gain from the red and blue class specialist for this family.
    for class_idx in class_indices:
        xtr, _, valid = prep(train, train, all_features)
        y = (train["target"].to_numpy(int) == class_idx).astype(float)
        mp = train[MARKET_COLS[class_idx]].to_numpy(float)
        d = xgb.DMatrix(xtr, label=y, feature_names=valid)
        d.set_base_margin(logit(mp))
        params = dict(PARAMS)
        params.update({"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 1})
        params.pop("num_class", None)
        booster = xgb.train(params, d, num_boost_round=300, verbose_eval=False)
        g = booster.get_score(importance_type="gain")
        total = sum(float(v) for v in g.values()) or 1.0
        for c in valid:
            gains[c] += float(g.get(c, 0.0)) / total
    return sorted(all_features, key=lambda c: (-gains[c], c)), gains


def evaluate_candidate(df, family, class_indices, features, depth, rounds):
    fold_parts = []
    class_losses = []
    market_losses = []
    for fold, train_end, val_start, val_end in FOLDS:
        train = df[df["date"] <= pd.Timestamp(train_end)].copy()
        val = df[(df["date"] >= pd.Timestamp(val_start)) & (df["date"] <= pd.Timestamp(val_end))].copy()
        part = val[["fight_id", "target"]].copy()
        part["fold"] = fold
        for idx in class_indices:
            p, r, _, valid = fit_binary(train, val, features, idx, depth, rounds)
            y = (val["target"].to_numpy(int) == idx).astype(float)
            mp = val[MARKET_COLS[idx]].to_numpy(float)
            class_losses.append((len(val), binary_ll(y, p)))
            market_losses.append((len(val), binary_ll(y, mp)))
            part[f"prob_{idx}"] = p
            part[f"resid_{idx}"] = r
            part[f"features_{idx}"] = len(valid)
        fold_parts.append(part)
    pooled = pd.concat(fold_parts, ignore_index=True)
    w = sum(n for n, _ in class_losses)
    model_ll = sum(n * ll for n, ll in class_losses) / w
    market_ll = sum(n * ll for n, ll in market_losses) / w
    return {
        "family": family,
        "feature_count_requested": len(features),
        "max_depth": depth,
        "rounds": rounds,
        "binary_log_loss": model_ll,
        "market_binary_log_loss": market_ll,
        "delta_vs_market": model_ll - market_ll,
    }, pooled


def main():
    all_features = load_features()
    df, _, _ = _build_rows(development_only=True, include_targets=True, forced_features=all_features)
    if (df["date"] > pd.Timestamp("2024-12-31")).any():
        raise RuntimeError("2025+ entered development")

    family_rankings = {}
    feature_meta = {}
    grid_rows = []
    candidate_ledgers = {}

    for family, indices in FAMILIES.items():
        ranked, gains = rank_family_features(df, all_features, indices)
        family_rankings[family] = ranked
        feature_meta[family] = {
            "ranking_cutoff": "2020-12-31",
            "ranked_features": ranked,
            "normalized_gain_sum": {k: float(gains[k]) for k in ranked},
        }
        for requested in FEATURE_COUNTS:
            cols = ranked[: min(requested, len(ranked))]
            for cap in CAPACITY:
                metrics, ledger = evaluate_candidate(
                    df, family, indices, cols, cap["max_depth"], cap["rounds"]
                )
                name = f"{family}_top{len(cols)}_{cap['name']}"
                metrics["candidate"] = name
                grid_rows.append(metrics)
                candidate_ledgers[name] = ledger

    grid = pd.DataFrame(grid_rows)
    selected = {}
    for family in FAMILIES:
        g = grid[grid["family"] == family].copy()
        # Primary: pooled chronological OOF binary LL. Tie breakers: fewer features,
        # shallower tree, fewer rounds.
        g = g.sort_values(
            ["binary_log_loss", "feature_count_requested", "max_depth", "rounds", "candidate"],
            ascending=[True, True, True, True, True],
        )
        selected[family] = g.iloc[0].to_dict()

    # Reassemble six residuals from independently selected family candidates.
    base = df[(df["date"] >= pd.Timestamp("2021-01-01")) & (df["date"] <= pd.Timestamp("2024-12-31"))].copy()
    base = base[["fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target"] + MARKET_COLS].copy()
    for family, indices in FAMILIES.items():
        name = selected[family]["candidate"]
        led = candidate_ledgers[name][["fight_id"] + [f"resid_{i}" for i in indices]].copy()
        base = base.merge(led, on="fight_id", how="inner", validate="one_to_one")

    if len(base) != 1604:
        raise RuntimeError(f"expected 1604 OOF fights, got {len(base)}")
    market = base[MARKET_COLS].to_numpy(float)
    residual = np.zeros_like(market)
    for family, indices in FAMILIES.items():
        for idx in indices:
            residual[:, idx] = base[f"resid_{idx}"].to_numpy(float)
    combined = softmax(np.log(np.clip(market, EPS, 1.0)) + residual)
    y = base["target"].to_numpy(int)

    for j, cname in enumerate(CLASS_ORDER):
        slug = cname.lower()
        base[f"specialist_{slug}"] = combined[:, j]
        base[f"residual_{slug}"] = residual[:, j]

    pooled_market = _metrics(y, market)
    pooled_model = _metrics(y, combined)

    # Fold metrics for final selected combination.
    folds = []
    for fold, _, val_start, val_end in FOLDS:
        mask = (base["date"] >= pd.Timestamp(val_start)) & (base["date"] <= pd.Timestamp(val_end))
        yy = y[mask]
        mm = market[mask]
        xx = combined[mask]
        folds.append({
            "fold": fold,
            "n": int(mask.sum()),
            "market": _metrics(yy, mm),
            "family_specific_specialist": _metrics(yy, xx),
            "delta_log_loss": float(_metrics(yy, xx)["log_loss"] - _metrics(yy, mm)["log_loss"]),
        })

    baseline_sixway = 1.5364276221525495
    fixed_specialist = 1.5417424341858281
    summary = {
        "experiment": "six_way_method_family_specific_specialists_v1",
        "period": "chronological 2021-2024 OOF only",
        "reads_2025_plus": False,
        "uses_roi": False,
        "cold_start_filter": "OFF, matching frozen method experiment",
        "feature_universe": len(all_features),
        "family_feature_ranking_cutoff": "2020-12-31",
        "selection_metric": "pooled 2021-2024 OOF binary log loss within each family; no ROI",
        "selection_note": "development selection is not nested; final comparison remains development OOF architecture research",
        "grid": {
            "feature_counts": FEATURE_COUNTS,
            "capacity": CAPACITY,
            "candidates_per_family": int(len(FEATURE_COUNTS) * len(CAPACITY)),
        },
        "selected": selected,
        "pooled_market": pooled_market,
        "pooled_family_specific_specialist": pooled_model,
        "delta_vs_market": {k: float(pooled_model[k] - pooled_market[k]) for k in ("log_loss", "brier", "top1_accuracy", "winner_accuracy", "method_accuracy")},
        "fixed_specialist_log_loss": fixed_specialist,
        "delta_vs_fixed_specialist_log_loss": float(pooled_model["log_loss"] - fixed_specialist),
        "frozen_sixway_log_loss": baseline_sixway,
        "delta_vs_frozen_sixway_log_loss": float(pooled_model["log_loss"] - baseline_sixway),
        "beats_fixed_specialist": bool(pooled_model["log_loss"] < fixed_specialist),
        "beats_frozen_sixway_primary": bool(pooled_model["log_loss"] < baseline_sixway),
        "folds": folds,
        "calibration": _calibration(y, combined),
    }

    OUT_GRID.write_text(grid.sort_values(["family", "binary_log_loss"]).to_csv(index=False))
    OUT_PRED.write_text(base.sort_values(["date", "fight_id"]).to_csv(index=False))
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    OUT_FEATURES.write_text(json.dumps(feature_meta, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
