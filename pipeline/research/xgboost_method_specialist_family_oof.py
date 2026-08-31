from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research.xgboost_method_market_offset import (
    ROOT,
    FOLDS,
    PARAMS,
    ROUNDS,
    EPS,
    CLASS_ORDER,
    MARKET_COLS,
    _build_rows,
    _metrics,
    _calibration,
)

FEATURE_LIST_PATH = ROOT / "xgboost_method_market_offset__feature_list.json"
OUT_PRED = ROOT / "xgboost_method_market_offset__specialist_family_oof_predictions.csv"
OUT_SUMMARY = ROOT / "xgboost_method_market_offset__specialist_family_oof_summary.json"

# Six independent binary market-offset specialists, grouped by method family.
# Each class specialist learns a residual log-odds correction against that class's
# normalized six-way market probability. The six corrected class scores are then
# normalized with a common softmax so the final prediction is coherent.
SPECIALISTS = [
    (0, "RED_KO_TKO", "KO_TKO"),
    (3, "BLUE_KO_TKO", "KO_TKO"),
    (1, "RED_SUB", "SUB"),
    (4, "BLUE_SUB", "SUB"),
    (2, "RED_DEC", "DEC"),
    (5, "BLUE_DEC", "DEC"),
]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def load_frozen_features() -> list[str]:
    obj = json.loads(FEATURE_LIST_PATH.read_text())
    if isinstance(obj, list):
        return [str(x) for x in obj]
    for key in ("features", "selected_features", "feature_list"):
        if key in obj and isinstance(obj[key], list):
            return [str(x) for x in obj[key]]
    raise RuntimeError(f"could not find frozen feature list in {FEATURE_LIST_PATH}")


def prep(train: pd.DataFrame, val: pd.DataFrame, features: list[str]):
    a = train[features].replace([np.inf, -np.inf], np.nan)
    b = val[features].replace([np.inf, -np.inf], np.nan)
    valid = [c for c in features if a[c].notna().any()]
    med = a[valid].median(numeric_only=True)
    return a[valid].fillna(med).fillna(0.0), b[valid].fillna(med).fillna(0.0), valid


def fit_binary(train: pd.DataFrame, val: pd.DataFrame, xtr: pd.DataFrame, xva: pd.DataFrame, class_idx: int) -> tuple[np.ndarray, np.ndarray]:
    y = (train["target"].to_numpy(int) == class_idx).astype(float)
    market_train = train[MARKET_COLS[class_idx]].to_numpy(float)
    market_val = val[MARKET_COLS[class_idx]].to_numpy(float)

    dtr = xgb.DMatrix(xtr, label=y, feature_names=list(xtr.columns))
    dva = xgb.DMatrix(xva, feature_names=list(xva.columns))
    dtr.set_base_margin(logit(market_train))
    dva.set_base_margin(logit(market_val))

    params = dict(PARAMS)
    params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    params.pop("num_class", None)
    booster = xgb.train(params, dtr, num_boost_round=ROUNDS)

    prob = np.asarray(booster.predict(dva), dtype=float)
    corrected_logit = logit(prob)
    residual = corrected_logit - logit(market_val)
    return prob, residual


def main() -> None:
    features = load_frozen_features()
    df, _, _ = _build_rows(development_only=True, include_targets=True, forced_features=features)

    rows = []
    fold_metrics = []
    for fold, train_end, val_start, val_end in FOLDS:
        train = df[df["date"] <= pd.Timestamp(train_end)].copy()
        val = df[(df["date"] >= pd.Timestamp(val_start)) & (df["date"] <= pd.Timestamp(val_end))].copy()
        if train.empty or val.empty:
            raise RuntimeError(f"empty train/val for fold {fold}")

        xtr, xva, valid = prep(train, val, features)
        market = val[MARKET_COLS].to_numpy(float)
        residual = np.zeros_like(market)
        raw_binary_prob = np.zeros_like(market)

        for class_idx, class_name, family in SPECIALISTS:
            p, r = fit_binary(train, val, xtr, xva, class_idx)
            raw_binary_prob[:, class_idx] = p
            residual[:, class_idx] = r

        # Common six-way score scale: log six-way market probability + specialist
        # residual log-odds correction; common softmax restores mutual exclusivity.
        scores = np.log(np.clip(market, EPS, 1.0)) + residual
        combined = softmax(scores)

        y = val["target"].to_numpy(int)
        fm = {
            "fold": fold,
            "n": int(len(val)),
            "market": _metrics(y, market),
            "specialist": _metrics(y, combined),
            "delta_log_loss": float(_metrics(y, combined)["log_loss"] - _metrics(y, market)["log_loss"]),
            "features": int(len(valid)),
        }
        fold_metrics.append(fm)

        for i, (_, r) in enumerate(val.iterrows()):
            out = {
                "fight_id": r["fight_id"],
                "date": r["date"].date().isoformat(),
                "event_name": r["event_name"],
                "red_fighter": r["red_fighter"],
                "blue_fighter": r["blue_fighter"],
                "target": int(r["target"]),
                "actual_class": CLASS_ORDER[int(r["target"])],
                "fold": fold,
            }
            for j, cname in enumerate(CLASS_ORDER):
                slug = cname.lower()
                out[f"market_{slug}"] = float(market[i, j])
                out[f"binary_{slug}"] = float(raw_binary_prob[i, j])
                out[f"residual_{slug}"] = float(residual[i, j])
                out[f"specialist_{slug}"] = float(combined[i, j])
            rows.append(out)

    pred = pd.DataFrame(rows).sort_values(["date", "fight_id"]).reset_index(drop=True)
    if len(pred) != 1604:
        raise RuntimeError(f"expected 1604 pooled OOF fights, got {len(pred)}")

    y = pred["target"].to_numpy(int)
    market_cols = [f"market_{c.lower()}" for c in CLASS_ORDER]
    model_cols = [f"specialist_{c.lower()}" for c in CLASS_ORDER]
    market = pred[market_cols].to_numpy(float)
    model = pred[model_cols].to_numpy(float)

    pooled_market = _metrics(y, market)
    pooled_model = _metrics(y, model)
    summary = {
        "experiment": "six_way_method_specialist_family_market_offset_v1",
        "period": "chronological 2021-2024 OOF only",
        "architecture": "six binary class specialists grouped KO/SUB/DEC; each learns log-odds residual vs normalized six-way market; six corrected scores combined by common softmax",
        "candidate_selection": "none; direct architecture comparison to frozen FULL six-way baseline",
        "reads_2025_plus": False,
        "uses_roi": False,
        "cold_start_filter": "OFF, matching frozen method experiment",
        "features": len(features),
        "rounds_per_specialist": ROUNDS,
        "pooled_market": pooled_market,
        "pooled_specialist": pooled_model,
        "delta_specialist_minus_market": {k: float(pooled_model[k] - pooled_market[k]) for k in ("log_loss", "brier", "top1_accuracy", "winner_accuracy", "method_accuracy")},
        "baseline_frozen_sixway_log_loss": 1.5364276221525495,
        "delta_specialist_minus_frozen_sixway_log_loss": float(pooled_model["log_loss"] - 1.5364276221525495),
        "beats_frozen_sixway_on_primary_metric": bool(pooled_model["log_loss"] < 1.5364276221525495),
        "folds": fold_metrics,
        "calibration": _calibration(y, model),
    }

    OUT_PRED.write_text(pred.to_csv(index=False))
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
