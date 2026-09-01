from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research import xgboost_method_market_offset as method

ROOT = Path("data/research/prop_mispricing")
FEATURE_LIST_PATH = ROOT / "xgboost_method_market_offset__feature_list.json"
HIER_OOF_PATH = ROOT / "xgboost_method_hierarchical_v5_oof_predictions.csv"
PRED_PATH = ROOT / "xgboost_ko_method_full_history_specialist_predictions.csv"
METRICS_PATH = ROOT / "xgboost_ko_method_full_history_specialist_metrics.csv"
IMPORTANCE_PATH = ROOT / "xgboost_ko_method_full_history_specialist_feature_importance.csv"
SUMMARY_PATH = ROOT / "xgboost_ko_method_full_history_specialist_summary.json"

FOLDS = method.FOLDS
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


def clip_p(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=int)
    p = clip_p(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def auc(y, p) -> float | None:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy(dtype=float)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def metric_row(scope: str, variant: str, family: str, y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "scope": scope,
        "variant": variant,
        "family": family,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "observed_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "auc": auc(y, p),
        "calibration_error_mean_p_minus_rate": float(p.mean() - y.mean()),
    }


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    spec = json.loads(FEATURE_LIST_PATH.read_text())
    features = list(spec["features"])
    if len(features) != 140 or len(set(features)) != 140:
        raise RuntimeError(f"expected frozen 140-feature list, got {len(features)}")
    if any(not c.endswith("_diff") for c in features):
        raise RuntimeError("frozen feature list contains non-difference feature")

    # This is the same historical labeled/feature universe used by the V5 method model.
    full, forced, _ = method._build_rows(True, True, forced_features=features)
    if forced != features:
        raise RuntimeError("forced feature order changed")
    full["date"] = pd.to_datetime(full["date"], errors="raise")
    if (full["date"] > pd.Timestamp("2024-12-31")).any():
        raise RuntimeError("2025+ entered KO development")
    full["actual_winner_side"] = np.where(full["target"].astype(int) <= 2, "red", "blue")
    full["actual_ko"] = full["target"].astype(int).isin([0, 3]).astype(int)
    full["actual_red_ko"] = full["target"].astype(int).eq(0).astype(int)
    full["actual_blue_ko"] = full["target"].astype(int).eq(3).astype(int)

    h = pd.read_csv(HIER_OOF_PATH)
    required = {
        "fight_id", "date", "fold", "target", "v5_model_p_red", "hier_red_ko", "hier_blue_ko"
    }
    missing = sorted(required - set(h.columns))
    if missing:
        raise RuntimeError(f"hierarchical OOF missing columns: {missing}")
    h["fight_id"] = h["fight_id"].astype(str)
    h["date"] = pd.to_datetime(h["date"], errors="raise")
    h["fold"] = pd.to_numeric(h["fold"], errors="raise").astype(int)
    h["target"] = pd.to_numeric(h["target"], errors="raise").astype(int)
    h["v5_model_p_red"] = pd.to_numeric(h["v5_model_p_red"], errors="raise")
    h["hier_red_ko"] = pd.to_numeric(h["hier_red_ko"], errors="raise")
    h["hier_blue_ko"] = pd.to_numeric(h["hier_blue_ko"], errors="raise")
    if h["fight_id"].duplicated().any():
        raise RuntimeError("duplicate hierarchical OOF fight_id")
    if set(h["fold"].unique()) != {2021, 2022, 2023, 2024}:
        raise RuntimeError(f"unexpected hierarchical folds: {sorted(h['fold'].unique())}")
    if not (h["date"].dt.year == h["fold"]).all():
        raise RuntimeError("hierarchical OOF fold/date mismatch")
    if not ((h["v5_model_p_red"] > 0) & (h["v5_model_p_red"] < 1)).all():
        raise RuntimeError("invalid V5 ML probabilities")

    base_cols = [
        "fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target",
        "betting_eligible", "cold_start", "red_prior_ufc_fights", "blue_prior_ufc_fights",
    ] + features
    val_features = full[base_cols].copy()
    val_features["fight_id"] = val_features["fight_id"].astype(str)
    scored = h.merge(val_features, on="fight_id", how="left", validate="one_to_one", suffixes=("", "_full"), indicator=True)
    if not scored["_merge"].eq("both").all():
        bad = scored.loc[scored["_merge"].ne("both"), "fight_id"].head(20).tolist()
        raise RuntimeError(f"hierarchical OOF fights missing full-history features: {bad}")
    scored = scored.drop(columns="_merge")
    if "date_full" in scored.columns:
        if not (pd.to_datetime(scored["date_full"]).dt.normalize() == scored["date"].dt.normalize()).all():
            raise RuntimeError("OOF/full-history date mismatch")
    if "target_full" in scored.columns:
        if not (scored["target_full"].astype(int) == scored["target"].astype(int)).all():
            raise RuntimeError("OOF/full-history target mismatch")
    scored["actual_winner_side"] = np.where(scored["target"].astype(int) <= 2, "red", "blue")
    scored["actual_ko"] = scored["target"].astype(int).isin([0, 3]).astype(int)
    scored["actual_red_ko"] = scored["target"].astype(int).eq(0).astype(int)
    scored["actual_blue_ko"] = scored["target"].astype(int).eq(3).astype(int)
    scored["projected_winner_side"] = np.where(scored["v5_model_p_red"] >= 0.5, "red", "blue")
    scored["projected_winner_ko"] = np.where(
        scored["projected_winner_side"].eq("red"), scored["actual_red_ko"], scored["actual_blue_ko"]
    ).astype(int)
    return full.sort_values(["date", "fight_id"]).reset_index(drop=True), scored.sort_values(["date", "fight_id"]).reset_index(drop=True), features


def oriented_x(frame: pd.DataFrame, features: list[str], side) -> pd.DataFrame:
    x = frame[features].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    if isinstance(side, str):
        if side == "red":
            return x
        if side == "blue":
            return -x
        raise ValueError(side)
    s = pd.Series(side, index=frame.index).astype(str)
    sign = np.where(s.eq("red"), 1.0, -1.0)
    return x.mul(sign, axis=0)


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[xgb.Booster, np.ndarray, np.ndarray, list[str]]:
    xtr = oriented_x(train, features, train["actual_winner_side"])
    ytr = train["actual_ko"].to_numpy(dtype=int)
    valid = [c for c in features if xtr[c].notna().any()]
    med = xtr[valid].median(numeric_only=True)
    tr = xtr[valid].fillna(med).fillna(0.0)

    xr = oriented_x(val, features, "red")[valid].fillna(med).fillna(0.0)
    xb = oriented_x(val, features, "blue")[valid].fillna(med).fillna(0.0)
    dtr = xgb.DMatrix(tr, label=ytr, feature_names=valid)
    dr = xgb.DMatrix(xr, feature_names=valid)
    db = xgb.DMatrix(xb, feature_names=valid)
    model = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    pr = np.asarray(model.predict(dr), dtype=float)
    pb = np.asarray(model.predict(db), dtype=float)
    if not np.isfinite(pr).all() or not np.isfinite(pb).all():
        raise RuntimeError("non-finite KO probability")
    if (pr < 0).any() or (pr > 1).any() or (pb < 0).any() or (pb > 1).any():
        raise RuntimeError("KO probability outside [0,1]")
    return model, pr, pb, valid


def importance_rows(model: xgb.Booster, fold: str, features: list[str]) -> list[dict]:
    gain = model.get_score(importance_type="gain")
    weight = model.get_score(importance_type="weight")
    cover = model.get_score(importance_type="cover")
    ordered = sorted(features, key=lambda c: (-float(gain.get(c, 0.0)), c))
    rank = {c: i + 1 for i, c in enumerate(ordered)}
    return [{
        "fold": fold,
        "feature": c,
        "gain": float(gain.get(c, 0.0)),
        "weight": float(weight.get(c, 0.0)),
        "cover": float(cover.get(c, 0.0)),
        "gain_rank": int(rank[c]),
        "used": bool(c in gain or c in weight),
    } for c in features]


def metrics_for_scope(d: pd.DataFrame, scope: str) -> list[dict]:
    rows = []
    variants = {
        "existing_hierarchical_v5": (
            "existing_cond_ko_red", "existing_cond_ko_blue", "existing_exact_ko_red", "existing_exact_ko_blue"
        ),
        "full_history_ko_specialist": (
            "specialist_cond_ko_red", "specialist_cond_ko_blue", "specialist_exact_ko_red", "specialist_exact_ko_blue"
        ),
    }
    actual_red = d["actual_winner_side"].eq("red").to_numpy()
    projected_red = d["projected_winner_side"].eq("red").to_numpy()
    for variant, (cr, cb, er, eb) in variants.items():
        cond = np.where(actual_red, d[cr].to_numpy(float), d[cb].to_numpy(float))
        rows.append(metric_row(scope, variant, "conditional_ko_given_actual_win", d["actual_ko"], cond))

        y_side = np.concatenate([d["actual_red_ko"].to_numpy(int), d["actual_blue_ko"].to_numpy(int)])
        p_side = np.concatenate([d[er].to_numpy(float), d[eb].to_numpy(float)])
        rows.append(metric_row(scope, variant, "exact_side_ko_two_rows_per_fight", y_side, p_side))

        fight_p = d[er].to_numpy(float) + d[eb].to_numpy(float)
        rows.append(metric_row(scope, variant, "fight_ko_event", d["actual_ko"], fight_p))

        projected_p = np.where(projected_red, d[er].to_numpy(float), d[eb].to_numpy(float))
        rows.append(metric_row(scope, variant, "projected_winner_exact_ko", d["projected_winner_ko"], projected_p))
    return rows


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    full, scored, features = load()

    parts = []
    imp = []
    fold_audit = []
    for fold, train_end, val_start, val_end in FOLDS:
        year = int(fold)
        train = full[full["date"] <= pd.Timestamp(train_end)].copy()
        val = scored[scored["fold"].eq(year)].copy()
        if train.empty or val.empty:
            raise RuntimeError(f"empty fold {fold}")
        if train["date"].max() >= val["date"].min():
            raise RuntimeError(f"non-chronological fold {fold}")

        model, pr, pb, valid = fit_predict(train, val, features)
        pmlr = clip_p(val["v5_model_p_red"])
        pmlb = clip_p(1.0 - val["v5_model_p_red"])
        out = val[[
            "fight_id", "date", "fold", "event_name", "red_fighter", "blue_fighter", "target",
            "betting_eligible", "cold_start", "v5_model_p_red", "hier_red_ko", "hier_blue_ko",
            "actual_winner_side", "actual_ko", "actual_red_ko", "actual_blue_ko",
            "projected_winner_side", "projected_winner_ko",
        ]].copy()
        out["train_through"] = train_end
        out["train_n"] = int(len(train))
        out["specialist_cond_ko_red"] = pr
        out["specialist_cond_ko_blue"] = pb
        out["specialist_exact_ko_red"] = pmlr * pr
        out["specialist_exact_ko_blue"] = pmlb * pb
        out["existing_exact_ko_red"] = out["hier_red_ko"].to_numpy(float)
        out["existing_exact_ko_blue"] = out["hier_blue_ko"].to_numpy(float)
        out["existing_cond_ko_red"] = np.clip(out["existing_exact_ko_red"].to_numpy(float) / pmlr, 0.0, 1.0)
        out["existing_cond_ko_blue"] = np.clip(out["existing_exact_ko_blue"].to_numpy(float) / pmlb, 0.0, 1.0)
        parts.append(out)
        imp.extend(importance_rows(model, fold, features))
        fold_audit.append({
            "fold": fold,
            "train_through": train_end,
            "train_n": int(len(train)),
            "train_ko": int(train["actual_ko"].sum()),
            "train_ko_rate": float(train["actual_ko"].mean()),
            "validation_n": int(len(val)),
            "validation_ko": int(val["actual_ko"].sum()),
            "feature_count": int(len(valid)),
        })

    pred = pd.concat(parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
    if pred["fight_id"].duplicated().any():
        raise RuntimeError("duplicate OOF predictions")
    if set(pred["fold"].astype(int).unique()) != {2021, 2022, 2023, 2024}:
        raise RuntimeError("missing OOF fold")
    pred.to_csv(PRED_PATH, index=False)

    rows = []
    for year in [2021, 2022, 2023, 2024]:
        rows.extend(metrics_for_scope(pred[pred["fold"].astype(int).eq(year)], str(year)))
    rows.extend(metrics_for_scope(pred, "pooled_2021_2024"))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_PATH, index=False)
    pd.DataFrame(imp).to_csv(IMPORTANCE_PATH, index=False)

    pooled = metrics[metrics["scope"].eq("pooled_2021_2024")]
    nested = {}
    for _, r in pooled.iterrows():
        nested.setdefault(str(r["family"]), {})[str(r["variant"])] = {
            "n": int(r["n"]),
            "positives": int(r["positives"]),
            "observed_rate": float(r["observed_rate"]),
            "mean_probability": float(r["mean_probability"]),
            "log_loss": float(r["log_loss"]),
            "brier": float(r["brier"]),
            "auc": None if pd.isna(r["auc"]) else float(r["auc"]),
            "calibration_error_mean_p_minus_rate": float(r["calibration_error_mean_p_minus_rate"]),
        }
    deltas = {}
    for family, vals in nested.items():
        s = vals["full_history_ko_specialist"]
        v = vals["existing_hierarchical_v5"]
        deltas[family] = {
            "specialist_minus_v5_log_loss": s["log_loss"] - v["log_loss"],
            "specialist_minus_v5_brier": s["brier"] - v["brier"],
            "specialist_minus_v5_auc": None if s["auc"] is None or v["auc"] is None else s["auc"] - v["auc"],
        }

    summary = {
        "experiment": "full_history_conditional_ko_specialist_v1",
        "design": "binary P(KO|side wins) XGBoost; train row oriented to historical actual winner; score both hypothetical winner sides; exact side KO = frozen V5 OOF P(side wins) * specialist P(KO|side wins)",
        "training_universe": "same full historical exact-method labeled feature universe used by V5, through each fold cutoff",
        "development_window": "chronological 2021-2024 OOF; all training strictly before scored fold",
        "reads_2025_plus": False,
        "market_odds_used_as_model_feature_or_margin": False,
        "v5_ml_used_as_specialist_feature": False,
        "v5_ml_used_only_for_probability_factorization": True,
        "roi_used_for_model_selection": False,
        "feature_count": len(features),
        "features": features,
        "hyperparameters": {**PARAMS, "num_boost_round": ROUNDS},
        "fold_audit": fold_audit,
        "pooled_2021_2024": nested,
        "deltas_specialist_minus_existing_v5": deltas,
        "selection_rule": "none; benchmark only",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "fold_audit": fold_audit,
        "pooled_2021_2024": nested,
        "deltas": deltas,
    }, indent=2))


if __name__ == "__main__":
    main()
