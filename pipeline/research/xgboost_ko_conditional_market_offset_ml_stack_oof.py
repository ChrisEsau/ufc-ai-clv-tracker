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
PREDICTIONS = OUT / "xgboost_ko_conditional_market_offset_ml_stack_oof_predictions.csv"
CANDIDATES = OUT / "xgboost_ko_conditional_market_offset_ml_stack_edge_candidates.csv"
METRICS = OUT / "xgboost_ko_conditional_market_offset_ml_stack_metrics.csv"
RULES = OUT / "xgboost_ko_conditional_market_offset_ml_stack_edge_rules.csv"
ROBUSTNESS = OUT / "xgboost_ko_conditional_market_offset_ml_stack_edge_robustness.csv"
SUMMARY = OUT / "xgboost_ko_conditional_market_offset_ml_stack_summary.json"
HIER_OOF = OUT / "xgboost_method_hierarchical_v5_oof_predictions.csv"

PARAMS = dict(pure.PARAMS)
ROUNDS = pure.ROUNDS
EPS = pure.EPS


def side_market_q(frame: pd.DataFrame, side: str) -> np.ndarray:
    cols = [f"market_{side}_ko", f"market_{side}_sub", f"market_{side}_dec"]
    total = frame[cols].sum(axis=1).to_numpy(float)
    return frame[f"market_{side}_ko"].to_numpy(float) / np.clip(total, EPS, None)


def winner_market_q(frame: pd.DataFrame) -> np.ndarray:
    red = side_market_q(frame, "red")
    blue = side_market_q(frame, "blue")
    return np.where(frame["target"].to_numpy(int) < 3, red, blue)


def oriented_x(frame: pd.DataFrame, features: list[str], orientation: str) -> tuple[pd.DataFrame, np.ndarray]:
    if orientation == "winner":
        sign = np.where(frame["target"].to_numpy(int) < 3, 1.0, -1.0)
        x = frame[features].replace([np.inf, -np.inf], np.nan).mul(sign, axis=0)
        ml = np.where(sign > 0, frame["model_p_red"].to_numpy(float), 1.0 - frame["model_p_red"].to_numpy(float))
        return x, ml
    if orientation == "red":
        return frame[features].replace([np.inf, -np.inf], np.nan).copy(), frame["model_p_red"].to_numpy(float)
    if orientation == "blue":
        return -frame[features].replace([np.inf, -np.inf], np.nan).copy(), 1.0 - frame["model_p_red"].to_numpy(float)
    raise ValueError(orientation)


def fit_offset(train: pd.DataFrame, score: pd.DataFrame, features: list[str], include_ml: bool, orientation: str):
    xtr, mltr = oriented_x(train, features, "winner")
    xsc, mlsc = oriented_x(score, features, orientation)
    ytr = (train["target"].to_numpy(int) % 3 == 0).astype(int)
    qtr = winner_market_q(train)
    if orientation == "winner":
        qsc = winner_market_q(score)
    else:
        qsc = side_market_q(score, orientation)

    valid = [c for c in features if xtr[c].notna().any()]
    med = xtr[valid].median(numeric_only=True)
    xtr = xtr[valid].fillna(med).fillna(0.0)
    xsc = xsc[valid].fillna(med).fillna(0.0)
    if include_ml:
        xtr = xtr.copy()
        xsc = xsc.copy()
        xtr["v5_ml_p_side"] = mltr
        xsc["v5_ml_p_side"] = mlsc

    dtr = xgb.DMatrix(xtr, label=ytr, base_margin=pure.logit(qtr), feature_names=list(xtr.columns))
    dsc = xgb.DMatrix(xsc, base_margin=pure.logit(qsc), feature_names=list(xsc.columns))
    booster = xgb.train(PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    pred = pure.clip_p(np.asarray(booster.predict(dsc), float))
    gain = booster.get_score(importance_type="gain")
    return pred, gain, len(valid) + int(include_ml), int(len(train))


def exact3_from_q(frame: pd.DataFrame, qred: np.ndarray, qblue: np.ndarray) -> np.ndarray:
    ml_red = frame["model_p_red"].to_numpy(float)
    p = np.column_stack([
        ml_red * qred,
        (1.0 - ml_red) * qblue,
        1.0 - ml_red * qred - (1.0 - ml_red) * qblue,
    ])
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def load_existing_hier() -> pd.DataFrame:
    if not HIER_OOF.exists():
        raise RuntimeError(f"missing existing hierarchical V5 OOF: {HIER_OOF}")
    h = pd.read_csv(HIER_OOF)
    h["fight_id"] = h["fight_id"].astype(str)
    required = {"fight_id", "hier_red_ko", "hier_blue_ko"}
    missing = sorted(required - set(h.columns))
    if missing:
        raise RuntimeError(f"hierarchical OOF missing {missing}")
    if h["fight_id"].duplicated().any():
        raise RuntimeError("duplicate hierarchical V5 OOF fight_id")
    return h[["fight_id", "hier_red_ko", "hier_blue_ko"]]


def metrics_for_exact(y3: np.ndarray, p3: np.ndarray) -> dict:
    return pure.multiclass3_metrics(y3, p3)


def run(v5_market_path: str, v5_feature_path: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df, features, excluded = method._build_rows(True, True)
    df["date"] = pd.to_datetime(df["date"])
    if (df["date"] > "2024-12-31").any():
        raise RuntimeError("2025+ entered market-offset KO development")

    ml_stack, _, v5_ll = pure.build_honest_v5_stack(v5_market_path, v5_feature_path)
    ml_stack["fight_id"] = ml_stack["fight_id"].astype(str)
    df["fight_id"] = df["fight_id"].astype(str)
    df = df.merge(ml_stack[["fight_id", "model_p_red", "market_p_red"]], on="fight_id", how="left")
    hier = load_existing_hier()

    parts = []
    metric_rows = []
    importance_rows = []

    for fold, train_end, val_start, val_end in method.FOLDS:
        train = df[(df["date"] <= train_end) & df["model_p_red"].notna()].copy()
        val = df[(df["date"] >= val_start) & (df["date"] <= val_end) & df["model_p_red"].notna()].copy()
        val = val.merge(hier, on="fight_id", how="left", validate="one_to_one")
        if train.empty or val.empty or val[["hier_red_ko", "hier_blue_ko"]].isna().any(axis=None):
            raise RuntimeError(f"incomplete fold {fold}")

        ycond = (val["target"].to_numpy(int) % 3 == 0).astype(int)
        y3 = pure.exact3_target(val["target"])
        market_q_win = winner_market_q(val)
        market3 = pure.exact3_market(val)
        v5_market3 = exact3_from_q(val, side_market_q(val, "red"), side_market_q(val, "blue"))
        mlred = val["model_p_red"].to_numpy(float)
        hier3 = np.column_stack([
            val["hier_red_ko"].to_numpy(float),
            val["hier_blue_ko"].to_numpy(float),
            1.0 - val["hier_red_ko"].to_numpy(float) - val["hier_blue_ko"].to_numpy(float),
        ])
        hier_q_win = np.where(
            val["target"].to_numpy(int) < 3,
            val["hier_red_ko"].to_numpy(float) / np.clip(mlred, EPS, None),
            val["hier_blue_ko"].to_numpy(float) / np.clip(1.0 - mlred, EPS, None),
        )
        hier_q_win = pure.clip_p(hier_q_win)

        for name, q, p3 in [
            ("method_market", market_q_win, market3),
            ("v5_ml_x_market_cond", market_q_win, v5_market3),
            ("existing_hierarchical_v5", hier_q_win, hier3),
        ]:
            metric_rows.append({"variant": name, "fold": fold, "metric": "conditional_ko", **pure.binary_metrics(ycond, q)})
            metric_rows.append({"variant": name, "fold": fold, "metric": "exact_3class", **metrics_for_exact(y3, p3)})
            metric_rows.append({"variant": name, "fold": fold, "metric": "fight_ko", **pure.binary_metrics((y3 != 2).astype(int), p3[:, 0] + p3[:, 1])})

        out = val[["fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target", "betting_eligible", "cold_start", "red_prior_ufc_fights", "blue_prior_ufc_fights", "model_p_red", "market_p_red", "hier_red_ko", "hier_blue_ko"]].copy()
        out["fold"] = fold
        out["market_q_red_ko_given_win"] = side_market_q(val, "red")
        out["market_q_blue_ko_given_win"] = side_market_q(val, "blue")

        for variant, include_ml in [("no_ml", False), ("with_ml", True)]:
            qwin, gain, fc, train_n = fit_offset(train, val, features, include_ml, "winner")
            qred, _, _, _ = fit_offset(train, val, features, include_ml, "red")
            qblue, _, _, _ = fit_offset(train, val, features, include_ml, "blue")
            p3 = exact3_from_q(val, qred, qblue)
            metric_rows.append({"variant": variant, "fold": fold, "metric": "conditional_ko", "feature_count": fc, "train_n": train_n, **pure.binary_metrics(ycond, qwin)})
            metric_rows.append({"variant": variant, "fold": fold, "metric": "exact_3class", "feature_count": fc, "train_n": train_n, **metrics_for_exact(y3, p3)})
            metric_rows.append({"variant": variant, "fold": fold, "metric": "fight_ko", "feature_count": fc, "train_n": train_n, **pure.binary_metrics((y3 != 2).astype(int), p3[:, 0] + p3[:, 1])})
            out[f"{variant}_q_winner_ko_given_win"] = qwin
            out[f"{variant}_q_red_ko_given_win"] = qred
            out[f"{variant}_q_blue_ko_given_win"] = qblue
            out[f"{variant}_p_red_ko"] = p3[:, 0]
            out[f"{variant}_p_blue_ko"] = p3[:, 1]
            out[f"{variant}_p_no_ko"] = p3[:, 2]
            importance_rows.append({
                "fold": fold,
                "variant": variant,
                "v5_ml_gain": float(gain.get("v5_ml_p_side", 0.0)),
                "v5_ml_used": bool(gain.get("v5_ml_p_side", 0.0) > 0),
            })
        parts.append(out)

    pred = pd.concat(parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
    ycond = (pred["target"].to_numpy(int) % 3 == 0).astype(int)
    y3 = pure.exact3_target(pred["target"])

    eval_df = pred.merge(df[["fight_id", "market_red_ko", "market_red_sub", "market_red_dec", "market_blue_ko", "market_blue_sub", "market_blue_dec"]], on="fight_id", how="left")
    market_q = winner_market_q(eval_df)
    market3 = pure.exact3_market(eval_df)
    v5_market3 = exact3_from_q(eval_df, side_market_q(eval_df, "red"), side_market_q(eval_df, "blue"))
    mlred = pred["model_p_red"].to_numpy(float)
    hier3 = np.column_stack([
        pred["hier_red_ko"].to_numpy(float),
        pred["hier_blue_ko"].to_numpy(float),
        1.0 - pred["hier_red_ko"].to_numpy(float) - pred["hier_blue_ko"].to_numpy(float),
    ])
    hier_q = np.where(
        pred["target"].to_numpy(int) < 3,
        pred["hier_red_ko"].to_numpy(float) / np.clip(mlred, EPS, None),
        pred["hier_blue_ko"].to_numpy(float) / np.clip(1.0 - mlred, EPS, None),
    )
    hier_q = pure.clip_p(hier_q)

    pooled = {}
    for name, q, p3 in [
        ("method_market", market_q, market3),
        ("v5_ml_x_market_cond", market_q, v5_market3),
        ("existing_hierarchical_v5", hier_q, hier3),
    ]:
        pooled[name] = {
            "conditional_ko": pure.binary_metrics(ycond, q),
            "exact_3class": metrics_for_exact(y3, p3),
            "fight_ko": pure.binary_metrics((y3 != 2).astype(int), p3[:, 0] + p3[:, 1]),
        }
    for variant in ["no_ml", "with_ml"]:
        p3 = pred[[f"{variant}_p_red_ko", f"{variant}_p_blue_ko", f"{variant}_p_no_ko"]].to_numpy(float)
        pooled[variant] = {
            "conditional_ko": pure.binary_metrics(ycond, pred[f"{variant}_q_winner_ko_given_win"]),
            "exact_3class": metrics_for_exact(y3, p3),
            "fight_ko": pure.binary_metrics((y3 != 2).astype(int), p3[:, 0] + p3[:, 1]),
        }

    for variant, metrics in pooled.items():
        for metric_name, vals in metrics.items():
            metric_rows.append({"variant": variant, "fold": "pooled_2021_2024", "metric": metric_name, **vals})

    selected = min(["no_ml", "with_ml"], key=lambda z: (pooled[z]["exact_3class"]["log_loss"], pooled[z]["exact_3class"]["brier"]))
    cand = pure.build_edge_candidates(pred)
    rule_df, robust_df = pure.evaluate_rules(cand)

    pred.to_csv(PREDICTIONS, index=False)
    cand.to_csv(CANDIDATES, index=False)
    pd.DataFrame(metric_rows).to_csv(METRICS, index=False)
    rule_df.to_csv(RULES, index=False)
    robust_df.to_csv(ROBUSTNESS, index=False)

    summary = {
        "experiment": "binary_conditional_ko_market_offset_with_frozen_v5_ml_feature_v1",
        "design": "sportsbook conditional KO probability is XGBoost base margin; signed prefight differences learn the residual; optional frozen V5 P(side wins) feature; exact KO = V5 ML * corrected conditional KO",
        "development_window": "chronological 2021-2024 OOF only",
        "reads_2025_plus": False,
        "roi_used_for_model_selection": False,
        "method_market_role": "base margin only, never target-derived postfight information; raw KO price used later only for edge diagnostics",
        "feature_count": len(features),
        "features": features,
        "excluded_leakage_features": excluded,
        "hyperparameters": {**PARAMS, "num_boost_round": ROUNDS},
        "v5_canonical_oof_log_loss": v5_ll,
        "selected_variant": selected,
        "selection_metric": "pooled exact red-KO / blue-KO / no-KO 3-class log loss; Brier tiebreak",
        "pooled_probability_metrics": pooled,
        "delta_with_ml_vs_no_ml": {
            "conditional_log_loss": pooled["with_ml"]["conditional_ko"]["log_loss"] - pooled["no_ml"]["conditional_ko"]["log_loss"],
            "exact_3class_log_loss": pooled["with_ml"]["exact_3class"]["log_loss"] - pooled["no_ml"]["exact_3class"]["log_loss"],
            "fight_ko_log_loss": pooled["with_ml"]["fight_ko"]["log_loss"] - pooled["no_ml"]["fight_ko"]["log_loss"],
        },
        "delta_selected_vs_existing_hierarchical_v5": {
            "conditional_log_loss": pooled[selected]["conditional_ko"]["log_loss"] - pooled["existing_hierarchical_v5"]["conditional_ko"]["log_loss"],
            "exact_3class_log_loss": pooled[selected]["exact_3class"]["log_loss"] - pooled["existing_hierarchical_v5"]["exact_3class"]["log_loss"],
            "fight_ko_log_loss": pooled[selected]["fight_ko"]["log_loss"] - pooled["existing_hierarchical_v5"]["fight_ko"]["log_loss"],
        },
        "v5_ml_feature_use_by_fold": importance_rows,
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
