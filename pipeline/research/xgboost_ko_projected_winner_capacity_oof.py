from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.research import xgboost_method_market_offset as method
from pipeline.research import xgboost_ko_conditional_ml_stack_oof as pure
from pipeline.research import xgboost_ko_projected_winner_edge_oof as base

OUT = Path("data/research/prop_mispricing")
SUMMARY = OUT / "xgboost_ko_projected_winner_capacity_oof_summary.json"
METRICS = OUT / "xgboost_ko_projected_winner_capacity_oof_metrics.csv"
PREDICTIONS = OUT / "xgboost_ko_projected_winner_capacity_oof_predictions.csv"
RANKING = OUT / "xgboost_ko_projected_winner_capacity_pre2021_feature_ranking.csv"

CANDIDATE_SIZES = {
    "FULL": None,
    "TOP100": 100,
    "TOP75": 75,
    "TOP50": 50,
    "TOP35": 35,
}


def prep_xy(frame: pd.DataFrame, features: list[str]):
    x, _, margin_p, _, y, _ = base.project(frame, features)
    valid = [c for c in features if x[c].notna().any()]
    med = x[valid].median(numeric_only=True)
    x = x[valid].fillna(med).fillna(0.0)
    return x, margin_p, y, valid


def rank_pre2021(df: pd.DataFrame, features: list[str]):
    train = df[(df["date"] <= "2020-12-31") & df["model_p_red"].notna()].copy()
    if train.empty:
        raise RuntimeError("no pre-2021 projected-winner KO training rows")
    x, margin_p, y, valid = prep_xy(train, features)
    d = xgb.DMatrix(
        x,
        label=y,
        base_margin=pure.logit(margin_p),
        feature_names=list(x.columns),
    )
    model = xgb.train(base.PARAMS, d, num_boost_round=base.ROUNDS, verbose_eval=False)
    gain = model.get_score(importance_type="gain")
    weight = model.get_score(importance_type="weight")
    rows = []
    for c in valid:
        rows.append({
            "feature": c,
            "gain": float(gain.get(c, 0.0)),
            "weight": float(weight.get(c, 0.0)),
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["gain", "weight", "feature"], ascending=[False, False, True]
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking, int(len(train))


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, features: list[str]):
    xa, _, margin_a, _, ya, _ = base.project(train, features)
    xb, _, margin_b, _, _, _ = base.project(val, features)
    valid = [c for c in features if xa[c].notna().any()]
    med = xa[valid].median(numeric_only=True)
    xa = xa[valid].fillna(med).fillna(0.0)
    xb = xb[valid].fillna(med).fillna(0.0)
    dtr = xgb.DMatrix(
        xa,
        label=ya,
        base_margin=pure.logit(margin_a),
        feature_names=list(xa.columns),
    )
    dva = xgb.DMatrix(
        xb,
        base_margin=pure.logit(margin_b),
        feature_names=list(xb.columns),
    )
    model = xgb.train(base.PARAMS, dtr, num_boost_round=base.ROUNDS, verbose_eval=False)
    p = pure.clip_p(np.asarray(model.predict(dva), float))
    return p, len(valid)


def run(v5_market_path: str, v5_feature_path: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df, features, excluded = method._build_rows(True, True)
    df["date"] = pd.to_datetime(df["date"])
    if (df["date"] > "2024-12-31").any():
        raise RuntimeError("2025+ entered projected-winner KO capacity development")

    ml_stack, _, v5_ll = pure.build_honest_v5_stack(v5_market_path, v5_feature_path)
    ml_stack["fight_id"] = ml_stack["fight_id"].astype(str)
    df["fight_id"] = df["fight_id"].astype(str)
    df = df.merge(
        ml_stack[["fight_id", "model_p_red", "market_p_red"]],
        on="fight_id",
        how="left",
        validate="one_to_one",
    )
    h = base.load_hier()

    ranking, ranking_train_n = rank_pre2021(df, features)
    ranking.to_csv(RANKING, index=False)
    ranked_features = ranking["feature"].tolist()
    candidates = {}
    for name, n in CANDIDATE_SIZES.items():
        candidates[name] = list(features) if n is None else ranked_features[: min(n, len(ranked_features))]

    metric_rows = []
    candidate_predictions: dict[str, pd.DataFrame] = {}
    fold_audit = []

    for candidate, cols in candidates.items():
        parts = []
        for fold, train_end, val_start, val_end in method.FOLDS:
            train = df[(df["date"] <= train_end) & df["model_p_red"].notna()].copy()
            val = df[
                (df["date"] >= val_start)
                & (df["date"] <= val_end)
                & df["model_p_red"].notna()
            ].copy()
            val = val.merge(h, on="fight_id", how="left", validate="one_to_one")
            if train.empty or val.empty or val[["hier_red_ko", "hier_blue_ko"]].isna().any(axis=None):
                raise RuntimeError(f"incomplete fold {fold} for {candidate}")

            _, _, fused, fair, y, side = base.project(val, features)
            hier = np.where(
                side == "red",
                val["hier_red_ko"].to_numpy(float),
                val["hier_blue_ko"].to_numpy(float),
            )
            p, fc = fit_predict(train, val, cols)
            metric_rows.append({
                "candidate": candidate,
                "fold": fold,
                "feature_count": fc,
                "train_n": int(len(train)),
                "validation_n": int(len(val)),
                **base.metrics(y, p),
            })
            if candidate == "FULL":
                for baseline, prob in [
                    ("METHOD_MARKET_FAIR", fair),
                    ("FUSED_BASE", fused),
                    ("EXISTING_HIERARCHICAL_V5", hier),
                ]:
                    metric_rows.append({
                        "candidate": baseline,
                        "fold": fold,
                        "feature_count": 0,
                        "train_n": int(len(train)),
                        "validation_n": int(len(val)),
                        **base.metrics(y, prob),
                    })
                fold_audit.append({
                    "fold": fold,
                    "train_through": train_end,
                    "train_n": int(len(train)),
                    "validation_n": int(len(val)),
                })

            out = val[[
                "fight_id", "date", "event_name", "red_fighter", "blue_fighter",
                "target", "model_p_red", "market_p_red"
            ]].copy()
            out["fold"] = fold
            out["projected_side"] = side
            out["projected_winner_ko"] = y
            out["candidate_p_ko"] = p
            parts.append(out)

        pred = pd.concat(parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
        candidate_predictions[candidate] = pred
        pooled = base.metrics(pred["projected_winner_ko"].to_numpy(int), pred["candidate_p_ko"].to_numpy(float))
        metric_rows.append({
            "candidate": candidate,
            "fold": "pooled_2021_2024",
            "feature_count": len(cols),
            "train_n": np.nan,
            "validation_n": int(len(pred)),
            **pooled,
        })

    metrics = pd.DataFrame(metric_rows)

    # Add pooled baselines from the selected cohort once, using FULL prediction IDs/order.
    full_pred = candidate_predictions["FULL"]
    full_join = full_pred.merge(h, on="fight_id", how="left", validate="one_to_one")
    rows = df[df["fight_id"].isin(full_pred["fight_id"])].copy()
    rows = full_pred[["fight_id"]].merge(rows, on="fight_id", how="left", validate="one_to_one")
    _, _, fused, fair, y, side = base.project(rows, features)
    hj = full_pred[["fight_id"]].merge(h, on="fight_id", how="left", validate="one_to_one")
    hier = np.where(side == "red", hj["hier_red_ko"].to_numpy(float), hj["hier_blue_ko"].to_numpy(float))
    for baseline, prob in [
        ("METHOD_MARKET_FAIR", fair),
        ("FUSED_BASE", fused),
        ("EXISTING_HIERARCHICAL_V5", hier),
    ]:
        metrics = pd.concat([metrics, pd.DataFrame([{
            "candidate": baseline,
            "fold": "pooled_2021_2024",
            "feature_count": 0,
            "train_n": np.nan,
            "validation_n": int(len(y)),
            **base.metrics(y, prob),
        }])], ignore_index=True)

    candidate_pool = metrics[
        metrics["fold"].eq("pooled_2021_2024")
        & metrics["candidate"].isin(candidates)
    ].copy()
    selected_row = candidate_pool.sort_values(
        ["log_loss", "brier", "feature_count", "candidate"]
    ).iloc[0]
    selected = str(selected_row["candidate"])
    selected_features = candidates[selected]
    selected_pred = candidate_predictions[selected].copy()
    selected_pred["selected_candidate"] = selected
    selected_pred.to_csv(PREDICTIONS, index=False)
    metrics.to_csv(METRICS, index=False)

    baseline_row = metrics[
        metrics["candidate"].eq("EXISTING_HIERARCHICAL_V5")
        & metrics["fold"].eq("pooled_2021_2024")
    ].iloc[0]
    fused_row = metrics[
        metrics["candidate"].eq("FUSED_BASE")
        & metrics["fold"].eq("pooled_2021_2024")
    ].iloc[0]

    summary = {
        "experiment": "projected_winner_exact_ko_market_offset_capacity_v2",
        "design": "direct projected-winner exact-KO binary residual on frozen V5 P(win) * sportsbook conditional KO; fixed depth-1 XGBoost; only feature capacity varies",
        "development_window": "chronological 2021-2024 OOF only",
        "reads_2025_plus": False,
        "roi_used_for_model_selection": False,
        "feature_ranking_cutoff": "2020-12-31",
        "feature_ranking_train_n": ranking_train_n,
        "feature_ranking_target": "projected-winner exact KO using same fused base margin",
        "candidate_family": {k: len(v) for k, v in candidates.items()},
        "hyperparameters": {**base.PARAMS, "num_boost_round": base.ROUNDS},
        "v5_canonical_oof_log_loss": v5_ll,
        "fold_audit": fold_audit,
        "selected_candidate": selected,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "selection_metric": "pooled 2021-2024 projected-winner exact-KO binary log loss; Brier then smaller feature count tiebreak",
        "selected_oof": {
            "log_loss": float(selected_row["log_loss"]),
            "brier": float(selected_row["brier"]),
            "auc": None if pd.isna(selected_row["auc"]) else float(selected_row["auc"]),
            "mean_probability": float(selected_row["mean_probability"]),
            "actual_rate": float(selected_row["actual_rate"]),
            "calibration_error": float(selected_row["calibration_error"]),
        },
        "existing_hierarchical_v5_oof": {
            "log_loss": float(baseline_row["log_loss"]),
            "brier": float(baseline_row["brier"]),
            "auc": None if pd.isna(baseline_row["auc"]) else float(baseline_row["auc"]),
        },
        "fused_base_oof": {
            "log_loss": float(fused_row["log_loss"]),
            "brier": float(fused_row["brier"]),
            "auc": None if pd.isna(fused_row["auc"]) else float(fused_row["auc"]),
        },
        "delta_selected_vs_existing_v5_log_loss": float(selected_row["log_loss"] - baseline_row["log_loss"]),
        "delta_selected_vs_existing_v5_brier": float(selected_row["brier"] - baseline_row["brier"]),
        "delta_selected_vs_fused_base_log_loss": float(selected_row["log_loss"] - fused_row["log_loss"]),
        "excluded_leakage_features": excluded,
        "artifacts": [str(PREDICTIONS), str(METRICS), str(RANKING)],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-market", required=True)
    ap.add_argument("--v5-features", required=True)
    args = ap.parse_args()
    run(args.v5_market, args.v5_features)
