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
FREEZE = OUT / "xgboost_ko_projected_winner_top50_freeze.json"
V5_HOLDOUT = OUT / "xgboost_method_hierarchical_v5_holdout_2025_2026_predictions.csv"
V5_TEST = OUT / "xgboost_v5_exact_reproduction_test_predictions.csv"
PREDICTIONS = OUT / "xgboost_ko_projected_winner_top50_holdout_2025_predictions.csv"
METRICS = OUT / "xgboost_ko_projected_winner_top50_holdout_2025_metrics.csv"
EDGE_RULES = OUT / "xgboost_ko_projected_winner_top50_holdout_2025_edge_rules.csv"
EDGE_CANDIDATES = OUT / "xgboost_ko_projected_winner_top50_holdout_2025_edge_candidates.csv"
SUMMARY = OUT / "xgboost_ko_projected_winner_top50_holdout_2025_summary.json"

TRAIN_CUTOFF = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")


def fit_predict(train: pd.DataFrame, score: pd.DataFrame, features: list[str]):
    xa, _, margin_a, _, ya, _ = base.project(train, features)
    xb, _, margin_b, _, _, _ = base.project(score, features)
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
    dsc = xgb.DMatrix(
        xb,
        base_margin=pure.logit(margin_b),
        feature_names=list(xb.columns),
    )
    model = xgb.train(base.PARAMS, dtr, num_boost_round=base.ROUNDS, verbose_eval=False)
    p = pure.clip_p(np.asarray(model.predict(dsc), float))
    return p, valid, med


def load_v5_holdout() -> pd.DataFrame:
    h = pd.read_csv(V5_HOLDOUT)
    h["fight_id"] = h["fight_id"].astype(str)
    h["date"] = pd.to_datetime(h["date"])
    h = h[(h["date"] >= TEST_START) & (h["date"] <= TEST_END)].copy()
    if h["fight_id"].duplicated().any():
        raise RuntimeError("duplicate 2025 hierarchical V5 holdout fight_id")
    return h[["fight_id", "v5_model_p_red", "hier_red_ko", "hier_blue_ko"]]


def load_market_ml_2025() -> pd.DataFrame:
    v = pd.read_csv(V5_TEST)
    v["fight_id"] = v["fight_id"].astype(str)
    v["date"] = pd.to_datetime(v["date"])
    v = v[
        (v["date"] >= TEST_START)
        & (v["date"] <= TEST_END)
        & v["market_key"].eq("moneyline")
        & v["bookmaker"].eq("legacy_consensus")
        & v["canonical_side"].astype(str).eq("red")
        & v["result_status"].eq("graded")
    ].copy()
    if v["fight_id"].duplicated().any():
        raise RuntimeError("duplicate 2025 frozen V5 market fight_id")
    return v[["fight_id", "fair_market_p"]].rename(columns={"fair_market_p": "market_p_red"})


def edge_stats(bets: pd.DataFrame) -> dict:
    return base.edge_stats(bets)


def fixed_edge_rules(cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # The locked cold-start rule applies to betting diagnostics. Probability
    # metrics above remain predictive metrics on the full scored holdout cohort.
    eligible = cand[cand["betting_eligible"].astype(bool)].copy()
    for variant in eligible["variant"].drop_duplicates():
        x = eligible[eligible["variant"].eq(variant)].copy()
        rules = base.rule_sets(x)
        for name in ["positive_ev", "positive_ev_agree", "residual_030", "residual_030_agree"]:
            bets = rules[name]
            rows.append({"variant": variant, "rule": name, "scope": "pooled_2025", **edge_stats(bets)})
    return pd.DataFrame(rows)


def run(v5_market_path: str, v5_feature_path: str):
    OUT.mkdir(parents=True, exist_ok=True)

    freeze = json.loads(FREEZE.read_text())
    if not freeze.get("frozen_before_2025_scoring", False):
        raise RuntimeError("TOP50 freeze is not marked frozen before 2025 scoring")
    if freeze.get("selected_candidate") != "TOP50":
        raise RuntimeError(f"unexpected frozen candidate: {freeze.get('selected_candidate')}")
    if freeze.get("roi_used_for_selection") is not False:
        raise RuntimeError("freeze indicates ROI entered model selection")
    features = list(freeze["selected_features"])
    if len(features) != 50:
        raise RuntimeError(f"expected 50 frozen features, got {len(features)}")
    frozen_params = dict(freeze["xgboost_params"])
    expected_params = {**base.PARAMS, "num_boost_round": base.ROUNDS}
    if frozen_params != expected_params:
        raise RuntimeError(f"frozen parameter mismatch: {frozen_params} vs {expected_params}")

    # Build only through 2025-12-31. This scorer does not evaluate calendar 2026.
    original_cutoff = method.DEV_CUTOFF
    method.DEV_CUTOFF = TEST_END
    try:
        df, used_features, excluded = method._build_rows(True, True, forced_features=features)
    finally:
        method.DEV_CUTOFF = original_cutoff
    df["date"] = pd.to_datetime(df["date"])
    if (df["date"] > TEST_END).any():
        raise RuntimeError("2026+ entered 2025 holdout frame")
    if list(used_features) != features:
        raise RuntimeError("frozen feature order changed")

    # Training base margins use the same honest historical/canonical V5 stack
    # used during development. The 2025 score side uses the already-frozen V5
    # holdout probability; 2025 never enters model fitting.
    stack, _, v5_oof_ll = pure.build_honest_v5_stack(v5_market_path, v5_feature_path)
    stack["fight_id"] = stack["fight_id"].astype(str)
    train = df[df["date"] <= TRAIN_CUTOFF].merge(
        stack[["fight_id", "model_p_red", "market_p_red"]],
        on="fight_id",
        how="inner",
        validate="one_to_one",
    )
    if train.empty or (train["date"] > TRAIN_CUTOFF).any():
        raise RuntimeError("invalid frozen training frame")

    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
    test = test.merge(load_v5_holdout(), on="fight_id", how="inner", validate="one_to_one")
    test = test.merge(load_market_ml_2025(), on="fight_id", how="left", validate="one_to_one")
    test = test.rename(columns={"v5_model_p_red": "model_p_red"})
    if test.empty:
        raise RuntimeError("no 2025 TOP50/V5/method-market overlap")
    if test["market_p_red"].isna().any():
        raise RuntimeError("missing 2025 normalized moneyline market probability")

    p, valid, med = fit_predict(train, test, features)
    _, ml, fused, fair, y, side = base.project(test, features)
    hier = np.where(
        side == "red",
        test["hier_red_ko"].to_numpy(float),
        test["hier_blue_ko"].to_numpy(float),
    )

    metric_rows = []
    probs = {
        "METHOD_MARKET_FAIR": fair,
        "FUSED_BASE": fused,
        "EXISTING_HIERARCHICAL_V5": hier,
        "FROZEN_TOP50": p,
    }
    for name, prob in probs.items():
        metric_rows.append({"variant": name, "scope": "holdout_2025", **base.metrics(y, prob)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(METRICS, index=False)

    pred = test[[
        "fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target",
        "betting_eligible", "model_p_red", "market_p_red"
    ]].copy()
    pred["projected_side"] = side
    pred["v5_ml_p_projected_winner"] = ml
    pred["projected_winner_ko"] = y
    pred["market_fair_exact_ko"] = fair
    pred["fused_base_p_ko"] = fused
    pred["hier_projected_p_ko"] = hier
    pred["top50_p_ko"] = p
    pred.to_csv(PREDICTIONS, index=False)

    priced = base.load_raw_projected_prices(pred)
    candidates = pd.concat([
        base.edge_frame(priced, "fused_base", "fused_base_p_ko"),
        base.edge_frame(priced, "existing_hierarchical_v5", "hier_projected_p_ko"),
        base.edge_frame(priced, "frozen_top50", "top50_p_ko"),
    ], ignore_index=True)
    candidates.to_csv(EDGE_CANDIDATES, index=False)
    edge_rules = fixed_edge_rules(candidates)
    edge_rules.to_csv(EDGE_RULES, index=False)

    top = metrics[metrics["variant"].eq("FROZEN_TOP50")].iloc[0]
    hier_m = metrics[metrics["variant"].eq("EXISTING_HIERARCHICAL_V5")].iloc[0]
    fused_m = metrics[metrics["variant"].eq("FUSED_BASE")].iloc[0]
    summary = {
        "experiment": "frozen_projected_winner_exact_ko_top50_holdout_2025_v1",
        "freeze_commit": "f56fbec4a8b6beedaea9c881857f4933fb0665e1",
        "development_window": "chronological 2021-2024 OOF only",
        "training_cutoff": str(TRAIN_CUTOFF.date()),
        "evaluation_period": f"{TEST_START.date()} through {TEST_END.date()}",
        "calendar_2026_scored": False,
        "roi_used_for_model_selection": False,
        "holdout_fights": int(len(pred)),
        "training_rows": int(len(train)),
        "feature_count": int(len(features)),
        "features": features,
        "hyperparameters": expected_params,
        "v5_canonical_oof_log_loss": float(v5_oof_ll),
        "frozen_oof": freeze["selected_oof"],
        "holdout_2025": {
            "frozen_top50": {
                "log_loss": float(top["log_loss"]),
                "brier": float(top["brier"]),
                "auc": None if pd.isna(top["auc"]) else float(top["auc"]),
                "actual_rate": float(top["actual_rate"]),
                "mean_probability": float(top["mean_probability"]),
                "calibration_error": float(top["calibration_error"]),
            },
            "existing_hierarchical_v5": {
                "log_loss": float(hier_m["log_loss"]),
                "brier": float(hier_m["brier"]),
                "auc": None if pd.isna(hier_m["auc"]) else float(hier_m["auc"]),
            },
            "fused_base": {
                "log_loss": float(fused_m["log_loss"]),
                "brier": float(fused_m["brier"]),
                "auc": None if pd.isna(fused_m["auc"]) else float(fused_m["auc"]),
            },
            "delta_top50_vs_existing_v5_log_loss": float(top["log_loss"] - hier_m["log_loss"]),
            "delta_top50_vs_existing_v5_brier": float(top["brier"] - hier_m["brier"]),
            "delta_top50_vs_fused_base_log_loss": float(top["log_loss"] - fused_m["log_loss"]),
        },
        "betting_diagnostics": {
            "selection_role": "diagnostic_only; no thresholds or model choices changed from holdout ROI",
            "cold_start_rule_applied": True,
            "rules_file": str(EDGE_RULES),
        },
        "excluded_leakage_features": excluded,
        "artifacts": [str(PREDICTIONS), str(METRICS), str(EDGE_CANDIDATES), str(EDGE_RULES)],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-market", required=True)
    ap.add_argument("--v5-features", required=True)
    args = ap.parse_args()
    run(args.v5_market, args.v5_features)
