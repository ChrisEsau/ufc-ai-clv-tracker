from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research import xgboost_method_market_offset as method
from pipeline.research import xgboost_ko_conditional_ml_stack_oof as pure
from pipeline.research import xgboost_ko_projected_winner_edge_oof as base

OUT = Path("data/research/prop_mispricing")
FREEZE = OUT / "xgboost_ko_projected_winner_top50_freeze.json"
HIER_HOLDOUT = OUT / "xgboost_method_hierarchical_v5_holdout_2025_2026_predictions.csv"
PREDICTIONS = OUT / "xgboost_ko_projected_winner_top50_2025_holdout_predictions.csv"
METRICS = OUT / "xgboost_ko_projected_winner_top50_2025_holdout_metrics.csv"
LEDGER = OUT / "xgboost_ko_projected_winner_top50_2025_holdout_edge_ledger.csv"
SUMMARY = OUT / "xgboost_ko_projected_winner_top50_2025_holdout_summary.json"

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2025-12-31")
TRAIN_END = pd.Timestamp("2024-12-31")
EPS = 1e-9


def _load_freeze() -> dict:
    if not FREEZE.exists():
        raise RuntimeError(f"missing freeze file: {FREEZE}")
    f = json.loads(FREEZE.read_text())
    if not f.get("frozen_before_2025_scoring"):
        raise RuntimeError("candidate was not frozen before holdout scoring")
    if f.get("selected_candidate") != "TOP50":
        raise RuntimeError(f"unexpected frozen candidate: {f.get('selected_candidate')}")
    if f.get("roi_used_for_selection"):
        raise RuntimeError("freeze says ROI was used for selection")
    return f


def _build_through_2025(features: list[str]) -> pd.DataFrame:
    original_cutoff = method.DEV_CUTOFF
    try:
        method.DEV_CUTOFF = END
        df, _, _ = method._build_rows(True, True, forced_features=features)
    finally:
        method.DEV_CUTOFF = original_cutoff
    df["date"] = pd.to_datetime(df["date"])
    if (df["date"] > END).any():
        raise RuntimeError("2026+ entered TOP50 holdout frame")
    return df


def _load_2025_hier() -> pd.DataFrame:
    h = pd.read_csv(HIER_HOLDOUT)
    h["fight_id"] = h["fight_id"].astype(str)
    h["date"] = pd.to_datetime(h["date"])
    h = h[(h["date"] >= START) & (h["date"] <= END)].copy()
    if h.empty:
        raise RuntimeError("no 2025 frozen hierarchical holdout rows")
    if h["fight_id"].duplicated().any():
        raise RuntimeError("duplicate 2025 hierarchical holdout fight_id")
    if (h["date"].dt.year != 2025).any():
        raise RuntimeError("non-2025 row survived hierarchical holdout filter")
    return h[["fight_id", "v5_model_p_red", "hier_red_ko", "hier_blue_ko"]]


def _load_raw_2025_ko_prices() -> pd.DataFrame:
    m = pd.read_parquet(
        method.MARKET_PATH,
        filters=[("date", ">=", START), ("date", "<=", END)],
    ).copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m = m[
        (m["bookmaker"] == "legacy_consensus")
        & (m["market_key"] == "win_by_ko_tko_dq")
        & m["outcome_side"].astype(str).isin(["red", "blue"])
        & (m["result_status"] == "graded")
        & m["won"].notna()
    ].copy()
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m["profit_per_100"] = pd.to_numeric(m["profit_per_100"], errors="coerce")
    m = m.dropna(subset=["implied_probability", "profit_per_100"])
    counts = m.groupby(["fight_id", "outcome_side"]).size()
    good = set((str(a), str(b)) for a, b in counts[counts.eq(1)].index)
    m = m[m.apply(lambda r: (str(r["fight_id"]), str(r["outcome_side"])) in good, axis=1)].copy()
    return m[["fight_id", "outcome_side", "implied_probability", "profit_per_100"]]


def _edge_rows(pred: pd.DataFrame, variant: str, pcol: str, raw: pd.DataFrame) -> pd.DataFrame:
    x = pred.copy()
    x["variant"] = variant
    x["model_p_ko"] = x[pcol].to_numpy(float)
    x = x.merge(
        raw,
        left_on=["fight_id", "projected_side"],
        right_on=["fight_id", "outcome_side"],
        how="inner",
        validate="one_to_one",
    )
    x["market_raw_p"] = x["implied_probability"].astype(float)
    x["win_profit_units"] = x["profit_per_100"].astype(float) / 100.0
    x["decimal_odds"] = 1.0 + x["win_profit_units"]
    x["american_odds"] = pure.american_from_profit_per_100(x["profit_per_100"])
    x["ev"] = x["model_p_ko"] * x["decimal_odds"] - 1.0
    x["profit_units"] = np.where(
        x["projected_winner_ko"].eq(1), x["win_profit_units"], -1.0
    )
    x["bet"] = x["ev"] > 0
    return x


def _bet_stats(x: pd.DataFrame) -> dict:
    bets = x[x["bet"]].copy()
    if bets.empty:
        return {
            "bets": 0,
            "wins": 0,
            "profit_units": 0.0,
            "roi": None,
            "hit_rate": None,
            "mean_american_odds": None,
            "median_american_odds": None,
        }
    profit = float(bets["profit_units"].sum())
    return {
        "bets": int(len(bets)),
        "wins": int(bets["projected_winner_ko"].sum()),
        "profit_units": profit,
        "roi": profit / len(bets),
        "hit_rate": float(bets["projected_winner_ko"].mean()),
        "mean_american_odds": float(bets["american_odds"].mean()),
        "median_american_odds": float(bets["american_odds"].median()),
    }


def run(v5_market_path: str, v5_feature_path: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = _load_freeze()
    features = list(freeze["selected_features"])
    if len(features) != 50:
        raise RuntimeError(f"freeze feature count mismatch: {len(features)}")

    df = _build_through_2025(features)
    df["fight_id"] = df["fight_id"].astype(str)
    train = df[df["date"] <= TRAIN_END].copy()
    test = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    if train.empty or test.empty:
        raise RuntimeError("empty TOP50 train or 2025 holdout frame")

    # Honest historical V5 probabilities are used only to define the projected-winner
    # training target/base margin; no 2025 target is used in fitting.
    stack, _, canonical_v5_ll = pure.build_honest_v5_stack(v5_market_path, v5_feature_path)
    stack["fight_id"] = stack["fight_id"].astype(str)
    train = train.merge(
        stack[["fight_id", "model_p_red", "market_p_red"]],
        on="fight_id",
        how="inner",
        validate="one_to_one",
    )
    if train.empty or train["date"].max() > TRAIN_END:
        raise RuntimeError("invalid frozen training frame")

    h = _load_2025_hier()
    test = test.merge(h, on="fight_id", how="inner", validate="one_to_one")
    test = test.rename(columns={"v5_model_p_red": "model_p_red"})
    if test.empty or (test["date"].dt.year != 2025).any():
        raise RuntimeError("invalid 2025 test frame")

    _, _, fused, fair, y, side = base.project(test, features)
    hier = np.where(
        side == "red",
        test["hier_red_ko"].to_numpy(float),
        test["hier_blue_ko"].to_numpy(float),
    )
    p, gain, feature_count = base.fit(train, test, features, False)
    if feature_count != 50:
        raise RuntimeError(f"frozen TOP50 active feature count changed: {feature_count}")

    metric_map = {
        "METHOD_MARKET_FAIR": base.metrics(y, fair),
        "FUSED_BASE": base.metrics(y, fused),
        "EXISTING_HIERARCHICAL_V5": base.metrics(y, hier),
        "FROZEN_TOP50": base.metrics(y, p),
    }
    metric_rows = [{"variant": k, **v} for k, v in metric_map.items()]
    pd.DataFrame(metric_rows).to_csv(METRICS, index=False)

    pred = test[[
        "fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target",
        "model_p_red", "market_red_ko", "market_red_sub", "market_red_dec",
        "market_blue_ko", "market_blue_sub", "market_blue_dec"
    ]].copy()
    pred["projected_side"] = side
    pred["projected_winner_ko"] = y
    pred["method_market_fair_exact_ko"] = fair
    pred["fused_base_p_ko"] = fused
    pred["existing_hierarchical_v5_p_ko"] = hier
    pred["frozen_top50_p_ko"] = p
    pred.to_csv(PREDICTIONS, index=False)

    raw = _load_raw_2025_ko_prices()
    top50_edge = _edge_rows(pred, "FROZEN_TOP50", "frozen_top50_p_ko", raw)
    v5_edge = _edge_rows(pred, "EXISTING_HIERARCHICAL_V5", "existing_hierarchical_v5_p_ko", raw)
    fused_edge = _edge_rows(pred, "FUSED_BASE", "fused_base_p_ko", raw)
    ledger = pd.concat([top50_edge, v5_edge, fused_edge], ignore_index=True)
    ledger = ledger.sort_values(["date", "fight_id", "variant"]).reset_index(drop=True)
    ledger.to_csv(LEDGER, index=False)

    edge = {
        "FROZEN_TOP50_positive_ev": _bet_stats(top50_edge),
        "EXISTING_HIERARCHICAL_V5_positive_ev": _bet_stats(v5_edge),
        "FUSED_BASE_positive_ev": _bet_stats(fused_edge),
    }

    top50 = metric_map["FROZEN_TOP50"]
    v5m = metric_map["EXISTING_HIERARCHICAL_V5"]
    summary = {
        "experiment": "frozen_projected_winner_exact_ko_top50_2025_holdout_v1",
        "candidate_freeze_file": str(FREEZE),
        "candidate_frozen_before_holdout": True,
        "training_cutoff": "2024-12-31",
        "evaluation_period": "2025-01-01 through 2025-12-31",
        "2026_evaluated": False,
        "architecture_changed_after_2025": False,
        "roi_used_for_selection": False,
        "holdout_roi_diagnostic_only": True,
        "selected_feature_count": 50,
        "training_rows": int(len(train)),
        "holdout_fights": int(len(test)),
        "canonical_v5_oof_log_loss_reproduced": float(canonical_v5_ll),
        "oof_selection_score_from_freeze": freeze["selected_oof"],
        "holdout_probability_metrics": metric_map,
        "delta_top50_vs_existing_v5_log_loss": float(top50["log_loss"] - v5m["log_loss"]),
        "delta_top50_vs_existing_v5_brier": float(top50["brier"] - v5m["brier"]),
        "fixed_positive_ev_diagnostic": edge,
        "top50_feature_gain_nonzero": int(sum(float(v) > 0 for v in gain.values())),
        "post_holdout_policy": "do not modify architecture, features, hyperparameters, or bet threshold based on this 2025 result",
        "artifacts": [str(PREDICTIONS), str(METRICS), str(LEDGER)],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-market", required=True)
    ap.add_argument("--v5-features", required=True)
    args = ap.parse_args()
    run(args.v5_market, args.v5_features)
