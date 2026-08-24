"""Matched-cohort validation of recent dominance beyond FSR V3.

Uses exactly the same complete-case train/test fights for FSR-only and
FSR+dominance models so any incremental signal cannot be caused by cohort
selection. Research-only; no source data, FSR state, or simulator mechanics are
modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_dominance_residual_audit import (
    MASTER_PATH,
    MARKET_PATH,
    ROUND_PATH,
    add_prefight_dominance,
    build_fight_dominance,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import (
    build_matchups,
    build_two_way_market,
    choose_trait_columns,
    safe_logit,
)


def market_metrics(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str) -> dict:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    ytr = safe_logit(train["market_favorite_fair_p"])
    yte = safe_logit(test["market_favorite_fair_p"])
    model.fit(train[features], ytr)
    pred = model.predict(test[features])
    prob = 1.0 / (1.0 + np.exp(-pred))
    return {
        "model": label,
        "feature_count": len(features),
        "train_n": len(train),
        "test_n": len(test),
        "test_r2_logit": r2_score(yte, pred),
        "test_rmse_logit": mean_squared_error(yte, pred) ** 0.5,
        "mean_abs_residual_pp": float(np.mean(np.abs(100.0 * (test["market_favorite_fair_p"].to_numpy() - prob)))),
        "corr_pred_market_p": float(np.corrcoef(prob, test["market_favorite_fair_p"].to_numpy())[0, 1]),
    }


def winner_metrics(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str) -> dict:
    model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=0.25, max_iter=2000))])
    ytr = train["favorite_won"].astype(int)
    yte = test["favorite_won"].astype(int).to_numpy()
    model.fit(train[features], ytr)
    prob = model.predict_proba(test[features])[:, 1]
    return {
        "model": label,
        "feature_count": len(features),
        "train_n": len(train),
        "test_n": len(test),
        "auc": roc_auc_score(yte, prob),
        "brier": brier_score_loss(yte, prob),
        "logloss": log_loss(yte, prob),
        "accuracy": float(np.mean((prob >= 0.5).astype(int) == yte)),
        "mean_pred_favorite_p": float(np.mean(prob)),
        "actual_favorite_win_rate": float(np.mean(yte)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rounds = pd.read_parquet(ROUND_PATH)
    master = pd.read_parquet(MASTER_PATH)
    fsr = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    market = build_two_way_market(MARKET_PATH)
    traits = choose_trait_columns(fsr)
    matchups = build_matchups(market, fsr, master, traits)
    dominance, usable_stats, resolved = build_fight_dominance(rounds, master)
    frame = add_prefight_dominance(matchups, dominance).sort_values(["fight_date", "fight_id"]).reset_index(drop=True)

    fsr_features = [f"delta__{c}" for c in traits]
    dom_features = ["delta_dom_last1", "delta_dom_last3", "delta_dom_ewm"]
    required = fsr_features + dom_features + ["market_favorite_fair_p", "favorite_won", "fight_date"]
    complete = frame.dropna(subset=required).copy().sort_values(["fight_date", "fight_id"]).reset_index(drop=True)
    if len(complete) < 500:
        raise RuntimeError(f"too few complete-case fights: {len(complete)}")

    cut = int(len(complete) * 0.70)
    train = complete.iloc[:cut].copy()
    test = complete.iloc[cut:].copy()
    cut_date = test["fight_date"].min()

    market_rows = [market_metrics(train, test, fsr_features, "fsr_only_matched")]
    for d in dom_features:
        market_rows.append(market_metrics(train, test, fsr_features + [d], f"fsr_plus_{d}_matched"))
    market_rows.append(market_metrics(train, test, fsr_features + dom_features, "fsr_plus_all_dominance_matched"))
    market_df = pd.DataFrame(market_rows)
    base_r2 = float(market_df.loc[market_df["model"] == "fsr_only_matched", "test_r2_logit"].iloc[0])
    base_rmse = float(market_df.loc[market_df["model"] == "fsr_only_matched", "test_rmse_logit"].iloc[0])
    market_df["delta_r2_vs_fsr"] = market_df["test_r2_logit"] - base_r2
    market_df["delta_rmse_vs_fsr"] = market_df["test_rmse_logit"] - base_rmse
    market_df = market_df.sort_values("test_rmse_logit")

    winner_rows = [
        winner_metrics(train, test, fsr_features, "fsr_only_matched"),
        winner_metrics(train, test, fsr_features + dom_features, "fsr_plus_all_dominance_matched"),
    ]
    winner_df = pd.DataFrame(winner_rows)
    base = winner_df[winner_df["model"] == "fsr_only_matched"].iloc[0]
    winner_df["delta_auc_vs_fsr"] = winner_df["auc"] - float(base["auc"])
    winner_df["delta_brier_vs_fsr"] = winner_df["brier"] - float(base["brier"])
    winner_df["delta_logloss_vs_fsr"] = winner_df["logloss"] - float(base["logloss"])

    # Also test dominance alone against the actual outcome to make its direction easy to inspect.
    simple_rows = []
    for d in dom_features:
        m = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=2000))])
        m.fit(train[[d]], train["favorite_won"].astype(int))
        p = m.predict_proba(test[[d]])[:, 1]
        y = test["favorite_won"].astype(int).to_numpy()
        simple_rows.append({"feature": d, "auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "logloss": log_loss(y, p)})
    simple_df = pd.DataFrame(simple_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    market_df.to_csv(args.out_dir / "matched_market_models.csv", index=False)
    winner_df.to_csv(args.out_dir / "matched_winner_models.csv", index=False)
    simple_df.to_csv(args.out_dir / "dominance_only_winner_models.csv", index=False)
    complete[["fight_id", "fight_date", "market_favorite_fair_p", "favorite_won"] + dom_features].to_csv(args.out_dir / "matched_cohort.csv", index=False)
    pd.DataFrame([{
        "joined_fights": len(frame),
        "matched_complete_fights": len(complete),
        "train_n": len(train),
        "test_n": len(test),
        "cut_date": str(pd.Timestamp(cut_date).date()),
        "usable_stats": ",".join(usable_stats),
        "resolved_columns": str(resolved),
    }]).to_csv(args.out_dir / "metadata.csv", index=False)

    print("FSR DOMINANCE MATCHED-COHORT AUDIT")
    print(f"joined={len(frame)} | complete={len(complete)} | train={len(train)} | test={len(test)} | cut={pd.Timestamp(cut_date).date()}")
    print(f"dominance stats={usable_stats}")
    print("\nMATCHED MARKET MODELS")
    print(market_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nMATCHED ACTUAL-WINNER MODELS")
    print(winner_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nDOMINANCE-ONLY ACTUAL-WINNER MODELS")
    print(simple_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
