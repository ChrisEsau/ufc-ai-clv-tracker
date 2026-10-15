"""Audit bantamweight Event Clock MC probabilities versus historical market odds and realized outcomes.

Uses already-generated fight summaries for the selected i10_b0 arm. No simulation or
FSR refit is performed here. Historical odds come from data/market/historical_market_outcomes.parquet.

This is retrospective research. The same cohort contributed to mechanics selection, so
realized ROI is descriptive/in-sample and not promotion-grade forward evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ARM = "i10_b0"
EDGE_THRESHOLDS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15]
EV_THRESHOLDS = [0.0, 0.025, 0.05, 0.10, 0.20]


def payout_per_unit(american: float) -> float:
    return american / 100.0 if american > 0 else 100.0 / abs(american)


def model_probability(row: pd.Series) -> float:
    side = str(row["outcome_side"]).lower()
    market = row["market_key"]
    prefix = "red" if side == "red" else "blue"
    if market == "moneyline":
        return float(row[f"p_{prefix}_win"])
    if market == "win_by_decision":
        return float(row[f"p_{prefix}_dec"])
    if market == "win_by_submission":
        return float(row[f"p_{prefix}_sub"])
    if market == "win_by_ko_tko_dq":
        return float(row[f"p_{prefix}_ko_tko"])
    raise KeyError(market)


def add_market_fair_probs(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    out["market_family"] = np.where(out["market_key"].eq("moneyline"), "moneyline", "exact_method")
    out["market_fair_probability"] = np.nan
    # ML: remove two-way vig within fight.
    ml = out["market_key"].eq("moneyline")
    sums = out.loc[ml].groupby("fight_id")["implied_probability"].transform("sum")
    out.loc[ml, "market_fair_probability"] = out.loc[ml, "implied_probability"] / sums
    # Exact-method: normalize all available exact outcomes within fight. This is diagnostic only.
    prop = ~ml
    sums = out.loc[prop].groupby("fight_id")["implied_probability"].transform("sum")
    out.loc[prop, "market_fair_probability"] = out.loc[prop, "implied_probability"] / sums
    return out


def strategy_summary(rows: pd.DataFrame, selector: str, thresholds: list[float], top_only: bool) -> pd.DataFrame:
    records = []
    for family in ["moneyline", "exact_method", "all"]:
        base = rows if family == "all" else rows[rows["market_family"].eq(family)]
        for t in thresholds:
            chosen = base[base[selector] >= t].copy()
            if top_only and not chosen.empty:
                chosen = chosen.sort_values(selector, ascending=False).drop_duplicates(["fight_id", "market_family"])
            if chosen.empty:
                records.append({"selector": selector, "threshold": t, "top_only": top_only, "market_family": family,
                                "bets": 0, "wins": 0, "hit_rate": np.nan, "units": 0.0, "roi": np.nan,
                                "avg_model_p": np.nan, "avg_market_implied": np.nan, "avg_market_fair": np.nan,
                                "avg_edge_raw": np.nan, "avg_edge_fair": np.nan, "avg_model_ev": np.nan})
                continue
            records.append({"selector": selector, "threshold": t, "top_only": top_only, "market_family": family,
                            "bets": len(chosen), "wins": int(chosen["won"].sum()), "hit_rate": float(chosen["won"].mean()),
                            "units": float(chosen["realized_units"].sum()), "roi": float(chosen["realized_units"].mean()),
                            "avg_model_p": float(chosen["model_probability"].mean()),
                            "avg_market_implied": float(chosen["implied_probability"].mean()),
                            "avg_market_fair": float(chosen["market_fair_probability"].mean()),
                            "avg_edge_raw": float(chosen["edge_raw"].mean()),
                            "avg_edge_fair": float(chosen["edge_fair"].mean()),
                            "avg_model_ev": float(chosen["model_ev"].mean())})
    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-path", type=Path, required=True)
    ap.add_argument("--market-path", type=Path, default=Path("data/market/historical_market_outcomes.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/bantamweight_market_edge_audit"))
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions_path)
    pred = pred[pred["arm"].eq(ARM)].copy()
    if pred["fight_id"].duplicated().any():
        raise ValueError("duplicate fight_id in selected prediction arm")

    market = pd.read_parquet(args.market_path)
    market = market[market["fight_id"].isin(pred["fight_id"])].copy()
    market = market[market["market_key"].isin(["moneyline", "win_by_decision", "win_by_submission", "win_by_ko_tko_dq"])].copy()
    market = market[market["result_status"].eq("graded") & market["american_odds"].notna()].copy()
    market = market.drop_duplicates(["fight_id", "market_key", "outcome_side", "bookmaker"], keep="last")

    keep = ["fight_id","event_name","event_date","red","blue","actual_winner","actual_method",
            "p_red_win","p_red_dec","p_red_ko_tko","p_red_sub","p_blue_win","p_blue_dec","p_blue_ko_tko","p_blue_sub",
            "red_prior_ufc_fights","blue_prior_ufc_fights","fight_evidence_bucket"]
    rows = market.merge(pred[keep], on="fight_id", how="inner", suffixes=("_market", ""))
    rows = add_market_fair_probs(rows)
    rows["model_probability"] = rows.apply(model_probability, axis=1)
    rows["edge_raw"] = rows["model_probability"] - rows["implied_probability"].astype(float)
    rows["edge_fair"] = rows["model_probability"] - rows["market_fair_probability"].astype(float)
    rows["payout_per_unit"] = rows["american_odds"].astype(float).map(payout_per_unit)
    rows["model_ev"] = rows["model_probability"] * rows["payout_per_unit"] - (1.0 - rows["model_probability"])
    rows["won"] = rows["won"].astype(bool)
    rows["realized_units"] = np.where(rows["won"], rows["payout_per_unit"], -1.0)

    # Market-vs-MC moneyline discrimination/calibration on fights with both sides priced.
    ml = rows[rows["market_key"].eq("moneyline")].copy()
    counts = ml.groupby("fight_id")["outcome_side"].nunique()
    complete_ids = counts[counts.eq(2)].index
    mlc = ml[ml["fight_id"].isin(complete_ids)].copy()
    redm = mlc[mlc["outcome_side"].eq("red")].copy()
    y_red = redm["actual_winner"].eq("red").astype(int).to_numpy()
    p_mc = redm["p_red_win"].astype(float).to_numpy()
    p_mkt = redm["market_fair_probability"].astype(float).to_numpy()
    comparison = pd.DataFrame([{
        "priced_fights": len(redm),
        "mc_accuracy": float(((p_mc >= .5).astype(int) == y_red).mean()) if len(redm) else np.nan,
        "market_accuracy": float(((p_mkt >= .5).astype(int) == y_red).mean()) if len(redm) else np.nan,
        "mc_auc": float(roc_auc_score(y_red, p_mc)) if len(np.unique(y_red)) == 2 else np.nan,
        "market_auc": float(roc_auc_score(y_red, p_mkt)) if len(np.unique(y_red)) == 2 else np.nan,
        "mc_brier": float(brier_score_loss(y_red, p_mc)) if len(redm) else np.nan,
        "market_brier": float(brier_score_loss(y_red, p_mkt)) if len(redm) else np.nan,
        "mc_logloss": float(log_loss(y_red, np.clip(p_mc,1e-9,1-1e-9))) if len(redm) else np.nan,
        "market_logloss": float(log_loss(y_red, np.clip(p_mkt,1e-9,1-1e-9))) if len(redm) else np.nan,
    }])

    strategy = pd.concat([
        strategy_summary(rows, "edge_raw", EDGE_THRESHOLDS, False),
        strategy_summary(rows, "edge_raw", EDGE_THRESHOLDS, True),
        strategy_summary(rows, "model_ev", EV_THRESHOLDS, False),
        strategy_summary(rows, "model_ev", EV_THRESHOLDS, True),
    ], ignore_index=True)

    # Market coverage and per-market descriptive ROI at positive model EV.
    coverage = rows.groupby("market_key", as_index=False).agg(
        outcomes=("fight_id","size"), fights=("fight_id","nunique"), avg_odds=("american_odds","mean"),
        avg_model_p=("model_probability","mean"), avg_implied_p=("implied_probability","mean"),
        avg_fair_p=("market_fair_probability","mean"),
    )
    positive = rows[rows["model_ev"] > 0].groupby("market_key", as_index=False).agg(
        bets=("fight_id","size"), wins=("won","sum"), units=("realized_units","sum"), roi=("realized_units","mean"),
        avg_edge=("edge_raw","mean"), avg_model_ev=("model_ev","mean"),
    )
    coverage = coverage.merge(positive, on="market_key", how="left")

    # Best opportunities and worst misses for inspection.
    details = rows.sort_values("model_ev", ascending=False).copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out_dir / "ml_market_comparison.csv", index=False)
    strategy.to_csv(args.out_dir / "strategy_roi_by_threshold.csv", index=False)
    coverage.to_csv(args.out_dir / "market_coverage_and_positive_ev.csv", index=False)
    details.to_csv(args.out_dir / "bet_level_audit.csv", index=False)

    print("BANTAMWEIGHT MARKET EDGE AUDIT — i10_b0")
    print(f"Prediction fights: {pred['fight_id'].nunique()} | market-covered fights: {rows['fight_id'].nunique()} | graded outcomes: {len(rows)}")
    print("\nML MODEL VS MARKET")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nMARKET COVERAGE + POSITIVE MODEL EV")
    print(coverage.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nEDGE-THRESHOLD STRATEGY (TOP-ONLY)")
    show = strategy[(strategy["selector"].eq("edge_raw")) & (strategy["top_only"]) & (strategy["market_family"].isin(["moneyline","exact_method","all"]))]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nMODEL-EV STRATEGY (TOP-ONLY)")
    show = strategy[(strategy["selector"].eq("model_ev")) & (strategy["top_only"]) & (strategy["market_family"].isin(["moneyline","exact_method","all"]))]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
