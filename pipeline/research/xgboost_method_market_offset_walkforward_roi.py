from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.research.xgboost_method_market_offset import (
    ROOT, FREEZE_PATH, MARKET_PATH, CLASS_SPECS, SLUGS, CLASS_ORDER,
    MARKET_COLS, _build_rows, _fit_predict, _metrics, _read_market,
)

OUT_PRED = ROOT / "xgboost_method_market_offset__2025_2026_walkforward_predictions.csv"
OUT_BETS = ROOT / "xgboost_method_market_offset__2025_2026_roi_bet_ledger.csv"
OUT_SUMMARY = ROOT / "xgboost_method_market_offset__2025_2026_roi_summary.json"
OUT_THRESH = ROOT / "xgboost_method_market_offset__2025_2026_roi_thresholds.csv"
OUT_YEAR = ROOT / "xgboost_method_market_offset__2025_2026_roi_by_year.csv"
OUT_CLASS = ROOT / "xgboost_method_market_offset__2025_2026_roi_by_class.csv"
OUT_BUCKET = ROOT / "xgboost_method_market_offset__2025_2026_roi_edge_buckets.csv"

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-12-31")
THRESHOLDS = [0.00, 0.025, 0.05, 0.075, 0.10]
BUCKETS = [-np.inf, 0, .025, .05, .075, .10, .15, np.inf]
BUCKET_LABELS = ["<0", "0-2.5pp", "2.5-5pp", "5-7.5pp", "7.5-10pp", "10-15pp", "15pp+"]


def _raw_prices() -> pd.DataFrame:
    m = _read_market(False)
    implied = m.pivot(index="fight_id", columns="class_slug", values="implied_probability").reindex(columns=SLUGS)
    implied = implied.dropna()
    out = pd.DataFrame({"fight_id": implied.index})
    for s in SLUGS:
        out[f"raw_{s}"] = implied[s].to_numpy(float)
    return out


def _roi(g: pd.DataFrame) -> dict:
    n = len(g)
    profit = float(g["profit_units"].sum()) if n else 0.0
    return {
        "bets": int(n), "wins": int(g["won"].sum()) if n else 0,
        "profit_units": profit, "roi": float(profit / n) if n else None,
        "avg_price_edge": float(g["price_edge"].mean()) if n else None,
        "avg_model_prob": float(g["model_prob"].mean()) if n else None,
        "avg_decimal_odds": float(g["decimal_odds"].mean()) if n else None,
    }


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    assert freeze["selected_candidate"] == "FULL"
    assert freeze["cold_start_eligibility"] == "OFF for this experiment"
    features = freeze["selected_features"]

    df, _, _ = _build_rows(False, True, forced_features=features)
    df["date"] = pd.to_datetime(df["date"])
    eval_df = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    if eval_df.empty:
        raise RuntimeError("no 2025-2026 six-way graded rows")

    pred_parts = []
    for d in sorted(eval_df["date"].drop_duplicates()):
        train = df[df["date"] < d].copy()
        val = eval_df[eval_df["date"] == d].copy()
        if train.empty or val.empty:
            continue
        pred, _, valid = _fit_predict(train, val, features)
        out = val[["fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target"]].copy()
        out["train_n"] = len(train)
        out["feature_count"] = len(valid)
        for j, s in enumerate(SLUGS):
            out[f"market_{s}"] = val[f"market_{s}"].to_numpy(float)
            out[f"model_{s}"] = pred[:, j]
        pred_parts.append(out)

    preds = pd.concat(pred_parts, ignore_index=True).sort_values(["date", "fight_id"]).reset_index(drop=True)
    prices = _raw_prices()
    preds = preds.merge(prices, on="fight_id", how="left", validate="one_to_one")
    preds.to_csv(OUT_PRED, index=False)

    bets = []
    for r in preds.itertuples(index=False):
        for j, s in enumerate(SLUGS):
            raw = float(getattr(r, f"raw_{s}"))
            fair = float(getattr(r, f"market_{s}"))
            mp = float(getattr(r, f"model_{s}"))
            if not np.isfinite(raw) or raw <= 0:
                continue
            dec = 1.0 / raw
            won = int(r.target == j)
            profit = (dec - 1.0) if won else -1.0
            bets.append({
                "fight_id": r.fight_id, "date": r.date, "year": int(pd.Timestamp(r.date).year),
                "event_name": r.event_name, "red_fighter": r.red_fighter, "blue_fighter": r.blue_fighter,
                "class_slug": s, "class_name": CLASS_ORDER[j], "won": won,
                "raw_implied_prob": raw, "fair_market_prob": fair, "model_prob": mp,
                "price_edge": mp - raw, "fair_edge": mp - fair,
                "decimal_odds": dec, "profit_units": profit,
            })
    bets = pd.DataFrame(bets).sort_values(["date", "fight_id", "class_slug"]).reset_index(drop=True)
    bets.to_csv(OUT_BETS, index=False)

    threshold_rows = []
    for t in THRESHOLDS:
        g = bets[bets["price_edge"] >= t]
        threshold_rows.append({"threshold": t, **_roi(g)})
    pd.DataFrame(threshold_rows).to_csv(OUT_THRESH, index=False)

    base = bets[bets["price_edge"] >= 0].copy()
    by_year = [{"year": int(y), **_roi(g)} for y, g in base.groupby("year")]
    pd.DataFrame(by_year).to_csv(OUT_YEAR, index=False)
    by_class = [{"class_name": c, **_roi(g)} for c, g in base.groupby("class_name")]
    pd.DataFrame(by_class).to_csv(OUT_CLASS, index=False)

    bets["edge_bucket"] = pd.cut(bets["price_edge"], bins=BUCKETS, labels=BUCKET_LABELS, right=False)
    by_bucket = [{"edge_bucket": str(b), **_roi(g)} for b, g in bets.groupby("edge_bucket", observed=True)]
    pd.DataFrame(by_bucket).to_csv(OUT_BUCKET, index=False)

    y = preds["target"].to_numpy(int)
    model_p = preds[[f"model_{s}" for s in SLUGS]].to_numpy(float)
    market_p = preds[MARKET_COLS].to_numpy(float)
    summary = {
        "status": "CONTAMINATED_SECONDARY_HOLDOUT_ANALYSIS",
        "warning": "2025+ was exposed by an earlier pre-compliance run; this is a chronological walk-forward ROI analysis, not a pristine holdout claim.",
        "period": ["2025-01-01", "2026-12-31"],
        "walk_forward_rule": "for each fight date, train only on graded fights strictly before that date; frozen FULL features/hyperparameters; cold start off",
        "bet_pricing": "legacy_consensus raw implied probability converted to decimal odds as 1/p; flat 1-unit stake",
        "threshold_policy": "descriptive predeclared price-edge grid; no threshold selected or tuned on 2025-2026",
        "fights": int(len(preds)),
        "market_metrics": _metrics(y, market_p),
        "model_metrics": _metrics(y, model_p),
        "thresholds": threshold_rows,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
