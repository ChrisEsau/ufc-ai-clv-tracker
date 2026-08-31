from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.research.xgboost_method_market_offset import (
    ROOT, FREEZE_PATH, CLASS_ORDER, SLUGS,
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
# Descriptive only. 0.20 is included to mirror the frozen V5 moneyline rule.
LOGIT_THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
BUCKETS = [-np.inf, 0, .05, .10, .15, .20, .25, .30, .40, np.inf]
BUCKET_LABELS = ["<0", "0-.05", ".05-.10", ".10-.15", ".15-.20", ".20-.25", ".25-.30", ".30-.40", ".40+"]


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(np.log(p / (1.0 - p)))


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
        "bets": int(n),
        "wins": int(g["won"].sum()) if n else 0,
        "profit_units": profit,
        "roi": float(profit / n) if n else None,
        "avg_signed_logit_residual": float(g["signed_logit_residual"].mean()) if n else None,
        "avg_price_edge": float(g["price_edge"].mean()) if n else None,
        "avg_model_prob": float(g["model_prob"].mean()) if n else None,
        "avg_decimal_odds": float(g["decimal_odds"].mean()) if n else None,
    }


def _select_one_per_fight(bets: pd.DataFrame, threshold: float) -> pd.DataFrame:
    eligible = bets[
        (bets["price_edge"] > 0)
        & (bets["signed_logit_residual"] >= float(threshold))
    ].copy()
    if eligible.empty:
        return eligible
    # Hard rule: at most one method bet per fight. Choose the strongest positive
    # model-vs-fair-market logit displacement. Stable class_slug tie-breaker.
    eligible = eligible.sort_values(
        ["date", "fight_id", "signed_logit_residual", "class_slug"],
        ascending=[True, True, False, True],
    )
    return eligible.drop_duplicates("fight_id", keep="first").reset_index(drop=True)


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

    bet_rows = []
    for r in preds.itertuples(index=False):
        for j, s in enumerate(SLUGS):
            raw = float(getattr(r, f"raw_{s}"))
            fair = float(getattr(r, f"market_{s}"))
            mp = float(getattr(r, f"model_{s}"))
            if not np.isfinite(raw) or raw <= 0 or raw >= 1:
                continue
            if not np.isfinite(fair) or fair <= 0 or fair >= 1:
                continue
            if not np.isfinite(mp) or mp <= 0 or mp >= 1:
                continue
            dec = 1.0 / raw
            won = int(r.target == j)
            profit = (dec - 1.0) if won else -1.0
            signed_logit_residual = _logit(mp) - _logit(fair)
            bet_rows.append({
                "fight_id": r.fight_id,
                "date": r.date,
                "year": int(pd.Timestamp(r.date).year),
                "event_name": r.event_name,
                "red_fighter": r.red_fighter,
                "blue_fighter": r.blue_fighter,
                "class_slug": s,
                "class_name": CLASS_ORDER[j],
                "won": won,
                "raw_implied_prob": raw,
                "fair_market_prob": fair,
                "model_prob": mp,
                "price_edge": mp - raw,
                "fair_edge": mp - fair,
                "signed_logit_residual": signed_logit_residual,
                "abs_logit_residual": abs(signed_logit_residual),
                "decimal_odds": dec,
                "profit_units": profit,
            })
    all_bets = pd.DataFrame(bet_rows).sort_values(["date", "fight_id", "class_slug"]).reset_index(drop=True)

    threshold_rows = []
    ledgers = []
    for t in LOGIT_THRESHOLDS:
        g = _select_one_per_fight(all_bets, t)
        g = g.copy()
        g["logit_threshold"] = t
        ledgers.append(g)
        threshold_rows.append({"logit_threshold": t, **_roi(g)})
    thresholds_df = pd.DataFrame(threshold_rows)
    thresholds_df.to_csv(OUT_THRESH, index=False)

    # Main committed ledger uses the frozen moneyline-comparable 0.20 logit gate.
    base = _select_one_per_fight(all_bets, 0.20)
    base.to_csv(OUT_BETS, index=False)

    by_year = [{"year": int(y), **_roi(g)} for y, g in base.groupby("year")]
    pd.DataFrame(by_year).to_csv(OUT_YEAR, index=False)
    by_class = [{"class_name": c, **_roi(g)} for c, g in base.groupby("class_name")]
    pd.DataFrame(by_class).to_csv(OUT_CLASS, index=False)

    all_bets["logit_bucket"] = pd.cut(
        all_bets["signed_logit_residual"], bins=BUCKETS, labels=BUCKET_LABELS, right=False
    )
    bucket_rows = []
    for b, g in all_bets.groupby("logit_bucket", observed=True):
        picked = g[(g["price_edge"] > 0)].sort_values(
            ["date", "fight_id", "signed_logit_residual", "class_slug"],
            ascending=[True, True, False, True],
        ).drop_duplicates("fight_id", keep="first")
        bucket_rows.append({"logit_bucket": str(b), **_roi(picked)})
    pd.DataFrame(bucket_rows).to_csv(OUT_BUCKET, index=False)

    y = preds["target"].to_numpy(int)
    model_p = preds[[f"model_{s}" for s in SLUGS]].to_numpy(float)
    market_p = preds[MARKET_COLS].to_numpy(float)
    summary = {
        "status": "CONTAMINATED_SECONDARY_HOLDOUT_ANALYSIS",
        "warning": "2025+ was exposed by an earlier pre-compliance run; this is a chronological walk-forward ROI analysis, not a pristine holdout claim.",
        "period": ["2025-01-01", "2026-12-31"],
        "walk_forward_rule": "for each fight date, train only on graded fights strictly before that date; frozen FULL features/hyperparameters; cold start off",
        "bet_pricing": "legacy_consensus raw implied probability is the offered-price proxy; decimal odds = 1/raw implied probability; flat 1-unit stake",
        "bet_eligibility": "model probability must exceed raw implied break-even probability AND signed logit(model)-logit(fair six-way market) must clear threshold",
        "one_bet_per_fight": "hard maximum one method bet per fight; choose eligible outcome with largest signed logit residual",
        "primary_logit_threshold": 0.20,
        "threshold_policy": "descriptive predeclared logit grid; 0.20 mirrors frozen V5 moneyline gate; no threshold tuned on 2025-2026",
        "fights": int(len(preds)),
        "market_metrics": _metrics(y, market_p),
        "model_metrics": _metrics(y, model_p),
        "thresholds": threshold_rows,
        "primary_020": _roi(base),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
