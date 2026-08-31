# Triggered betting diagnostic for frozen hierarchical V5 OOF.
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research/prop_mispricing"
PRED = OUT / "xgboost_method_hierarchical_v5_oof_predictions.csv"
MARKET = ROOT / "data/market/historical_market_outcomes.parquet"
LEDGER = OUT / "xgboost_method_hierarchical_v5_betting_ledger.csv"
SUMMARY = OUT / "xgboost_method_hierarchical_v5_betting_summary.json"

THRESHOLD = 0.30
EPS = 1e-12
SLUGS = ["red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec"]
CLASS_META = {
    "red_ko": ("red", "win_by_ko_tko_dq", 0),
    "red_sub": ("red", "win_by_submission", 1),
    "red_dec": ("red", "win_by_decision", 2),
    "blue_ko": ("blue", "win_by_ko_tko_dq", 3),
    "blue_sub": ("blue", "win_by_submission", 4),
    "blue_dec": ("blue", "win_by_decision", 5),
}

def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def summarize(d):
    if d.empty:
        return {"bets": 0, "wins": 0, "losses": 0, "stake_units": 0.0, "profit_units": 0.0, "roi": None, "hit_rate": None, "fights_bet": 0, "bets_per_fight": None}
    stake = float(d["stake_units"].sum())
    profit = float(d["profit_units"].sum())
    return {
        "bets": int(len(d)),
        "wins": int(d["won"].sum()),
        "losses": int((1 - d["won"]).sum()),
        "stake_units": stake,
        "profit_units": profit,
        "roi": float(profit / stake) if stake else None,
        "hit_rate": float(d["won"].mean()),
        "fights_bet": int(d["fight_id"].nunique()),
        "bets_per_fight": float(len(d) / d["fight_id"].nunique()),
    }

pred = pd.read_csv(PRED)
pred["fight_id"] = pred["fight_id"].astype(str)
pred["date"] = pd.to_datetime(pred["date"])

m = pd.read_parquet(MARKET).copy()
m["fight_id"] = m["fight_id"].astype(str)
m = m[(m["bookmaker"] == "legacy_consensus") & m["outcome_side"].astype(str).isin(["red", "blue"])].copy()
m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
m = m[np.isfinite(m["implied_probability"]) & (m["implied_probability"] > 0) & (m["implied_probability"] < 1)].copy()

price_rows = []
for slug, (side, market_key, class_idx) in CLASS_META.items():
    z = m[(m["outcome_side"].astype(str) == side) & (m["market_key"] == market_key)].copy()
    z = z.sort_values(["fight_id"]).drop_duplicates("fight_id", keep=False)
    z = z[["fight_id", "implied_probability"]].copy()
    z["slug"] = slug
    z["class_idx"] = class_idx
    price_rows.append(z)
prices = pd.concat(price_rows, ignore_index=True)
price_map = {(r.fight_id, r.slug): float(r.implied_probability) for r in prices.itertuples(index=False)}

rows = []
for r in pred.itertuples(index=False):
    predicted_side = "red" if float(r.v5_model_p_red) >= 0.5 else "blue"
    for slug in SLUGS:
        side, _, class_idx = CLASS_META[slug]
        if side != predicted_side:
            continue
        model_p = float(getattr(r, f"hier_{slug}"))
        market_p = float(getattr(r, f"market_{slug}"))
        residual = float(logit(model_p) - logit(market_p))
        if residual < THRESHOLD:
            continue
        raw_imp = price_map.get((str(r.fight_id), slug))
        if raw_imp is None:
            continue
        decimal_odds = 1.0 / raw_imp
        won = int(int(r.target) == class_idx)
        profit = (decimal_odds - 1.0) if won else -1.0
        rows.append({
            "fight_id": str(r.fight_id),
            "date": r.date,
            "fold": str(r.fold),
            "event_name": r.event_name,
            "red_fighter": r.red_fighter,
            "blue_fighter": r.blue_fighter,
            "predicted_side": predicted_side,
            "bet_slug": slug,
            "model_probability": model_p,
            "normalized_market_probability": market_p,
            "signed_logit_residual": residual,
            "raw_implied_probability": raw_imp,
            "decimal_odds": decimal_odds,
            "won": won,
            "stake_units": 1.0,
            "profit_units": profit,
        })

ledger = pd.DataFrame(rows)
if not ledger.empty:
    ledger = ledger.sort_values(["date", "fight_id", "bet_slug"]).reset_index(drop=True)
ledger.to_csv(LEDGER, index=False)

by_fold = {}
if not ledger.empty:
    for fold, g in ledger.groupby("fold", sort=True):
        by_fold[str(fold)] = summarize(g)

multi = {"fights_with_1_bet": 0, "fights_with_2_bets": 0, "fights_with_3_bets": 0}
if not ledger.empty:
    counts = ledger.groupby("fight_id").size()
    for n in [1, 2, 3]:
        multi[f"fights_with_{n}_bet" if n == 1 else f"fights_with_{n}_bets"] = int((counts == n).sum())

summary = {
    "experiment": "hierarchical_v5_winner_side_only_method_betting_oof_v1",
    "model": "new hierarchical V5 exact-method model only",
    "period": "chronological 2021-2024 OOF",
    "rule": {
        "winner_side": "red if frozen V5 P(red) >= 0.5 else blue",
        "eligible_methods": "KO/SUB/DEC on predicted winner side only",
        "signed_logit_residual_threshold": THRESHOLD,
        "multiple_bets_per_fight_allowed": True,
        "stake_per_bet_units": 1.0,
        "residual_market_probability": "normalized six-way legacy-consensus market probability",
        "payout_price": "raw legacy-consensus implied probability converted to decimal odds as 1/p",
    },
    "pooled": summarize(ledger),
    "by_fold": by_fold,
    "bet_count_distribution": multi,
}
SUMMARY.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
