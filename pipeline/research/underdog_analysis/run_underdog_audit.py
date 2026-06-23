"""Isolated favorite/underdog diagnostic research runner.

This side study reads historical model outcome probabilities and historical
market outcomes, then writes exploratory artifacts under ``data/research``.
It intentionally does not modify Model Lab, model configs, feature registries,
production model artifacts, live prediction outputs, or CLV artifacts.

Run from repo root:

    python -m pipeline.research.underdog_analysis.run_underdog_audit \
        --model-id moneyline_xgboost_v11
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.backtesting.run_backtest_v2 import (
    american_profit_per_1,
    kelly_fraction,
    standardize_market,
    standardize_model,
)
from pipeline.common.paths import MARKET_DIR, PREDICTIONS_DIR

DEFAULT_MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"
DEFAULT_MARKET_OUTCOMES_PATH = MARKET_DIR / "historical_market_outcomes.parquet"
DEFAULT_OUTPUT_ROOT = Path("data/research/underdog_audit")

DOG_BUCKET_ORDER = [
    "Favorite",
    "Small Underdog (+100 to +150)",
    "Medium Underdog (+150 to +250)",
    "Large Underdog (+250+)",
]
EDGE_BUCKETS = [-1.0, 0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 10.0]
PROBABILITY_BUCKETS = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated underdog/favorite audit.")
    parser.add_argument("--model-id", required=True, help="Model ID to audit from model_outcomes.parquet.")
    parser.add_argument("--market-key", default="moneyline")
    parser.add_argument("--model-outcomes-path", default=str(DEFAULT_MODEL_OUTCOMES_PATH))
    parser.add_argument("--historical-market-path", default=str(DEFAULT_MARKET_OUTCOMES_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--flat-stake", type=float, default=100.0)
    parser.add_argument("--starting-bankroll", type=float, default=10000.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--max-stake-pct", type=float, default=0.05)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-odds", type=float, default=-1000.0)
    parser.add_argument("--max-odds", type=float, default=3000.0)
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = Path(args.model_outcomes_path)
    market_path = Path(args.historical_market_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model outcomes not found: {model_path}")
    if not market_path.exists():
        raise FileNotFoundError(f"Historical market outcomes not found: {market_path}")

    model_raw = pd.read_parquet(model_path)
    market_raw = pd.read_parquet(market_path)

    model_raw = model_raw[model_raw["market_key"].astype(str).str.lower() == args.market_key.lower()].copy()
    market_raw = market_raw[market_raw["market_key"].astype(str).str.lower() == args.market_key.lower()].copy()
    if "model_id" in model_raw.columns:
        model_raw = model_raw[model_raw["model_id"].astype(str) == str(args.model_id)].copy()

    return standardize_model(model_raw), standardize_market(market_raw)


def favorite_bucket(odds: Any) -> str:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "Unknown"
    value = float(value)
    if value < 100:
        return "Favorite"
    if value <= 150:
        return "Small Underdog (+100 to +150)"
    if value <= 250:
        return "Medium Underdog (+150 to +250)"
    return "Large Underdog (+250+)"


def apply_research_filters(joined: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = joined.copy()
    if args.start_date and "date" in out.columns:
        out = out[out["date"] >= pd.to_datetime(args.start_date)]
    if args.end_date and "date" in out.columns:
        out = out[out["date"] <= pd.to_datetime(args.end_date)]

    out["edge"] = out["model_probability"] - out["implied_probability"]
    mask = (
        out["model_probability"].notna()
        & out["implied_probability"].notna()
        & out["american_odds"].between(args.min_odds, args.max_odds)
        & out["edge"].ge(args.min_edge)
        & out["confidence_score"].ge(args.min_confidence)
        & out["won"].notna()
    )
    return out[mask].copy()


def score_candidates(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = candidates.sort_values([c for c in ["date", "fight_id", "market_key"] if c in candidates.columns]).copy()
    out["favorite_bucket"] = out["american_odds"].map(favorite_bucket)
    out["side_type"] = out["favorite_bucket"].where(out["favorite_bucket"].eq("Favorite"), "Underdog")
    out["profit_per_1"] = out["american_odds"].map(american_profit_per_1)
    out["flat_stake"] = float(args.flat_stake)
    out["flat_profit"] = out.apply(
        lambda row: row["flat_stake"] * row["profit_per_1"] if bool(row["won"]) else -row["flat_stake"],
        axis=1,
    )

    bankroll = float(args.starting_bankroll)
    stakes: list[float] = []
    profits: list[float] = []
    bankrolls: list[float] = []
    for _, row in out.iterrows():
        raw_kelly = kelly_fraction(row["model_probability"], row["american_odds"])
        stake_pct = min(raw_kelly * float(args.kelly_fraction), float(args.max_stake_pct))
        stake = bankroll * stake_pct
        profit = stake * row["profit_per_1"] if bool(row["won"]) else -stake
        bankroll += profit
        stakes.append(float(stake))
        profits.append(float(profit))
        bankrolls.append(float(bankroll))

    out["kelly_stake"] = stakes
    out["kelly_profit"] = profits
    out["kelly_bankroll"] = bankrolls
    out["flat_bankroll"] = float(args.starting_bankroll) + out["flat_profit"].cumsum()
    return out


def _summarize_group(group: pd.DataFrame) -> pd.Series:
    flat_risked = group["flat_stake"].sum()
    kelly_risked = group["kelly_stake"].sum()
    return pd.Series(
        {
            "bets": int(len(group)),
            "wins": int(group["won"].sum()),
            "win_rate": float(group["won"].mean()) if len(group) else 0.0,
            "avg_model_probability": float(group["model_probability"].mean()) if len(group) else 0.0,
            "avg_implied_probability": float(group["implied_probability"].mean()) if len(group) else 0.0,
            "avg_edge": float(group["edge"].mean()) if len(group) else 0.0,
            "calibration_error": float(group["won"].mean() - group["model_probability"].mean()) if len(group) else 0.0,
            "flat_profit": float(group["flat_profit"].sum()),
            "flat_risked": float(flat_risked),
            "flat_roi": float(group["flat_profit"].sum() / flat_risked) if flat_risked else 0.0,
            "kelly_profit": float(group["kelly_profit"].sum()),
            "kelly_risked": float(kelly_risked),
            "kelly_roi": float(group["kelly_profit"].sum() / kelly_risked) if kelly_risked else 0.0,
        }
    )


def summarize_by(scored: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    return scored.groupby(columns, dropna=False).apply(_summarize_group).reset_index()


def build_calibration(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    frames = []
    by_fav = summarize_by(scored, ["favorite_bucket"])
    by_fav.insert(0, "bucket_type", "favorite_bucket")
    by_side = summarize_by(scored, ["side_type"])
    by_side.insert(0, "bucket_type", "side_type")

    prob = scored.copy()
    prob["probability_bucket"] = pd.cut(prob["model_probability"], bins=PROBABILITY_BUCKETS, include_lowest=True).astype(str)
    by_prob = summarize_by(prob, ["side_type", "probability_bucket"])
    by_prob.insert(0, "bucket_type", "side_probability_bucket")

    frames.extend([by_fav, by_side, by_prob])
    return pd.concat(frames, ignore_index=True, sort=False)


def build_edge_distribution(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    out = scored.copy()
    out["edge_bucket"] = pd.cut(out["edge"], bins=EDGE_BUCKETS, include_lowest=True).astype(str)
    return summarize_by(out, ["side_type", "edge_bucket"])


def build_summary(scored: pd.DataFrame, args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    total = _summarize_group(scored) if not scored.empty else pd.Series(dtype=float)
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "market_key": args.market_key,
        "rows": int(len(scored)),
        "total": total.to_dict() if not scored.empty else {},
        "filters": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "min_edge": args.min_edge,
            "min_confidence": args.min_confidence,
            "min_odds": args.min_odds,
            "max_odds": args.max_odds,
        },
        "artifact_note": "Research-only output. Does not modify Model Lab, configs, production models, live predictions, or CLV artifacts.",
    }


def write_outputs(scored: pd.DataFrame, args: argparse.Namespace, run_id: str) -> Path:
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    roi = summarize_by(scored, ["side_type"])
    fav_calibration = build_calibration(scored)
    edge_distribution = build_edge_distribution(scored)
    odds_buckets = summarize_by(scored, ["favorite_bucket"])
    summary = build_summary(scored, args, run_id)

    scored.to_parquet(output_dir / "underdog_predictions.parquet", index=False)
    roi.to_csv(output_dir / "underdog_roi.csv", index=False)
    fav_calibration.to_csv(output_dir / "underdog_calibration.csv", index=False)
    edge_distribution.to_csv(output_dir / "underdog_edge_distribution.csv", index=False)
    odds_buckets.to_csv(output_dir / "underdog_odds_buckets.csv", index=False)
    (output_dir / "underdog_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    registry_path = Path(args.output_root) / "underdog_audit_registry.parquet"
    registry_row = pd.DataFrame([{**summary, "output_dir": str(output_dir)}])
    if registry_path.exists():
        registry = pd.read_parquet(registry_path)
        registry = pd.concat([registry, registry_row], ignore_index=True)
    else:
        registry = registry_row
    registry.to_parquet(registry_path, index=False)
    return output_dir


def main() -> None:
    args = parse_args()
    run_time = datetime.now(timezone.utc)
    run_id = f"{args.model_id}_{args.market_key}_underdog_audit_{run_time.strftime('%Y%m%d_%H%M%S')}"
    model, market = load_inputs(args)
    joined = model.merge(market, on=["fight_id", "market_key", "outcome_join_key"], how="inner", suffixes=("_model", "_market"))
    candidates = apply_research_filters(joined, args)
    scored = score_candidates(candidates, args)
    output_dir = write_outputs(scored, args, run_id)

    print("=" * 80)
    print("UNDERDOG AUDIT RESEARCH")
    print("=" * 80)
    print("Run ID:", run_id)
    print("Model rows:", len(model))
    print("Market rows:", len(market))
    print("Joined rows:", len(joined))
    print("Filtered candidates:", len(scored))
    print("Output dir:", output_dir)
    if not scored.empty:
        print("Favorite/underdog ROI:")
        print(summarize_by(scored, ["side_type"]).to_string(index=False))


if __name__ == "__main__":
    main()
