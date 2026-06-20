from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import MARKET_DIR, MODEL_LAB_DIR, PREDICTIONS_DIR, ensure_data_dirs

DEFAULT_MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"
DEFAULT_MARKET_OUTCOMES_PATH = MARKET_DIR / "historical_market_outcomes.parquet"
DEFAULT_OUTPUT_ROOT = MODEL_LAB_DIR / "backtests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generic V2 market/outcome-level backtest.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--market-key", default="moneyline")
    parser.add_argument("--model-outcomes-path", default=str(DEFAULT_MODEL_OUTCOMES_PATH))
    parser.add_argument("--historical-market-path", default=str(DEFAULT_MARKET_OUTCOMES_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--starting-bankroll", type=float, default=10000.0)
    parser.add_argument("--flat-stake", type=float, default=100.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--max-stake-pct", type=float, default=0.05)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-odds", type=float, default=-1000.0)
    parser.add_argument("--max-odds", type=float, default=3000.0)
    return parser.parse_args()


def american_profit_per_1(odds: Any) -> float | None:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or float(value) == 0:
        return None
    value = float(value)
    return value / 100.0 if value > 0 else 100.0 / abs(value)


def kelly_fraction(probability: Any, odds: Any) -> float:
    p = pd.to_numeric(pd.Series([probability]), errors="coerce").iloc[0]
    b = american_profit_per_1(odds)
    if pd.isna(p) or b is None or b <= 0:
        return 0.0
    q = 1.0 - float(p)
    k = ((b * float(p)) - q) / b
    return max(0.0, float(k))


def choose_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def load_inputs(model_path: Path, market_path: Path, market_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model outcomes not found: {model_path}")
    if not market_path.exists():
        raise FileNotFoundError(f"Historical market outcomes not found: {market_path}")
    model = pd.read_parquet(model_path)
    market = pd.read_parquet(market_path)
    model = model[model["market_key"].astype(str).str.lower() == market_key.lower()].copy()
    market = market[market["market_key"].astype(str).str.lower() == market_key.lower()].copy()
    return model, market


def standardize_model(model: pd.DataFrame) -> pd.DataFrame:
    probability_col = choose_col(model, ["model_probability", "predicted_probability", "probability"])
    confidence_col = choose_col(model, ["confidence_score", "confidence", "model_confidence"], required=False)
    keep = ["fight_id", "market_key", "outcome_join_key", probability_col]
    optional = [c for c in ["model_id", "event_name", "outcome_label", "outcome_fighter_id"] if c in model.columns]
    if confidence_col:
        optional.append(confidence_col)
    out = model[keep + optional].copy()
    out = out.rename(columns={probability_col: "model_probability"})
    if confidence_col:
        out = out.rename(columns={confidence_col: "confidence_score"})
    else:
        out["confidence_score"] = 1.0
    out["outcome_join_key"] = out["outcome_join_key"].astype(str)
    out["model_probability"] = pd.to_numeric(out["model_probability"], errors="coerce")
    out["confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce").fillna(1.0)
    return out


def standardize_market(market: pd.DataFrame) -> pd.DataFrame:
    required = ["fight_id", "market_key", "outcome_join_key", "american_odds", "implied_probability", "won"]
    missing = [c for c in required if c not in market.columns]
    if missing:
        raise ValueError(f"Historical market outcomes missing columns: {missing}")
    optional = [c for c in ["date", "event_name", "bookmaker", "outcome_label", "outcome_fighter_id"] if c in market.columns]
    out = market[required + optional].copy()
    out["outcome_join_key"] = out["outcome_join_key"].astype(str)
    out["american_odds"] = pd.to_numeric(out["american_odds"], errors="coerce")
    out["implied_probability"] = pd.to_numeric(out["implied_probability"], errors="coerce")
    out["won"] = out["won"].astype("boolean")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    if start_date and "date" in out.columns:
        out = out[out["date"] >= pd.to_datetime(start_date)]
    if end_date and "date" in out.columns:
        out = out[out["date"] <= pd.to_datetime(end_date)]
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


def score_bets(bets: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = bets.sort_values([c for c in ["date", "fight_id", "market_key"] if c in bets.columns]).copy()
    out["profit_per_1"] = out["american_odds"].map(american_profit_per_1)
    out["flat_stake"] = float(args.flat_stake)
    out["flat_profit"] = out.apply(lambda r: r["flat_stake"] * r["profit_per_1"] if bool(r["won"]) else -r["flat_stake"], axis=1)
    bankroll = float(args.starting_bankroll)
    bankrolls = []
    stakes = []
    profits = []
    for _, row in out.iterrows():
        raw_kelly = kelly_fraction(row["model_probability"], row["american_odds"])
        stake_pct = min(raw_kelly * float(args.kelly_fraction), float(args.max_stake_pct))
        stake = bankroll * stake_pct
        profit = stake * row["profit_per_1"] if bool(row["won"]) else -stake
        bankroll += profit
        stakes.append(stake)
        profits.append(profit)
        bankrolls.append(bankroll)
    out["kelly_stake"] = stakes
    out["kelly_profit"] = profits
    out["kelly_bankroll"] = bankrolls
    out["flat_bankroll"] = float(args.starting_bankroll) + out["flat_profit"].cumsum()
    return out


def summarize(scored: pd.DataFrame, args: argparse.Namespace, backtest_id: str) -> dict[str, Any]:
    total_bets = int(len(scored))
    flat_risked = float(scored["flat_stake"].sum()) if total_bets else 0.0
    kelly_risked = float(scored["kelly_stake"].sum()) if total_bets else 0.0
    flat_profit = float(scored["flat_profit"].sum()) if total_bets else 0.0
    kelly_profit = float(scored["kelly_profit"].sum()) if total_bets else 0.0
    return {
        "backtest_id": backtest_id,
        "model_id": args.model_id,
        "market_key": args.market_key,
        "total_bets": total_bets,
        "wins": int(scored["won"].sum()) if total_bets else 0,
        "win_rate": float(scored["won"].mean()) if total_bets else 0.0,
        "flat_risked": flat_risked,
        "flat_profit": flat_profit,
        "flat_roi": flat_profit / flat_risked if flat_risked else 0.0,
        "kelly_risked": kelly_risked,
        "kelly_profit": kelly_profit,
        "kelly_roi": kelly_profit / kelly_risked if kelly_risked else 0.0,
        "starting_bankroll": float(args.starting_bankroll),
        "ending_kelly_bankroll": float(scored["kelly_bankroll"].iloc[-1]) if total_bets else float(args.starting_bankroll),
        "min_edge": float(args.min_edge),
        "min_confidence": float(args.min_confidence),
        "min_odds": float(args.min_odds),
        "max_odds": float(args.max_odds),
        "start_date": getattr(args, "start_date", None),
        "end_date": getattr(args, "end_date", None),
    }


def bucket_summary(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=["bucket_type", "bucket", "bets", "win_rate", "flat_profit", "flat_roi"])
    frames = []
    specs = {
        "edge": [-1, 0, .05, .10, .15, .20, 10],
        "confidence": [0, .50, .60, .70, .80, .90, 1.01],
        "odds": [-10000, -300, -150, 100, 200, 500, 10000],
    }
    for name, bins in specs.items():
        temp = scored.copy()
        col = "american_odds" if name == "odds" else name if name != "confidence" else "confidence_score"
        temp["bucket"] = pd.cut(temp[col], bins=bins, include_lowest=True).astype(str)
        group = temp.groupby("bucket", dropna=False).agg(
            bets=("fight_id", "count"),
            win_rate=("won", "mean"),
            flat_profit=("flat_profit", "sum"),
            flat_risked=("flat_stake", "sum"),
        ).reset_index()
        group["flat_roi"] = group["flat_profit"] / group["flat_risked"].replace({0: pd.NA})
        group.insert(0, "bucket_type", name)
        frames.append(group.drop(columns=["flat_risked"]))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    run_time = datetime.now(timezone.utc)
    backtest_id = f"{args.model_id}_{args.market_key}_{run_time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / backtest_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_raw, market_raw = load_inputs(Path(args.model_outcomes_path), Path(args.historical_market_path), args.market_key)
    model = standardize_model(model_raw)
    market = standardize_market(market_raw)
    joined = model.merge(market, on=["fight_id", "market_key", "outcome_join_key"], how="inner", suffixes=("_model", "_market"))
    candidates = apply_filters(joined, args)
    scored = score_bets(candidates, args)
    summary = summarize(scored, args, backtest_id)
    buckets = bucket_summary(scored)
    config = vars(args) | {"backtest_id": backtest_id, "created_at_utc": run_time.isoformat()}
    (output_dir / "backtest_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (output_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    scored.to_parquet(output_dir / "backtest_bets.parquet", index=False)
    buckets.to_parquet(output_dir / "backtest_bucket_summary.parquet", index=False)
    registry_path = Path(args.output_root) / "backtest_registry.parquet"
    registry_row = pd.DataFrame([summary | {"output_dir": str(output_dir), "created_at_utc": run_time.isoformat()}])
    if registry_path.exists():
        registry = pd.read_parquet(registry_path)
        registry = pd.concat([registry, registry_row], ignore_index=True)
    else:
        registry = registry_row
    registry.to_parquet(registry_path, index=False)
    print("=" * 80)
    print("GENERIC BACKTEST V2")
    print("=" * 80)
    print("Backtest ID:", backtest_id)
    print("Model rows:", len(model))
    print("Market rows:", len(market))
    print("Joined rows:", len(joined))
    print("Bet candidates:", len(scored))
    print("Summary:")
    print(json.dumps(summary, indent=2, default=str))
    print("Output dir:", output_dir)


if __name__ == "__main__":
    main()
