"""Research-only underdog audit from Model Lab backtest bet artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BACKTEST_ROOT = Path("data/model_lab/backtests")
OUTPUT_ROOT = Path("data/research/underdog_audit")
BET_FILES = [
    "backtest_bets.parquet",
    "backtest_results.parquet",
    "bet_results.parquet",
    "backtest_bets.csv",
    "backtest_results.csv",
    "bet_results.csv",
]
IMPLIED_PROB_BINS = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
IMPLIED_PROB_LABELS = ["<20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
DELTA_BINS = [-10.0, 0.0, 0.05, 0.10, 0.15, 0.20, 10.0]
DELTA_LABELS = ["<=0%", "0-5%", "5-10%", "10-15%", "15-20%", "20%+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit underdog performance from Model Lab backtest bets.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--market-key", default="moneyline")
    parser.add_argument("--backtest-dir", default="")
    parser.add_argument("--backtest-root", default=str(BACKTEST_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-odds", type=float, default=-1000.0)
    parser.add_argument("--max-odds", type=float, default=3000.0)
    return parser.parse_args()


def latest_backtest_dir(model_id: str, market_key: str, root: Path) -> Path:
    prefix = f"{model_id}_{market_key}"
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        candidates = [p for p in root.iterdir() if p.is_dir() and model_id in p.name]
    if not candidates:
        examples = sorted(p.name for p in root.iterdir() if p.is_dir())[:25] if root.exists() else []
        raise FileNotFoundError(f"No Model Lab backtest dir found for {model_id}/{market_key}. Examples: {examples}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def find_bet_file(backtest_dir: Path) -> Path:
    for name in BET_FILES:
        path = backtest_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No bet-level file found in {backtest_dir}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def implied_probability_from_american(odds) -> float:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or float(value) == 0:
        return float("nan")
    value = float(value)
    if value < 0:
        return abs(value) / (abs(value) + 100.0)
    return 100.0 / (value + 100.0)


def favorite_bucket(odds) -> str:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "Unknown"
    return "Favorite" if float(value) < 100 else "Underdog"


def summarize(group: pd.DataFrame) -> pd.Series:
    flat_risked = group["flat_stake"].sum()
    flat_profit = group["flat_profit"].sum()
    kelly_risked = group["kelly_stake"].sum() if "kelly_stake" in group.columns else 0.0
    kelly_profit = group["kelly_profit"].sum() if "kelly_profit" in group.columns else 0.0
    market_probability = group["implied_probability"].mean() if "implied_probability" in group.columns else float("nan")
    model_probability = group["model_probability"].mean() if "model_probability" in group.columns else float("nan")
    actual_probability = group["won"].mean() if len(group) else 0.0
    return pd.Series(
        {
            "bets": int(len(group)),
            "wins": int(group["won"].sum()),
            "market_probability": float(market_probability) if pd.notna(market_probability) else 0.0,
            "model_probability": float(model_probability) if pd.notna(model_probability) else 0.0,
            "actual_probability": float(actual_probability),
            "market_error": float(actual_probability - market_probability) if pd.notna(market_probability) else 0.0,
            "model_error": float(actual_probability - model_probability) if pd.notna(model_probability) else 0.0,
            "model_market_delta": float(model_probability - market_probability) if pd.notna(model_probability) and pd.notna(market_probability) else 0.0,
            "avg_edge": float(group["edge"].mean()) if len(group) else 0.0,
            "flat_profit": float(flat_profit),
            "flat_risked": float(flat_risked),
            "flat_roi": float(flat_profit / flat_risked) if flat_risked else 0.0,
            "kelly_profit": float(kelly_profit),
            "kelly_risked": float(kelly_risked),
            "kelly_roi": float(kelly_profit / kelly_risked) if kelly_risked else 0.0,
        }
    )


def summarize_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby(group_cols, dropna=False).apply(summarize).reset_index()


def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir) if args.backtest_dir else latest_backtest_dir(args.model_id, args.market_key, Path(args.backtest_root))
    bet_file = find_bet_file(backtest_dir)
    bets = read_table(bet_file)

    required = {"edge", "confidence_score", "american_odds", "won", "flat_profit", "flat_stake"}
    missing = sorted(required - set(bets.columns))
    if missing:
        raise ValueError(f"Backtest bet table missing required columns: {missing}")

    out = bets.copy()
    for column in ["edge", "confidence_score", "american_odds", "won", "flat_profit", "flat_stake", "kelly_profit", "kelly_stake", "model_probability", "implied_probability"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "implied_probability" not in out.columns:
        out["implied_probability"] = out["american_odds"].map(implied_probability_from_american)
    if "model_probability" not in out.columns:
        raise ValueError("Backtest bet table missing model_probability; probability calibration requires it.")
    out["side_type"] = out["american_odds"].map(favorite_bucket)
    out = out[
        out["edge"].ge(args.min_edge)
        & out["confidence_score"].ge(args.min_confidence)
        & out["american_odds"].between(args.min_odds, args.max_odds)
        & out["won"].notna()
        & out["implied_probability"].notna()
        & out["model_probability"].notna()
    ].copy()

    out["implied_probability_bucket"] = pd.cut(
        out["implied_probability"],
        bins=IMPLIED_PROB_BINS,
        labels=IMPLIED_PROB_LABELS,
        include_lowest=True,
    ).astype(str)
    out["model_market_delta_bucket"] = pd.cut(
        out["model_probability"] - out["implied_probability"],
        bins=DELTA_BINS,
        labels=DELTA_LABELS,
        include_lowest=True,
    ).astype(str)

    run_id = f"{args.model_id}_{args.market_key}_underdog_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_dir / "underdog_predictions.parquet", index=False)

    by_side = summarize_by(out, ["side_type"])
    by_implied = summarize_by(out, ["implied_probability_bucket"])
    by_side_implied = summarize_by(out, ["side_type", "implied_probability_bucket"])
    by_delta = summarize_by(out, ["side_type", "model_market_delta_bucket"])

    by_side.to_csv(output_dir / "underdog_roi.csv", index=False)
    by_implied.to_csv(output_dir / "probability_calibration_by_implied_bucket.csv", index=False)
    by_side_implied.to_csv(output_dir / "probability_calibration_by_side_and_implied_bucket.csv", index=False)
    by_delta.to_csv(output_dir / "probability_calibration_by_model_market_delta.csv", index=False)

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "market_key": args.market_key,
        "source_backtest_dir": str(backtest_dir),
        "source_bet_file": str(bet_file),
        "raw_bet_rows": int(len(bets)),
        "filtered_rows": int(len(out)),
        "outputs": {
            "side_summary": "underdog_roi.csv",
            "implied_probability_calibration": "probability_calibration_by_implied_bucket.csv",
            "side_implied_probability_calibration": "probability_calibration_by_side_and_implied_bucket.csv",
            "model_market_delta_calibration": "probability_calibration_by_model_market_delta.csv",
        },
    }
    (output_dir / "underdog_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    registry_path = Path(args.output_root) / "underdog_audit_registry.csv"
    row = pd.DataFrame([{**summary, "output_dir": str(output_dir)}]).drop(columns=["outputs"])
    if registry_path.exists():
        row = pd.concat([pd.read_csv(registry_path), row], ignore_index=True)
    row.to_csv(registry_path, index=False)

    print("=" * 80)
    print("UNDERDOG AUDIT RESEARCH FROM MODEL LAB BACKTEST")
    print("=" * 80)
    print("Run ID:", run_id)
    print("Source bet file:", bet_file)
    print("Raw bet rows:", len(bets))
    print("Filtered rows:", len(out))
    print("Output dir:", output_dir)
    print("\nProbability calibration by implied probability bucket:")
    if not by_implied.empty:
        print(by_implied.to_string(index=False))
    print("\nFavorite/underdog summary:")
    if not by_side.empty:
        print(by_side.to_string(index=False))


if __name__ == "__main__":
    main()
