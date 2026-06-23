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
    return pd.Series(
        {
            "bets": int(len(group)),
            "wins": int(group["won"].sum()),
            "win_rate": float(group["won"].mean()) if len(group) else 0.0,
            "avg_edge": float(group["edge"].mean()) if len(group) else 0.0,
            "flat_profit": float(flat_profit),
            "flat_risked": float(flat_risked),
            "flat_roi": float(flat_profit / flat_risked) if flat_risked else 0.0,
            "kelly_profit": float(kelly_profit),
            "kelly_risked": float(kelly_risked),
            "kelly_roi": float(kelly_profit / kelly_risked) if kelly_risked else 0.0,
        }
    )


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
    for column in ["edge", "confidence_score", "american_odds", "won", "flat_profit", "flat_stake", "kelly_profit", "kelly_stake"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["side_type"] = out["american_odds"].map(favorite_bucket)
    out = out[
        out["edge"].ge(args.min_edge)
        & out["confidence_score"].ge(args.min_confidence)
        & out["american_odds"].between(args.min_odds, args.max_odds)
        & out["won"].notna()
    ].copy()

    run_id = f"{args.model_id}_{args.market_key}_underdog_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_dir / "underdog_predictions.parquet", index=False)
    by_side = out.groupby("side_type", dropna=False).apply(summarize).reset_index() if not out.empty else pd.DataFrame()
    by_side.to_csv(output_dir / "underdog_roi.csv", index=False)
    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "market_key": args.market_key,
        "source_backtest_dir": str(backtest_dir),
        "source_bet_file": str(bet_file),
        "raw_bet_rows": int(len(bets)),
        "filtered_rows": int(len(out)),
    }
    (output_dir / "underdog_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    registry_path = Path(args.output_root) / "underdog_audit_registry.csv"
    row = pd.DataFrame([{**summary, "output_dir": str(output_dir)}])
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
    if not by_side.empty:
        print(by_side.to_string(index=False))


if __name__ == "__main__":
    main()
