"""Research-only probability audit from Model Lab backtest artifacts."""

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
ALL_ROW_FILES = [
    "joined_model_market_rows.parquet",
    "joined_model_market_rows.csv",
]
IMPLIED_PROB_BINS = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
IMPLIED_PROB_LABELS = ["<20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
MODEL_PROB_BINS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
MODEL_PROB_LABELS = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
DELTA_BINS = [-10.0, 0.0, 0.05, 0.10, 0.15, 0.20, 10.0]
DELTA_LABELS = ["<=0%", "0-5%", "5-10%", "10-15%", "15-20%", "20%+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit probability calibration from Model Lab backtest artifacts.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--market-key", default="moneyline")
    parser.add_argument("--backtest-dir", default="")
    parser.add_argument("--backtest-root", default=str(BACKTEST_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--source", choices=["bets", "all_rows"], default="bets")
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


def find_first_existing(backtest_dir: Path, names: list[str]) -> Path:
    for name in names:
        path = backtest_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No source file found in {backtest_dir}; expected one of {names}")


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
    flat_risked = group["flat_stake"].sum() if "flat_stake" in group.columns else 0.0
    flat_profit = group["flat_profit"].sum() if "flat_profit" in group.columns else 0.0
    kelly_risked = group["kelly_stake"].sum() if "kelly_stake" in group.columns else 0.0
    kelly_profit = group["kelly_profit"].sum() if "kelly_profit" in group.columns else 0.0
    market_probability = group["implied_probability"].mean() if "implied_probability" in group.columns else float("nan")
    model_probability = group["model_probability"].mean() if "model_probability" in group.columns else float("nan")
    actual_probability = group["won"].mean() if len(group) else 0.0
    return pd.Series(
        {
            "rows": int(len(group)),
            "wins": int(group["won"].sum()),
            "market_probability": float(market_probability) if pd.notna(market_probability) else 0.0,
            "model_probability": float(model_probability) if pd.notna(model_probability) else 0.0,
            "actual_probability": float(actual_probability),
            "market_error": float(actual_probability - market_probability) if pd.notna(market_probability) else 0.0,
            "model_error": float(actual_probability - model_probability) if pd.notna(model_probability) else 0.0,
            "model_market_delta": float(model_probability - market_probability) if pd.notna(model_probability) and pd.notna(market_probability) else 0.0,
            "avg_edge": float(group["edge"].mean()) if "edge" in group.columns and len(group) else 0.0,
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


def normalize_source(df: pd.DataFrame, source: str, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    for column in ["edge", "confidence_score", "american_odds", "won", "flat_profit", "flat_stake", "kelly_profit", "kelly_stake", "model_probability", "implied_probability"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "implied_probability" not in out.columns and "american_odds" in out.columns:
        out["implied_probability"] = out["american_odds"].map(implied_probability_from_american)
    if "edge" not in out.columns and {"model_probability", "implied_probability"}.issubset(out.columns):
        out["edge"] = out["model_probability"] - out["implied_probability"]
    if "confidence_score" not in out.columns:
        out["confidence_score"] = 1.0
    if "model_probability" not in out.columns:
        raise ValueError("Source table missing model_probability; probability calibration requires it.")
    if "won" not in out.columns:
        raise ValueError("Source table missing won; probability calibration requires realized outcomes.")
    if "american_odds" in out.columns:
        out["side_type"] = out["american_odds"].map(favorite_bucket)
    else:
        out["side_type"] = "Unknown"

    mask = out["won"].notna() & out["model_probability"].notna()
    if "implied_probability" in out.columns:
        mask = mask & out["implied_probability"].notna()
    if source == "bets":
        mask = (
            mask
            & out["edge"].ge(args.min_edge)
            & out["confidence_score"].ge(args.min_confidence)
            & out["american_odds"].between(args.min_odds, args.max_odds)
        )
    return out[mask].copy()


def add_probability_buckets(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["model_probability_bucket"] = pd.cut(
        out["model_probability"],
        bins=MODEL_PROB_BINS,
        labels=MODEL_PROB_LABELS,
        include_lowest=True,
    ).astype(str)
    if "implied_probability" in out.columns:
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
    return out


def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir) if args.backtest_dir else latest_backtest_dir(args.model_id, args.market_key, Path(args.backtest_root))
    source_file = find_first_existing(backtest_dir, ALL_ROW_FILES if args.source == "all_rows" else BET_FILES)
    raw = read_table(source_file)
    out = add_probability_buckets(normalize_source(raw, args.source, args))

    run_id = f"{args.model_id}_{args.market_key}_{args.source}_probability_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_dir / "probability_audit_rows.parquet", index=False)

    by_side = summarize_by(out, ["side_type"])
    by_model = summarize_by(out, ["model_probability_bucket"])
    by_side_model = summarize_by(out, ["side_type", "model_probability_bucket"])
    by_model.to_csv(output_dir / "all_rows_probability_calibration_by_model_bucket.csv", index=False)
    by_side_model.to_csv(output_dir / "all_rows_probability_calibration_by_model_bucket_and_side.csv", index=False)
    by_side.to_csv(output_dir / "all_rows_probability_calibration_by_side.csv", index=False)

    outputs = {
        "audit_rows": "probability_audit_rows.parquet",
        "model_probability_calibration": "all_rows_probability_calibration_by_model_bucket.csv",
        "side_model_probability_calibration": "all_rows_probability_calibration_by_model_bucket_and_side.csv",
        "side_summary": "all_rows_probability_calibration_by_side.csv",
    }
    if "implied_probability_bucket" in out.columns:
        by_implied = summarize_by(out, ["implied_probability_bucket"])
        by_side_implied = summarize_by(out, ["side_type", "implied_probability_bucket"])
        by_delta = summarize_by(out, ["side_type", "model_market_delta_bucket"])
        by_implied.to_csv(output_dir / "all_rows_probability_calibration_by_implied_bucket.csv", index=False)
        by_side_implied.to_csv(output_dir / "all_rows_probability_calibration_by_side_and_implied_bucket.csv", index=False)
        by_delta.to_csv(output_dir / "all_rows_probability_calibration_by_model_market_delta.csv", index=False)
        outputs.update(
            {
                "implied_probability_calibration": "all_rows_probability_calibration_by_implied_bucket.csv",
                "side_implied_probability_calibration": "all_rows_probability_calibration_by_side_and_implied_bucket.csv",
                "model_market_delta_calibration": "all_rows_probability_calibration_by_model_market_delta.csv",
            }
        )

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "market_key": args.market_key,
        "source": args.source,
        "source_backtest_dir": str(backtest_dir),
        "source_file": str(source_file),
        "raw_rows": int(len(raw)),
        "audit_rows": int(len(out)),
        "outputs": outputs,
    }
    (output_dir / "probability_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    registry_path = Path(args.output_root) / "probability_audit_registry.csv"
    row = pd.DataFrame([{**summary, "output_dir": str(output_dir)}]).drop(columns=["outputs"])
    if registry_path.exists():
        row = pd.concat([pd.read_csv(registry_path), row], ignore_index=True)
    row.to_csv(registry_path, index=False)

    print("=" * 80)
    print("MODEL PROBABILITY AUDIT FROM MODEL LAB BACKTEST")
    print("=" * 80)
    print("Run ID:", run_id)
    print("Source:", args.source)
    print("Source file:", source_file)
    print("Raw rows:", len(raw))
    print("Audit rows:", len(out))
    print("Output dir:", output_dir)
    print("\nProbability calibration by model probability bucket:")
    if not by_model.empty:
        print(by_model.to_string(index=False))
    print("\nSide summary:")
    if not by_side.empty:
        print(by_side.to_string(index=False))


if __name__ == "__main__":
    main()
