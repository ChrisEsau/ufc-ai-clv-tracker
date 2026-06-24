from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_BACKTEST_DIR = Path(
    "data/model_lab/backtests/moneyline_xgboost_v11_moneyline_full_20260623_220808"
)
DEFAULT_OUTPUT_ROOT = Path("data/research/backtest_bet_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Model Lab backtest_bets.parquet by year, side, confidence, and edge."
    )
    parser.add_argument(
        "--backtest-dir",
        default=str(DEFAULT_BACKTEST_DIR),
        help="Backtest artifact directory containing backtest_bets.parquet.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where audit outputs should be written.",
    )
    return parser.parse_args()


def american_implied_probability(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    positive = odds > 0
    implied = pd.Series(pd.NA, index=odds.index, dtype="Float64")
    implied.loc[positive] = 100.0 / (odds.loc[positive] + 100.0)
    implied.loc[~positive & odds.notna()] = (-odds.loc[~positive & odds.notna()]) / (
        -odds.loc[~positive & odds.notna()] + 100.0
    )
    return implied.astype("float64")


def load_bets(backtest_dir: Path) -> pd.DataFrame:
    path = backtest_dir / "backtest_bets.parquet"
    if not path.exists():
        raise FileNotFoundError(f"backtest_bets.parquet not found: {path}")
    bets = pd.read_parquet(path).copy()
    if bets.empty:
        raise ValueError(f"backtest_bets.parquet is empty: {path}")
    return bets


def prepare_bets(bets: pd.DataFrame) -> pd.DataFrame:
    out = bets.copy()
    if "date" not in out.columns:
        raise ValueError("backtest_bets.parquet must include a date column.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = out["date"].dt.year

    numeric_columns = [
        "won",
        "model_probability",
        "implied_probability",
        "edge",
        "confidence_score",
        "american_odds",
        "flat_profit",
        "flat_stake",
        "kelly_profit",
        "kelly_stake",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    if "implied_probability" not in out.columns and "american_odds" in out.columns:
        out["implied_probability"] = american_implied_probability(out["american_odds"])

    if "side_type" not in out.columns and "american_odds" in out.columns:
        out["side_type"] = out["american_odds"].apply(
            lambda odds: "Favorite" if pd.notna(odds) and float(odds) < 0 else "Underdog"
        )

    if "confidence_score" not in out.columns and "model_probability" in out.columns:
        out["confidence_score"] = out["model_probability"].where(
            out["model_probability"] >= 0.5,
            1.0 - out["model_probability"],
        )

    if "edge" not in out.columns and {"model_probability", "implied_probability"}.issubset(out.columns):
        out["edge"] = out["model_probability"] - out["implied_probability"]

    return out


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    aggregations: dict[str, Any] = {
        "bets": ("won", "size"),
        "wins": ("won", "sum"),
        "win_rate": ("won", "mean"),
    }

    optional_means = [
        "model_probability",
        "implied_probability",
        "edge",
        "confidence_score",
        "american_odds",
    ]
    for column in optional_means:
        if column in df.columns:
            aggregations[f"avg_{column}"] = (column, "mean")

    if "flat_profit" in df.columns:
        aggregations["flat_profit"] = ("flat_profit", "sum")
    if "flat_stake" in df.columns:
        aggregations["flat_risked"] = ("flat_stake", "sum")
    if "kelly_profit" in df.columns:
        aggregations["kelly_profit"] = ("kelly_profit", "sum")
    if "kelly_stake" in df.columns:
        aggregations["kelly_risked"] = ("kelly_stake", "sum")

    summary = df.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()

    if {"flat_profit", "flat_risked"}.issubset(summary.columns):
        summary["flat_roi"] = summary["flat_profit"] / summary["flat_risked"].replace({0: pd.NA})
    if {"kelly_profit", "kelly_risked"}.issubset(summary.columns):
        summary["kelly_roi"] = summary["kelly_profit"] / summary["kelly_risked"].replace({0: pd.NA})

    return summary


def build_bucket_summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = df.copy()

    if "model_probability" in out.columns:
        out["model_probability_bucket"] = pd.cut(
            out["model_probability"],
            bins=[0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
            labels=["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"],
            include_lowest=True,
        ).astype(str)

    if "confidence_score" in out.columns:
        out["confidence_bucket"] = pd.cut(
            out["confidence_score"],
            bins=[0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0],
            labels=["<=55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-85%", "85-90%", "90-100%"],
            include_lowest=True,
        ).astype(str)

    if "edge" in out.columns:
        out["edge_bucket"] = pd.cut(
            out["edge"],
            bins=[-10.0, 0.0, 0.05, 0.10, 0.15, 0.20, 10.0],
            labels=["<=0%", "0-5%", "5-10%", "10-15%", "15-20%", "20%+"],
            include_lowest=True,
        ).astype(str)

    summaries: dict[str, pd.DataFrame] = {}
    summaries["year_summary"] = summarize_group(out, ["year"])
    if "side_type" in out.columns:
        summaries["year_side_summary"] = summarize_group(out, ["year", "side_type"])
        summaries["side_summary"] = summarize_group(out, ["side_type"])
    if "model_probability_bucket" in out.columns:
        summaries["year_model_probability_bucket_summary"] = summarize_group(out, ["year", "model_probability_bucket"])
        summaries["model_probability_bucket_summary"] = summarize_group(out, ["model_probability_bucket"])
    if "confidence_bucket" in out.columns:
        summaries["year_confidence_bucket_summary"] = summarize_group(out, ["year", "confidence_bucket"])
    if "edge_bucket" in out.columns:
        summaries["year_edge_bucket_summary"] = summarize_group(out, ["year", "edge_bucket"])
    return summaries


def select_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir)
    output_root = Path(args.output_root)
    run_time = datetime.now(timezone.utc)
    audit_id = f"{backtest_dir.name}_bet_audit_{run_time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_root / audit_id
    output_dir.mkdir(parents=True, exist_ok=True)

    bets = prepare_bets(load_bets(backtest_dir))
    summaries = build_bucket_summaries(bets)

    for name, table in summaries.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    top_columns = select_columns(
        bets,
        [
            "date",
            "year",
            "event_name",
            "fight_id",
            "outcome_label",
            "outcome_fighter_id",
            "model_pick",
            "model_probability",
            "implied_probability",
            "edge",
            "confidence_score",
            "american_odds",
            "side_type",
            "won",
            "flat_profit",
        ],
    )
    if "model_probability" in bets.columns:
        bets.sort_values("model_probability", ascending=False)[top_columns].head(100).to_csv(
            output_dir / "top_100_model_probability_bets.csv",
            index=False,
        )
    if "edge" in bets.columns:
        bets.sort_values("edge", ascending=False)[top_columns].head(100).to_csv(
            output_dir / "top_100_edge_bets.csv",
            index=False,
        )

    train_years = bets[bets["year"].le(2022)]
    holdout_years = bets[bets["year"].ge(2023)]
    split_summary = pd.concat(
        [
            summarize_group(train_years.assign(split="train_years_2022_and_prior"), ["split"]),
            summarize_group(holdout_years.assign(split="holdout_years_2023_plus"), ["split"]),
        ],
        ignore_index=True,
    )
    split_summary.to_csv(output_dir / "train_vs_holdout_summary.csv", index=False)

    manifest = {
        "audit_id": audit_id,
        "created_at_utc": run_time.isoformat(),
        "backtest_dir": str(backtest_dir),
        "input_path": str(backtest_dir / "backtest_bets.parquet"),
        "rows": int(len(bets)),
        "columns": list(bets.columns),
        "output_dir": str(output_dir),
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("=" * 80)
    print("BACKTEST BET PARQUET AUDIT")
    print("=" * 80)
    print(f"Audit ID    : {audit_id}")
    print(f"Backtest dir: {backtest_dir}")
    print(f"Rows        : {len(bets)}")
    print(f"Output dir  : {output_dir}")

    for title, key in [
        ("YEAR SUMMARY", "year_summary"),
        ("SIDE SUMMARY", "side_summary"),
        ("TRAIN VS HOLDOUT", "train_vs_holdout_summary"),
    ]:
        print("\n" + title)
        if key == "train_vs_holdout_summary":
            print(split_summary.to_string(index=False))
        elif key in summaries:
            print(summaries[key].to_string(index=False))

    if "year_side_summary" in summaries:
        print("\nYEAR + SIDE SUMMARY")
        print(summaries["year_side_summary"].to_string(index=False))

    if "year_confidence_bucket_summary" in summaries:
        print("\nYEAR + CONFIDENCE BUCKET SUMMARY")
        print(summaries["year_confidence_bucket_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
