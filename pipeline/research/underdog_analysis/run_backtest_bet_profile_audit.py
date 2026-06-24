from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_BACKTEST_DIR = "data/model_lab/backtests/moneyline_xgboost_v11_moneyline_full_20260623_220808"
DEFAULT_OUTPUT_ROOT = "data/research/backtest_bet_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Model Lab backtest bet parquet outputs.")
    parser.add_argument("--backtest-dir", default=DEFAULT_BACKTEST_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--validation-year", type=int, default=2023)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def read_backtest_table(backtest_dir: Path, names: list[str]) -> pd.DataFrame:
    for name in names:
        path = backtest_dir / name
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
    raise FileNotFoundError(f"None of these files exist in {backtest_dir}: {names}")


def prep(df: pd.DataFrame, train_end_year: int, validation_year: int) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["year"] = out["date"].dt.year
    if "won" in out.columns:
        out["won"] = pd.to_numeric(out["won"], errors="coerce")
    for col in ["model_probability", "implied_probability", "edge", "confidence_score", "american_odds", "flat_profit", "flat_stake", "kelly_profit", "kelly_stake"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "american_odds" in out.columns:
        out["side_type"] = out["american_odds"].apply(lambda x: "Favorite" if pd.notna(x) and x < 0 else "Underdog")
    else:
        out["side_type"] = "Unknown"
    if "year" in out.columns:
        out["era"] = "Holdout"
        out.loc[out["year"].le(train_end_year), "era"] = "Train/In-Sample"
        out.loc[out["year"].eq(validation_year), "era"] = "Validation"
    if "model_probability" in out.columns:
        out["model_probability_bucket"] = pd.cut(out["model_probability"], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1], labels=["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"], include_lowest=True).astype(str)
    if "edge" in out.columns:
        out["edge_bucket"] = pd.cut(out["edge"], bins=[-10,0,.05,.10,.15,.20,10], labels=["<=0%","0-5%","5-10%","10-15%","15-20%","20%+"], include_lowest=True).astype(str)
    return out


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or any(c not in df.columns for c in group_cols):
        return pd.DataFrame()
    agg = {"rows": ("won", "size"), "wins": ("won", "sum"), "win_rate": ("won", "mean")}
    for col in ["model_probability", "implied_probability", "edge", "confidence_score", "american_odds"]:
        if col in df.columns:
            agg[f"avg_{col}"] = (col, "mean")
    for col in ["flat_profit", "flat_stake", "kelly_profit", "kelly_stake"]:
        if col in df.columns:
            agg[col] = (col, "sum")
    out = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    if {"flat_profit", "flat_stake"}.issubset(out.columns):
        out["flat_roi"] = out["flat_profit"] / out["flat_stake"].replace({0: pd.NA})
    if {"kelly_profit", "kelly_stake"}.issubset(out.columns):
        out["kelly_roi"] = out["kelly_profit"] / out["kelly_stake"].replace({0: pd.NA})
    return out


def top_rows(df: pd.DataFrame, sort_col: str, top_n: int) -> pd.DataFrame:
    if sort_col not in df.columns:
        return pd.DataFrame()
    cols = ["date", "year", "fight_id", "event_name", "outcome_label", "fighter_name", "side_type", "american_odds", "implied_probability", "model_probability", "confidence_score", "edge", "won", "flat_profit"]
    cols = [c for c in cols if c in df.columns]
    return df.sort_values(sort_col, ascending=False)[cols].head(top_n)


def write_csv(table: pd.DataFrame, path: Path) -> None:
    if not table.empty:
        table.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir)
    run_time = datetime.now(timezone.utc)
    run_id = f"{backtest_dir.name}_profile_{run_time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    bets = read_backtest_table(backtest_dir, ["backtest_bets.parquet", "backtest_bets.csv", "backtest_results.parquet", "backtest_results.csv"])
    bets = prep(bets, args.train_end_year, args.validation_year)

    tables = {
        "year_summary": summarize(bets, ["year"]),
        "year_side_summary": summarize(bets, ["year", "side_type"]),
        "era_summary": summarize(bets, ["era"]),
        "era_side_summary": summarize(bets, ["era", "side_type"]),
        "model_probability_bucket_summary": summarize(bets, ["model_probability_bucket"]),
        "year_probability_bucket_summary": summarize(bets, ["year", "model_probability_bucket"]),
        "edge_bucket_summary": summarize(bets, ["edge_bucket"]),
        "year_edge_bucket_summary": summarize(bets, ["year", "edge_bucket"]),
        "top_model_probability_rows": top_rows(bets, "model_probability", args.top_n),
        "top_edge_rows": top_rows(bets, "edge", args.top_n),
        "top_confidence_rows": top_rows(bets, "confidence_score", args.top_n),
    }

    joined_path = backtest_dir / "joined_model_market_rows.parquet"
    if joined_path.exists():
        joined = prep(pd.read_parquet(joined_path), args.train_end_year, args.validation_year)
        tables["all_rows_year_summary"] = summarize(joined, ["year"])
        tables["all_rows_year_side_summary"] = summarize(joined, ["year", "side_type"])
        tables["all_rows_probability_bucket_summary"] = summarize(joined, ["model_probability_bucket"])

    for name, table in tables.items():
        write_csv(table, output_dir / f"{name}.csv")

    summary = {
        "run_id": run_id,
        "created_at_utc": run_time.isoformat(),
        "backtest_dir": str(backtest_dir),
        "output_dir": str(output_dir),
        "bet_rows": int(len(bets)),
        "bet_wins": float(bets["won"].sum()) if "won" in bets.columns else None,
        "bet_win_rate": float(bets["won"].mean()) if "won" in bets.columns else None,
        "train_end_year": args.train_end_year,
        "validation_year": args.validation_year,
        "outputs": sorted(tables.keys()),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("BACKTEST BET PROFILE AUDIT")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    for name in ["era_summary", "year_summary", "year_side_summary"]:
        table = tables.get(name)
        if table is not None and not table.empty:
            print(f"\n{name.upper()}")
            print(table.to_string(index=False))


if __name__ == "__main__":
    main()
