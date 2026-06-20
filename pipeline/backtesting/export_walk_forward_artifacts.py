"""Export walk-forward parquet artifacts to reviewable CSV/JSON files.

Run from repo root after a walk-forward backtest:

    python -m pipeline.backtesting.export_walk_forward_artifacts \
        --output-root data/model_lab/walk_forward_backtests

If --run-dir is omitted, the newest run directory under --output-root is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import MODEL_LAB_DIR

DEFAULT_OUTPUT_ROOT = MODEL_LAB_DIR / "walk_forward_backtests"


EXPORT_FILES = {
    "walk_forward_yearly_results.parquet": "walk_forward_yearly_results",
    "walk_forward_model_metrics.parquet": "walk_forward_model_metrics",
    "walk_forward_bucket_summary.parquet": "walk_forward_bucket_summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export walk-forward parquet artifacts to CSV/JSON.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-dir", default=None, help="Specific walk-forward run directory. Defaults to newest run.")
    parser.add_argument("--include-bets", action="store_true", help="Also export full bet-level table, which may be large.")
    return parser.parse_args()


def newest_run_dir(output_root: Path) -> Path:
    if not output_root.exists():
        raise FileNotFoundError(f"Walk-forward output root not found: {output_root}")
    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No walk-forward run directories found in: {output_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def write_json_records(df: pd.DataFrame, path: Path) -> None:
    records: list[dict[str, Any]] = json.loads(df.to_json(orient="records", date_format="iso"))
    path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def export_table(run_dir: Path, parquet_name: str, output_stem: str) -> None:
    parquet_path = run_dir / parquet_name
    if not parquet_path.exists():
        print(f"Skipping missing artifact: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    csv_path = run_dir / f"{output_stem}.csv"
    json_path = run_dir / f"{output_stem}.json"
    df.to_csv(csv_path, index=False)
    write_json_records(df, json_path)
    print(f"Exported {len(df):,} rows: {csv_path}")
    print(f"Exported {len(df):,} rows: {json_path}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(output_root)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Walk-forward run directory not found: {run_dir}")

    for parquet_name, output_stem in EXPORT_FILES.items():
        export_table(run_dir, parquet_name, output_stem)

    if args.include_bets:
        export_table(run_dir, "walk_forward_bets.parquet", "walk_forward_bets")


if __name__ == "__main__":
    main()
