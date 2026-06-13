# ============================================================
# pipeline/market/run_historical_odds_csv_inspect.py
# ============================================================

"""Inspect the root historical odds CSV before building a parquet importer.

This diagnostic is intentionally read-only. It prints the CSV size, row count,
headers, inferred dtypes, null counts, and a small preview so we can design the
historical moneyline odds parquet mapper from the actual file schema.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.common.paths import AUDITS_DIR, ensure_data_dirs

DEFAULT_INPUT_PATH = Path("ufc-master w odds.csv")
DEFAULT_OUTPUT_PATH = AUDITS_DIR / "ufc_historical_odds_csv_inspect.json"
DEFAULT_PREVIEW_OUTPUT_PATH = AUDITS_DIR / "ufc_historical_odds_csv_preview.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect historical odds CSV schema.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--preview-output-path", default=str(DEFAULT_PREVIEW_OUTPUT_PATH))
    parser.add_argument("--preview-rows", type=int, default=25)
    return parser.parse_args()


def _safe_read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    # utf-8-sig handles files exported from Excel/Sheets with BOMs.
    return pd.read_csv(path, nrows=nrows, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    ensure_data_dirs()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    preview_output_path = Path(args.preview_output_path)

    print("=" * 80)
    print("HISTORICAL ODDS CSV INSPECT")
    print("=" * 80)
    print("Input path:", input_path)
    print("Output path:", output_path)
    print("Preview output path:", preview_output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Historical odds CSV not found: {input_path}")

    file_size_bytes = input_path.stat().st_size
    header_df = _safe_read_csv(input_path, nrows=0)
    preview_df = _safe_read_csv(input_path, nrows=max(int(args.preview_rows), 1))

    # Full read is useful for row count/dtypes/nulls. This file should be manageable,
    # but the diagnostic still keeps output compact.
    full_df = _safe_read_csv(input_path)

    columns = list(full_df.columns)
    dtype_map = {column: str(dtype) for column, dtype in full_df.dtypes.items()}
    null_counts = {column: int(full_df[column].isna().sum()) for column in columns}
    non_null_counts = {column: int(full_df[column].notna().sum()) for column in columns}

    odds_like_columns = [
        column for column in columns
        if "odd" in column.lower() or column.lower() in {"r_odds", "b_odds", "red_odds", "blue_odds"}
    ]
    fighter_like_columns = [
        column for column in columns
        if "fighter" in column.lower() or column.lower() in {"r_name", "b_name", "red_fighter", "blue_fighter"}
    ]
    id_like_columns = [column for column in columns if column.lower().endswith("_id") or column.lower() == "fight_id"]
    date_like_columns = [column for column in columns if "date" in column.lower()]

    payload = {
        "inspect_run_id": datetime.now(timezone.utc).strftime("historical_odds_csv_inspect_%Y%m%d_%H%M%S"),
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "file_size_bytes": int(file_size_bytes),
        "row_count": int(len(full_df)),
        "column_count": int(len(columns)),
        "columns": columns,
        "dtypes": dtype_map,
        "null_counts": null_counts,
        "non_null_counts": non_null_counts,
        "odds_like_columns": odds_like_columns,
        "fighter_like_columns": fighter_like_columns,
        "id_like_columns": id_like_columns,
        "date_like_columns": date_like_columns,
        "preview_rows": preview_df.head(int(args.preview_rows)).to_dict(orient="records"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    preview_df.to_parquet(preview_output_path, index=False)

    print()
    print("========== HISTORICAL ODDS CSV SUMMARY ==========")
    print("File size bytes:", file_size_bytes)
    print("Rows:", len(full_df))
    print("Columns:", len(columns))
    print("Headers:")
    for idx, column in enumerate(columns, start=1):
        print(f"  {idx:03d}. {column}")
    print("Odds-like columns:", odds_like_columns)
    print("Fighter-like columns:", fighter_like_columns)
    print("ID-like columns:", id_like_columns)
    print("Date-like columns:", date_like_columns)
    print()
    print("Preview:")
    print(preview_df.head(int(args.preview_rows)).to_string(index=False))
    print()
    print("Audit saved:", output_path)
    print("Preview saved:", preview_output_path)


if __name__ == "__main__":
    main()
