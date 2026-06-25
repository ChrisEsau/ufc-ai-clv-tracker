# ============================================================
# pipeline/market/run_merge_market_catalogs.py
# ============================================================

"""Merge provider-specific market catalogs into canonical market catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    CANONICAL_MARKET_CATALOG_PATH,
    DRAFTKINGS_MARKET_CATALOG_PATH,
    FANDUEL_MARKET_CATALOG_PATH,
    ensure_data_dirs,
)
from pipeline.market.normalizers.canonical_market_schema import ensure_canonical_market_columns


DEFAULT_INPUTS = [
    DRAFTKINGS_MARKET_CATALOG_PATH,
    FANDUEL_MARKET_CATALOG_PATH,
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge provider-specific market catalogs.")
    parser.add_argument(
        "--input-path",
        action="append",
        default=None,
        help="Provider catalog path. Repeat for multiple. Defaults to DraftKings + FanDuel provider catalogs.",
    )
    parser.add_argument(
        "--output-path",
        default=str(CANONICAL_MARKET_CATALOG_PATH),
        help="Merged canonical market catalog output path.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow no available provider catalogs and write an empty canonical catalog.",
    )
    return parser.parse_args()


def merge_market_catalogs(
    *,
    input_paths: list[Path] | None = None,
    output_path: Path = CANONICAL_MARKET_CATALOG_PATH,
    allow_empty: bool = False,
) -> pd.DataFrame:
    paths = input_paths or DEFAULT_INPUTS

    frames = []
    used_paths = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"SKIP missing provider catalog: {path}")
            continue
        df = pd.read_parquet(path)
        if df.empty:
            print(f"SKIP empty provider catalog: {path}")
            continue
        frames.append(ensure_canonical_market_columns(df))
        used_paths.append(str(path))

    if frames:
        out = pd.concat(frames, ignore_index=True)
    elif allow_empty:
        out = ensure_canonical_market_columns(pd.DataFrame())
    else:
        raise RuntimeError("No provider market catalogs available to merge.")

    out = ensure_canonical_market_columns(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print("=" * 80)
    print("MERGE MARKET CATALOGS")
    print("=" * 80)
    print("Used inputs:", used_paths)
    print("Rows:", len(out))
    if not out.empty and "bookmaker" in out.columns:
        print("Rows by bookmaker:")
        print(out["bookmaker"].value_counts(dropna=False).to_string())
    print("Output:", output_path)

    return out


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()

    input_paths = [Path(p) for p in args.input_path] if args.input_path else None
    merge_market_catalogs(
        input_paths=input_paths,
        output_path=Path(args.output_path),
        allow_empty=bool(args.allow_empty),
    )


if __name__ == "__main__":
    main()
