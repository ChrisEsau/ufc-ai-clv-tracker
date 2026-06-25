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

def _stabilize_catalog_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid parquet write failures from mixed provider dtypes."""

    out = df.copy()
    string_columns = [
        "snapshot_run_id",
        "snapshot_timestamp",
        "source",
        "bookmaker",
        "provider_event_id",
        "event_name",
        "event_start_timestamp",
        "provider_subcategory_id",
        "provider_subcategory_name",
        "provider_market_id",
        "provider_market_name",
        "provider_market_type_id",
        "provider_market_type_name",
        "provider_selection_id",
        "provider_selection_name",
        "market_family",
        "market_key",
        "outcome_type",
        "outcome_key",
        "side",
        "fighter_name",
        "fighter_provider_id",
        "condition_key",
        "method_key",
        "raw_payload_path",
        "request_url",
    ]
    numeric_columns = [
        "line",
        "american_odds",
        "decimal_odds",
        "true_odds",
        "implied_probability",
        "round_number",
    ]
    bool_columns = [
        "is_conditional_no_action",
        "is_parlay",
        "is_boost",
        "is_promo",
    ]

    for column in string_columns:
        if column in out.columns:
            out[column] = out[column].astype("string")

    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in bool_columns:
        if column in out.columns:
            out[column] = out[column].astype("boolean")

    return out


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
    out = _stabilize_catalog_dtypes(out)

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
