# ============================================================
# pipeline/market/run_draftkings_discovery.py
# ============================================================

"""Standalone DraftKings UFC market discovery runner.

This runner is intentionally isolated from Market V2 production outputs.
It captures a raw DraftKings public JSON payload and writes a diagnostic
market/selection table for schema discovery.

It does not write market_outcomes.parquet, betting_outcomes.parquet, CLV
artifacts, or dashboard artifacts.

Usage:
    python -m pipeline.market.run_draftkings_discovery --url "<public-json-url>"

or set:
    DRAFTKINGS_DISCOVERY_URL="<public-json-url>"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    DRAFTKINGS_MARKET_DIAGNOSTIC_PATH,
    DRAFTKINGS_RAW_DIR,
    DRAFTKINGS_RAW_INDEX_PATH,
    ensure_data_dirs,
)
from pipeline.market.providers.draftkings_public import (
    fetch_public_json,
    flatten_market_diagnostics,
    save_raw_snapshot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a DraftKings public UFC market payload for raw discovery."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("DRAFTKINGS_DISCOVERY_URL"),
        help="Public DraftKings JSON endpoint to fetch. Can also be set with DRAFTKINGS_DISCOVERY_URL.",
    )
    parser.add_argument(
        "--diagnostic-path",
        default=str(DRAFTKINGS_MARKET_DIAGNOSTIC_PATH),
        help="Output parquet path for the flattened diagnostic table.",
    )
    parser.add_argument(
        "--raw-index-path",
        default=str(DRAFTKINGS_RAW_INDEX_PATH),
        help="Output parquet path for the raw snapshot index.",
    )
    return parser.parse_args()


def _append_raw_index(index_path: Path, snapshot, diagnostic_rows: int) -> pd.DataFrame:
    """Append one raw snapshot metadata row to the discovery index."""

    latest = pd.DataFrame(
        [
            {
                "snapshot_run_id": snapshot.snapshot_run_id,
                "snapshot_timestamp": snapshot.snapshot_timestamp,
                "source": "draftkings_public",
                "bookmaker": "DraftKings",
                "raw_payload_path": str(snapshot.raw_payload_path),
                "diagnostic_rows": int(diagnostic_rows),
            }
        ]
    )

    if index_path.exists():
        existing = pd.read_parquet(index_path)
        return pd.concat([existing, latest], ignore_index=True)

    return latest


def main() -> None:
    args = _parse_args()

    if not args.url:
        raise RuntimeError(
            "DraftKings discovery URL is required. Pass --url or set DRAFTKINGS_DISCOVERY_URL."
        )

    ensure_data_dirs()

    print("=" * 80)
    print("DRAFTKINGS MARKET DISCOVERY")
    print("=" * 80)
    print("Mode: raw snapshot + diagnostic only")
    print("Production Market V2 outputs: untouched")

    payload = fetch_public_json(args.url)
    snapshot = save_raw_snapshot(payload, raw_root=DRAFTKINGS_RAW_DIR)
    diagnostic_df = flatten_market_diagnostics(payload, snapshot=snapshot)

    diagnostic_path = Path(args.diagnostic_path)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_df.to_parquet(diagnostic_path, index=False)

    raw_index_path = Path(args.raw_index_path)
    raw_index_path.parent.mkdir(parents=True, exist_ok=True)
    raw_index_df = _append_raw_index(raw_index_path, snapshot, diagnostic_rows=len(diagnostic_df))
    raw_index_df.to_parquet(raw_index_path, index=False)

    print()
    print("========== DRAFTKINGS DISCOVERY SUMMARY ==========")
    print("Snapshot run ID:", snapshot.snapshot_run_id)
    print("Snapshot timestamp:", snapshot.snapshot_timestamp)
    print("Raw payload:", snapshot.raw_payload_path)
    print("Diagnostic rows:", len(diagnostic_df))

    if not diagnostic_df.empty:
        print("Supported market families:")
        print(diagnostic_df["supported_market_family"].value_counts(dropna=False).to_dict())
        print("Parlay rows:", int(diagnostic_df["is_parlay"].fillna(False).sum()))
        print("Boost rows:", int(diagnostic_df["is_boost"].fillna(False).sum()))
        print("Promo rows:", int(diagnostic_df["is_promo"].fillna(False).sum()))

    print()
    print("Files saved:")
    print(diagnostic_path)
    print(raw_index_path)


if __name__ == "__main__":
    main()
