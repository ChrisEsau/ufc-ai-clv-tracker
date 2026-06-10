# ============================================================
# pipeline/market/run_draftkings_discovery.py
# ============================================================

"""Standalone DraftKings UFC market discovery runner.

This runner is intentionally isolated from Market V2 production outputs.
It captures raw DraftKings public JSON payloads and writes a diagnostic
market/selection table for schema discovery.

It does not write market_outcomes.parquet, betting_outcomes.parquet, CLV
artifacts, or dashboard artifacts.

Usage examples:
    # Registry-driven full event discovery
    python -m pipeline.market.run_draftkings_discovery --event-id 33525834

    # Single manually discovered URL
    python -m pipeline.market.run_draftkings_discovery --url "<public-json-url>"

    # Custom registry path
    python -m pipeline.market.run_draftkings_discovery \
        --event-id 33525834 \
        --registry-path configs/market/providers/draftkings_ufc_registry.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline.common.paths import (
    DRAFTKINGS_MARKET_DIAGNOSTIC_PATH,
    DRAFTKINGS_RAW_DIR,
    DRAFTKINGS_RAW_INDEX_PATH,
    ensure_data_dirs,
)
from pipeline.market.providers.draftkings_public import (
    DEFAULT_BASE_URL,
    build_event_subcategory_markets_url,
    fetch_public_json,
    flatten_market_diagnostics,
    save_raw_snapshot,
    utc_snapshot,
)

DEFAULT_REGISTRY_PATH = Path("configs/market/providers/draftkings_ufc_registry.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture DraftKings public UFC market payloads for raw discovery."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("DRAFTKINGS_DISCOVERY_URL"),
        help="Optional single public DraftKings JSON endpoint to fetch.",
    )
    parser.add_argument(
        "--event-id",
        default=os.getenv("DRAFTKINGS_EVENT_ID"),
        help="DraftKings event ID for registry-driven subcategory discovery.",
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="DraftKings UFC provider registry YAML path.",
    )
    parser.add_argument(
        "--subcategory-id",
        action="append",
        default=None,
        help="Optional subcategory ID filter. Repeat for multiple IDs.",
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


def _load_registry(path: Path) -> dict[str, Any]:
    """Load a DraftKings provider registry YAML file."""

    if not path.exists():
        raise FileNotFoundError(f"DraftKings registry not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    if not isinstance(registry, dict):
        raise ValueError(f"DraftKings registry must be a mapping: {path}")

    return registry


def _registry_subcategories(
    registry: dict[str, Any],
    selected_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return configured subcategory entries, optionally filtered by ID."""

    subcategories = registry.get("subcategories", [])
    if not isinstance(subcategories, list):
        raise ValueError("DraftKings registry field 'subcategories' must be a list.")

    selected = {str(x) for x in selected_ids or []}
    entries = []
    for item in subcategories:
        if not isinstance(item, dict):
            continue
        subcategory_id = str(item.get("subcategory_id", "")).strip()
        if not subcategory_id:
            continue
        if selected and subcategory_id not in selected:
            continue
        entries.append(item)

    return entries


def _append_raw_index(index_path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Append raw snapshot metadata rows to the discovery index."""

    latest = pd.DataFrame(rows)

    if index_path.exists():
        existing = pd.read_parquet(index_path)
        return pd.concat([existing, latest], ignore_index=True)

    return latest


def _run_single_url(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run legacy/manual single-URL discovery mode."""

    snapshot_run_id, snapshot_timestamp = utc_snapshot()
    payload = fetch_public_json(args.url)
    snapshot = save_raw_snapshot(
        payload,
        raw_root=DRAFTKINGS_RAW_DIR,
        snapshot_run_id=snapshot_run_id,
        snapshot_timestamp=snapshot_timestamp,
    )
    diagnostic_df = flatten_market_diagnostics(
        payload,
        snapshot=snapshot,
        request_url=args.url,
    )

    index_rows = [
        {
            "snapshot_run_id": snapshot.snapshot_run_id,
            "snapshot_timestamp": snapshot.snapshot_timestamp,
            "source": "draftkings_public",
            "bookmaker": "DraftKings",
            "provider_event_id": None,
            "provider_subcategory_id": None,
            "provider_subcategory_name": None,
            "registry_family": None,
            "request_url": args.url,
            "raw_payload_path": str(snapshot.raw_payload_path),
            "diagnostic_rows": int(len(diagnostic_df)),
            "status": "success",
            "error": None,
        }
    ]
    return diagnostic_df, index_rows


def _run_registry_discovery(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run event/subcategory registry-driven DraftKings discovery."""

    registry_path = Path(args.registry_path)
    registry = _load_registry(registry_path)
    subcategories = _registry_subcategories(registry, selected_ids=args.subcategory_id)

    if not args.event_id:
        raise RuntimeError("Registry-driven discovery requires --event-id or DRAFTKINGS_EVENT_ID.")
    if not subcategories:
        raise RuntimeError("No DraftKings subcategories selected from registry.")

    base_url = registry.get("base_url") or DEFAULT_BASE_URL
    snapshot_run_id, snapshot_timestamp = utc_snapshot()

    all_diagnostics: list[pd.DataFrame] = []
    index_rows: list[dict[str, Any]] = []

    for idx, subcategory in enumerate(subcategories, start=1):
        subcategory_id = str(subcategory.get("subcategory_id"))
        request_url = build_event_subcategory_markets_url(
            str(args.event_id),
            subcategory_id,
            base_url=base_url,
        )

        print()
        print(f"[{idx}/{len(subcategories)}] Fetching subcategory {subcategory_id}: {subcategory.get('name')}")

        try:
            payload = fetch_public_json(request_url)
            snapshot = save_raw_snapshot(
                payload,
                raw_root=DRAFTKINGS_RAW_DIR,
                snapshot_run_id=snapshot_run_id,
                snapshot_timestamp=snapshot_timestamp,
                event_id=str(args.event_id),
                subcategory_id=subcategory_id,
            )
            diagnostic_df = flatten_market_diagnostics(
                payload,
                snapshot=snapshot,
                request_url=request_url,
                registry_entry=subcategory,
            )
            all_diagnostics.append(diagnostic_df)
            status = "success"
            error = None
            diagnostic_rows = len(diagnostic_df)
            raw_payload_path = str(snapshot.raw_payload_path)
            print(f"Success: {diagnostic_rows} diagnostic rows")

        except Exception as exc:
            status = "failed"
            error = str(exc)
            diagnostic_rows = 0
            raw_payload_path = None
            print("FAILED")
            print(error)

        index_rows.append(
            {
                "snapshot_run_id": snapshot_run_id,
                "snapshot_timestamp": snapshot_timestamp,
                "source": "draftkings_public",
                "bookmaker": "DraftKings",
                "provider_event_id": str(args.event_id),
                "provider_subcategory_id": subcategory_id,
                "provider_subcategory_name": subcategory.get("name"),
                "registry_family": subcategory.get("family"),
                "request_url": request_url,
                "raw_payload_path": raw_payload_path,
                "diagnostic_rows": int(diagnostic_rows),
                "status": status,
                "error": error,
            }
        )

    if all_diagnostics:
        diagnostic_df = pd.concat(all_diagnostics, ignore_index=True)
    else:
        diagnostic_df = pd.DataFrame()

    return diagnostic_df, index_rows


def main() -> None:
    args = _parse_args()

    if not args.url and not args.event_id:
        raise RuntimeError(
            "DraftKings discovery requires either --url or --event-id. "
            "Use --event-id for registry-driven UFC discovery."
        )

    ensure_data_dirs()

    print("=" * 80)
    print("DRAFTKINGS MARKET DISCOVERY")
    print("=" * 80)
    print("Mode: raw snapshot + diagnostic only")
    print("Production Market V2 outputs: untouched")

    if args.url:
        print("Discovery mode: single URL")
        diagnostic_df, index_rows = _run_single_url(args)
    else:
        print("Discovery mode: registry-driven event/subcategory loop")
        print("Event ID:", args.event_id)
        print("Registry:", args.registry_path)
        diagnostic_df, index_rows = _run_registry_discovery(args)

    diagnostic_path = Path(args.diagnostic_path)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_df.to_parquet(diagnostic_path, index=False)

    raw_index_path = Path(args.raw_index_path)
    raw_index_path.parent.mkdir(parents=True, exist_ok=True)
    raw_index_df = _append_raw_index(raw_index_path, index_rows)
    raw_index_df.to_parquet(raw_index_path, index=False)

    print()
    print("========== DRAFTKINGS DISCOVERY SUMMARY ==========")
    if index_rows:
        print("Snapshot run ID:", index_rows[0].get("snapshot_run_id"))
        print("Snapshot timestamp:", index_rows[0].get("snapshot_timestamp"))
    print("Requests attempted:", len(index_rows))
    print("Successful requests:", sum(1 for row in index_rows if row.get("status") == "success"))
    print("Failed requests:", sum(1 for row in index_rows if row.get("status") == "failed"))
    print("Diagnostic rows:", len(diagnostic_df))

    if not diagnostic_df.empty:
        print("Registry families:")
        print(diagnostic_df["registry_family"].value_counts(dropna=False).to_dict())
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
