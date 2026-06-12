# ============================================================
# pipeline/market/run_draftkings_matched_discovery.py
# ============================================================

"""Run DraftKings discovery only for live-card matched provider events.

This runner consumes the offline card-filter artifact and discovers markets only
for provider_event_id values where is_matched is true. It preserves the existing
Market V2 separation:

- no canonical normalization
- no market matching
- no betting outcomes
- no CLV calculations
- no dashboard writes

Storage policy:
- current snapshot only for DraftKings diagnostics
- raw-index rows are replaced/upserted by provider_event_id + provider_subcategory_id
- no historical odds accumulation for already-seen event/subcategory pairs

Scraping safety policy:
- no concurrency
- one request at a time
- configurable sleep between event/subcategory requests
- no proxy rotation, CAPTCHA handling, or ban-evasion logic
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import (
    DRAFTKINGS_EVENT_CARD_MATCH_PATH,
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
from pipeline.market.run_draftkings_discovery import (
    DEFAULT_REGISTRY_PATH,
    _load_registry,
    _registry_subcategories,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover DraftKings markets for matched live-card event IDs only."
    )
    parser.add_argument(
        "--card-match-path",
        default=str(DRAFTKINGS_EVENT_CARD_MATCH_PATH),
        help="DraftKings event/live-card match artifact path.",
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
        help="Output parquet path for latest matched-event diagnostics.",
    )
    parser.add_argument(
        "--raw-index-path",
        default=str(DRAFTKINGS_RAW_INDEX_PATH),
        help="Output parquet path for latest raw snapshot index rows.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Delay between DraftKings requests. Applies between event/subcategory calls.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap for testing. Runs the first N matched provider events only.",
    )
    return parser.parse_args()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)


def _matched_event_ids(card_match_df: pd.DataFrame, max_events: int | None = None) -> list[str]:
    """Return unique matched DraftKings provider event IDs in artifact order."""

    required_columns = {"provider_event_id", "is_matched"}
    missing = sorted(required_columns - set(card_match_df.columns))
    if missing:
        raise ValueError(f"Card-match artifact missing required columns: {missing}")

    matched = card_match_df[card_match_df["is_matched"].fillna(False)].copy()
    if matched.empty:
        raise RuntimeError("No matched DraftKings provider events found in card-match artifact.")

    event_ids = []
    seen = set()
    for value in matched["provider_event_id"]:
        if pd.isna(value):
            continue
        event_id = str(value).strip()
        if event_id and event_id not in seen:
            event_ids.append(event_id)
            seen.add(event_id)

    if max_events is not None:
        event_ids = event_ids[: int(max_events)]

    if not event_ids:
        raise RuntimeError("No valid provider_event_id values found in matched card artifact.")

    return event_ids


def _replace_raw_index_rows(
    *,
    raw_index_path: Path,
    latest_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Replace latest raw-index rows by provider_event_id + provider_subcategory_id.

    This preserves raw-index rows for unrelated events, but prevents historical
    accumulation for event/subcategory pairs that were refreshed in this run.
    """

    latest = pd.DataFrame(latest_rows)
    if latest.empty:
        return latest

    key_columns = ["provider_event_id", "provider_subcategory_id"]
    for column in key_columns:
        if column not in latest.columns:
            latest[column] = pd.NA

    latest_keys = set(
        tuple(row)
        for row in latest[key_columns]
        .astype("string")
        .fillna("")
        .itertuples(index=False, name=None)
    )

    if raw_index_path.exists():
        existing = pd.read_parquet(raw_index_path)
        for column in key_columns:
            if column not in existing.columns:
                existing[column] = pd.NA

        existing_keys = existing[key_columns].astype("string").fillna("")
        keep_mask = [tuple(row) not in latest_keys for row in existing_keys.itertuples(index=False, name=None)]
        existing = existing.loc[keep_mask].copy()
        out = pd.concat([existing, latest], ignore_index=True)
    else:
        out = latest

    return out


def run_matched_discovery(
    *,
    card_match_path: Path = DRAFTKINGS_EVENT_CARD_MATCH_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    selected_subcategory_ids: list[str] | None = None,
    diagnostic_path: Path = DRAFTKINGS_MARKET_DIAGNOSTIC_PATH,
    raw_index_path: Path = DRAFTKINGS_RAW_INDEX_PATH,
    sleep_seconds: float = 3.0,
    max_events: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run safe matched-event DraftKings discovery and write latest artifacts."""

    card_match_df = _load_required_parquet(card_match_path, "DraftKings card-match artifact")
    event_ids = _matched_event_ids(card_match_df, max_events=max_events)

    registry = _load_registry(registry_path)
    subcategories = _registry_subcategories(registry, selected_ids=selected_subcategory_ids)
    if not subcategories:
        raise RuntimeError("No DraftKings subcategories selected from registry.")

    base_url = registry.get("base_url") or DEFAULT_BASE_URL
    snapshot_run_id, snapshot_timestamp = utc_snapshot()

    all_diagnostics: list[pd.DataFrame] = []
    index_rows: list[dict[str, Any]] = []
    total_requests = len(event_ids) * len(subcategories)
    request_number = 0

    for event_idx, event_id in enumerate(event_ids, start=1):
        print()
        print(f"Event [{event_idx}/{len(event_ids)}]: {event_id}")

        for subcategory_idx, subcategory in enumerate(subcategories, start=1):
            request_number += 1
            subcategory_id = str(subcategory.get("subcategory_id"))
            request_url = build_event_subcategory_markets_url(
                event_id,
                subcategory_id,
                base_url=base_url,
            )

            print(
                f"  Request [{request_number}/{total_requests}] "
                f"subcategory {subcategory_id}: {subcategory.get('name')}"
            )

            try:
                payload = fetch_public_json(request_url)
                snapshot = save_raw_snapshot(
                    payload,
                    raw_root=DRAFTKINGS_RAW_DIR,
                    snapshot_run_id=snapshot_run_id,
                    snapshot_timestamp=snapshot_timestamp,
                    event_id=event_id,
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
                print(f"    Success: {diagnostic_rows} diagnostic rows")

            except Exception as exc:
                status = "failed"
                error = str(exc)
                diagnostic_rows = 0
                raw_payload_path = None
                print("    FAILED")
                print(f"    {error}")

            index_rows.append(
                {
                    "snapshot_run_id": snapshot_run_id,
                    "snapshot_timestamp": snapshot_timestamp,
                    "source": "draftkings_public",
                    "bookmaker": "DraftKings",
                    "provider_event_id": event_id,
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

            if request_number < total_requests and sleep_seconds > 0:
                time.sleep(float(sleep_seconds))

    if all_diagnostics:
        diagnostic_df = pd.concat(all_diagnostics, ignore_index=True)
    else:
        diagnostic_df = pd.DataFrame()

    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_df.to_parquet(diagnostic_path, index=False)

    raw_index_path.parent.mkdir(parents=True, exist_ok=True)
    raw_index_df = _replace_raw_index_rows(
        raw_index_path=raw_index_path,
        latest_rows=index_rows,
    )
    raw_index_df.to_parquet(raw_index_path, index=False)

    return diagnostic_df, pd.DataFrame(index_rows)


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()

    print("=" * 80)
    print("DRAFTKINGS MATCHED MARKET DISCOVERY")
    print("=" * 80)
    print("Mode: matched live-card events only")
    print("No concurrency. No historical odds accumulation.")
    print("Card-match path:", args.card_match_path)
    print("Registry path:", args.registry_path)
    print("Diagnostic path:", args.diagnostic_path)
    print("Raw index path:", args.raw_index_path)
    print("Sleep seconds:", args.sleep_seconds)
    if args.max_events is not None:
        print("Max events:", args.max_events)

    diagnostic_df, index_df = run_matched_discovery(
        card_match_path=Path(args.card_match_path),
        registry_path=Path(args.registry_path),
        selected_subcategory_ids=args.subcategory_id,
        diagnostic_path=Path(args.diagnostic_path),
        raw_index_path=Path(args.raw_index_path),
        sleep_seconds=float(args.sleep_seconds),
        max_events=args.max_events,
    )

    print()
    print("========== DRAFTKINGS MATCHED DISCOVERY SUMMARY ==========")
    print("Requests attempted:", len(index_df))
    print("Successful requests:", int((index_df["status"] == "success").sum()) if not index_df.empty else 0)
    print("Failed requests:", int((index_df["status"] == "failed").sum()) if not index_df.empty else 0)
    print("Unique provider events:", index_df["provider_event_id"].nunique() if not index_df.empty else 0)
    print("Diagnostic rows:", len(diagnostic_df))
    print()
    print("Files saved:")
    print(args.diagnostic_path)
    print(args.raw_index_path)


if __name__ == "__main__":
    main()
