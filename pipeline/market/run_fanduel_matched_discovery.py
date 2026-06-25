# ============================================================
# pipeline/market/run_fanduel_matched_discovery.py
# ============================================================

"""Run FanDuel discovery only for live-card matched provider events."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    FANDUEL_EVENT_CARD_MATCH_PATH,
    FANDUEL_MARKET_DIAGNOSTIC_PATH,
    FANDUEL_RAW_DIR,
    ensure_data_dirs,
)
from pipeline.market.providers.fanduel_public import (
    fetch_public_json,
    flatten_market_diagnostics,
    save_raw_snapshot,
)

EVENT_DETAIL_URL_TEMPLATE = (
    "https://api.sportsbook.fanduel.com/sbapi/event-page"
    "?_ak=FhMFpcPWXMeyZxOx&eventId={event_id}"
    "&useCombinedTouchdownsVirtualMarket=true&useQuickBets=true"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover FanDuel markets for matched live-card event IDs only.")
    parser.add_argument("--card-match-path", default=str(FANDUEL_EVENT_CARD_MATCH_PATH))
    parser.add_argument("--diagnostic-path", default=str(FANDUEL_MARKET_DIAGNOSTIC_PATH))
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-events", type=int, default=None)
    return parser.parse_args()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)


def _matched_event_ids(card_match_df: pd.DataFrame, max_events: int | None = None) -> list[str]:
    required = {"provider_event_id", "is_matched"}
    missing = sorted(required - set(card_match_df.columns))
    if missing:
        raise ValueError(f"Card-match artifact missing required columns: {missing}")

    matched = card_match_df[card_match_df["is_matched"].fillna(False)].copy()
    if matched.empty:
        raise RuntimeError("No matched FanDuel provider events found in card-match artifact.")

    event_ids: list[str] = []
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
        raise RuntimeError("No valid provider_event_id values found in matched FanDuel artifact.")

    return event_ids


def run_matched_discovery(
    *,
    card_match_path: Path = FANDUEL_EVENT_CARD_MATCH_PATH,
    diagnostic_path: Path = FANDUEL_MARKET_DIAGNOSTIC_PATH,
    sleep_seconds: float = 1.0,
    max_events: int | None = None,
) -> pd.DataFrame:
    card_match_df = _load_required_parquet(card_match_path, "FanDuel card-match artifact")
    event_ids = _matched_event_ids(card_match_df, max_events=max_events)

    all_diagnostics: list[pd.DataFrame] = []

    print("=" * 80)
    print("FANDUEL MATCHED MARKET DISCOVERY")
    print("=" * 80)
    print("Matched FanDuel events:", len(event_ids))
    print("Output diagnostic path:", diagnostic_path)

    for i, event_id in enumerate(event_ids, start=1):
        request_url = EVENT_DETAIL_URL_TEMPLATE.format(event_id=event_id)
        print(f"[{i}/{len(event_ids)}] Fetching FanDuel event {event_id}")

        try:
            payload = fetch_public_json(request_url)
            snapshot = save_raw_snapshot(
                payload,
                raw_root=FANDUEL_RAW_DIR,
                request_label=f"matched_event_{event_id}",
            )
            diagnostic_df = flatten_market_diagnostics(
                payload,
                request_url=request_url,
                snapshot=snapshot,
            )
            all_diagnostics.append(diagnostic_df)
            print(f"  rows: {len(diagnostic_df)}")
        except Exception as exc:
            print(f"  FAILED: {exc}")

        if i < len(event_ids) and sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    if not all_diagnostics:
        raise RuntimeError("FanDuel matched discovery returned zero diagnostic tables.")

    out = pd.concat(all_diagnostics, ignore_index=True)

    if out.empty:
        raise RuntimeError("FanDuel matched discovery returned zero diagnostic rows.")

    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(diagnostic_path, index=False)
    return out


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()

    df = run_matched_discovery(
        card_match_path=Path(args.card_match_path),
        diagnostic_path=Path(args.diagnostic_path),
        sleep_seconds=float(args.sleep_seconds),
        max_events=args.max_events,
    )

    print()
    print("========== FANDUEL MATCHED DISCOVERY SUMMARY ==========")
    print("Rows:", len(df))
    print("Events:", df["provider_event_id"].nunique())
    print()
    print("Market families:")
    print(df["supported_market_family"].value_counts(dropna=False).to_string())
    print()
    print("Output:", args.diagnostic_path)


if __name__ == "__main__":
    main()
