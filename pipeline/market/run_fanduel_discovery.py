import argparse
import time

import pandas as pd

from pipeline.common.paths import (
    FANDUEL_EVENT_INDEX_PATH,
    FANDUEL_MARKET_DIAGNOSTIC_PATH,
    FANDUEL_RAW_DIR,
    ensure_data_dirs,
)
from pipeline.market.providers.fanduel_public import (
    DEFAULT_URL,
    build_event_index,
    fetch_public_json,
    flatten_market_diagnostics,
    save_raw_snapshot,
)

EVENT_DETAIL_URL_TEMPLATE = (
    "https://api.sportsbook.fanduel.com/sbapi/event-page"
    "?_ak=FhMFpcPWXMeyZxOx&eventId={event_id}"
    "&useCombinedTouchdownsVirtualMarket=true&useQuickBets=true"
)

def _load_or_build_event_index() -> pd.DataFrame:
    if FANDUEL_EVENT_INDEX_PATH.exists():
        return pd.read_parquet(FANDUEL_EVENT_INDEX_PATH)

    payload = fetch_public_json(DEFAULT_URL)
    snapshot = save_raw_snapshot(payload, raw_root=FANDUEL_RAW_DIR, request_label="event_index")
    event_index_df = build_event_index(payload, request_url=DEFAULT_URL, snapshot=snapshot)
    FANDUEL_EVENT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    event_index_df.to_parquet(FANDUEL_EVENT_INDEX_PATH, index=False)
    return event_index_df


def main() -> None:
    parser = argparse.ArgumentParser(description="FanDuel event-level MMA market discovery.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap for diagnostics.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay between event requests.")
    parser.add_argument("--event-id", type=str, default=None, help="Optional single FanDuel event id.")
    args = parser.parse_args()

    ensure_data_dirs()
    event_index_df = _load_or_build_event_index()

    event_ids = (
        event_index_df["provider_event_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if args.event_id:
        event_ids = [args.event_id]
    elif args.max_events is not None:
        event_ids = event_ids[: args.max_events]

    all_dfs = []

    print("FANDUEL EVENT-LEVEL MARKET DISCOVERY")
    print("Events to fetch:", len(event_ids))

    for i, event_id in enumerate(event_ids, start=1):
        request_url = EVENT_DETAIL_URL_TEMPLATE.format(event_id=event_id)
        print(f"[{i}/{len(event_ids)}] Fetching FanDuel event {event_id}")

        try:
            payload = fetch_public_json(request_url)
            snapshot = save_raw_snapshot(
                payload,
                raw_root=FANDUEL_RAW_DIR,
                request_label=f"event_{event_id}",
            )
            df = flatten_market_diagnostics(
                payload,
                request_url=request_url,
                snapshot=snapshot,
            )
            all_dfs.append(df)
            print(f"  event-page rows: {len(df)}")

        except Exception as exc:
            print(f"  FAILED: {exc}")

        if i < len(event_ids) and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if not all_dfs:
        raise RuntimeError("FanDuel market discovery returned zero diagnostic tables.")

    diagnostic_df = pd.concat(all_dfs, ignore_index=True)

    if diagnostic_df.empty:
        raise RuntimeError("FanDuel market discovery returned zero diagnostic rows.")

    FANDUEL_MARKET_DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_df.to_parquet(FANDUEL_MARKET_DIAGNOSTIC_PATH, index=False)

    print()
    print("========== FANDUEL MARKET DISCOVERY SUMMARY ==========")
    print("Rows:", len(diagnostic_df))
    print("Events:", diagnostic_df["provider_event_id"].nunique())
    print()
    print("Market types:")
    print(diagnostic_df["provider_market_type_name"].value_counts(dropna=False).head(50).to_string())
    print()
    print("Market families:")
    print(diagnostic_df["supported_market_family"].value_counts(dropna=False).head(50).to_string())
    print()
    print("Output:", FANDUEL_MARKET_DIAGNOSTIC_PATH)


if __name__ == "__main__":
    main()
