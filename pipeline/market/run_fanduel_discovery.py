import argparse

from pipeline.common.paths import FANDUEL_MARKET_DIAGNOSTIC_PATH, FANDUEL_RAW_DIR, ensure_data_dirs
from pipeline.market.providers.fanduel_public import DEFAULT_URL, fetch_public_json, flatten_market_diagnostics, save_raw_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="FanDuel public MMA market discovery spike.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap for diagnostics.")
    args = parser.parse_args()

    ensure_data_dirs()
    payload = fetch_public_json(DEFAULT_URL)
    snapshot = save_raw_snapshot(payload, raw_root=FANDUEL_RAW_DIR, request_label="market_discovery")
    df = flatten_market_diagnostics(payload, request_url=DEFAULT_URL, snapshot=snapshot)

    if args.max_events is not None and not df.empty:
        event_ids = df["provider_event_id"].dropna().astype(str).drop_duplicates().head(args.max_events).tolist()
        df = df[df["provider_event_id"].astype(str).isin(event_ids)].copy()

    if df.empty:
        raise RuntimeError("FanDuel market discovery returned zero diagnostic rows.")

    FANDUEL_MARKET_DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FANDUEL_MARKET_DIAGNOSTIC_PATH, index=False)

    print("FANDUEL MARKET DISCOVERY")
    print("Rows:", len(df))
    print("Events:", df["provider_event_id"].nunique())
    print("Market families:")
    print(df["supported_market_family"].value_counts(dropna=False).to_string())
    print("Output:", FANDUEL_MARKET_DIAGNOSTIC_PATH)


if __name__ == "__main__":
    main()
