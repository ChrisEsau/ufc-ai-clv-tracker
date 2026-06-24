from pathlib import Path

from pipeline.common.paths import FANDUEL_EVENT_INDEX_PATH, FANDUEL_RAW_DIR, ensure_data_dirs
from pipeline.market.providers.fanduel_public import DEFAULT_URL, build_event_index, fetch_public_json, save_raw_snapshot


def main() -> None:
    ensure_data_dirs()
    payload = fetch_public_json(DEFAULT_URL)
    snapshot = save_raw_snapshot(payload, raw_root=FANDUEL_RAW_DIR, request_label="event_index")
    df = build_event_index(payload, request_url=DEFAULT_URL, snapshot=snapshot)

    if df.empty:
        raise RuntimeError("FanDuel event index returned zero rows.")

    FANDUEL_EVENT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FANDUEL_EVENT_INDEX_PATH, index=False)

    print("FANDUEL EVENT INDEX")
    print("Rows:", len(df))
    print("Output:", FANDUEL_EVENT_INDEX_PATH)
    print(df[["provider_event_id", "provider_event_name", "provider_start_time"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
