from datetime import datetime, timezone

import pandas as pd

from scrapers.ufcstats_fights import scrape_event_fights

from pipeline.common.paths import (
    MISSING_EVENTS_PATH,
    STAGED_FIGHT_ROWS_PATH,
    FIGHT_SCRAPE_AUDIT_PATH,
)

STAGED_OUTPUT = STAGED_FIGHT_ROWS_PATH
AUDIT_OUTPUT = FIGHT_SCRAPE_AUDIT_PATH


def extract_id_from_url(url):
    if pd.isna(url):
        return None

    return (
        str(url)
        .strip()
        .rstrip("/")
        .split("/")[-1]
    )


def filter_missing_events(missing_events, event_id=None, max_events=None):
    filtered = missing_events.copy()

    if event_id is not None:
        filtered = filtered[
            filtered["ufcstats_event_id"].astype(str) == str(event_id)
        ].copy()

    if max_events is not None:
        filtered = filtered.head(max_events)

    return filtered.reset_index(drop=True)


def run_fight_scrape(event_id=None, max_events=None):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_timestamp = datetime.now(timezone.utc).isoformat()

    missing_events = pd.read_parquet(MISSING_EVENTS_PATH)

    print("Missing events:", len(missing_events))

    missing_events = filter_missing_events(
        missing_events,
        event_id=event_id,
        max_events=max_events,
    )

    print("Events selected:", len(missing_events))

    all_fights = []
    audit_rows = []

    for idx, row in missing_events.iterrows():
        event_name = row.get("ufcstats_event_name")
        event_date = row.get("ufcstats_event_date")
        event_url = row.get("ufcstats_event_url")
        event_id_value = row.get("ufcstats_event_id")

        if pd.isna(event_id_value) or event_id_value is None:
            event_id_value = extract_id_from_url(event_url)

        print()
        print(f"[{idx + 1}/{len(missing_events)}] Scraping: {event_name}")

        try:
            fights = scrape_event_fights(
                event_url=event_url,
                event_name=event_name,
                event_date=event_date,
            )

            # UFCStats event pages include a blank summary row.
            # Keep only real fight-detail rows.
            fights = fights[
                fights["fight_url"].astype(str).str.contains(
                    "fight-details",
                    na=False,
                )
            ].copy()

            fights = fights.reset_index(drop=True)

            fights["event_url"] = event_url
            fights["event_id"] = event_id_value
            fights["event_name"] = event_name
            fights["event_date"] = event_date

            fights["fight_id"] = (
                fights["fight_url"]
                .astype(str)
                .str.rstrip("/")
                .str.split("/")
                .str[-1]
            )

            fights["run_id"] = run_id
            fights["run_timestamp"] = run_timestamp

            all_fights.append(fights)

            audit_rows.append(
                {
                    "run_id": run_id,
                    "run_timestamp": run_timestamp,
                    "event_name": event_name,
                    "event_date": event_date,
                    "event_url": event_url,
                    "event_id": event_id_value,
                    "status": "success",
                    "fight_count": len(fights),
                    "error": None,
                }
            )

            print(f"Success: {len(fights)} fights")

        except Exception as e:
            audit_rows.append(
                {
                    "run_id": run_id,
                    "run_timestamp": run_timestamp,
                    "event_name": event_name,
                    "event_date": event_date,
                    "event_url": event_url,
                    "event_id": event_id_value,
                    "status": "failed",
                    "fight_count": 0,
                    "error": str(e),
                }
            )

            print(f"FAILED: {event_name}")
            print(e)

    if len(all_fights) > 0:
        staged_df = pd.concat(all_fights, ignore_index=True)
    else:
        staged_df = pd.DataFrame()

    audit_df = pd.DataFrame(audit_rows)

    staged_df.to_parquet(STAGED_OUTPUT, index=False)
    audit_df.to_parquet(AUDIT_OUTPUT, index=False)

    print()
    print("========== SCRAPE SUMMARY ==========")
    print("Total staged fights:", len(staged_df))
    print("Events attempted:", len(audit_df))

    if not audit_df.empty:
        print("Successful events:", (audit_df["status"] == "success").sum())
        print("Failed events:", (audit_df["status"] == "failed").sum())

    if not staged_df.empty:
        print()
        print("========== STAGED COLUMNS ==========")
        print(list(staged_df.columns))

        print()
        print("========== EVENT ID SAMPLE ==========")
        print(
            staged_df[
                [
                    "event_name",
                    "event_id",
                    "fight_id",
                    "fight_url",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("Saved:", STAGED_OUTPUT)
    print("Saved:", AUDIT_OUTPUT)

    return staged_df, audit_df


if __name__ == "__main__":
    # Debug/default mode.
    # Set max_events=None for all missing events.
    run_fight_scrape(
        event_id=None,
        max_events=1,
    )