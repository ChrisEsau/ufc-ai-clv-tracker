from __future__ import annotations

import pandas as pd

from scrapers.ufcstats_events import scrape_completed_events

from pipeline.common.paths import (
    MASTER_PATH,
    UFCSTATS_EVENT_CHECK_PATH,
    MISSING_EVENTS_PATH,
)

EVENT_CHECK_OUTPUT = UFCSTATS_EVENT_CHECK_PATH
MISSING_EVENTS_OUTPUT = MISSING_EVENTS_PATH


def extract_id_from_url(url):
    if pd.isna(url):
        return None

    return (
        str(url)
        .strip()
        .rstrip("/")
        .split("/")[-1]
    )


def run_ufcstats_event_check() -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_parquet(MASTER_PATH)

    print("\n========== EVENT ID HEALTH ==========")

    if "event_id" not in master.columns:
        raise ValueError("ufc_master.parquet does not contain event_id column.")

    print(
        "Null event_ids:",
        master["event_id"].isna().sum(),
    )

    print(
        "Unique event_ids:",
        master["event_id"]
        .dropna()
        .nunique(),
    )

    local_event_ids = set(
        master["event_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    print("\n========== LOCAL EVENT ID SAMPLE ==========")

    sample_ids = (
        master["event_id"]
        .dropna()
        .astype(str)
        .unique()[:10]
    )

    for x in sample_ids:
        print(x)

    print("Local events:", len(local_event_ids))

    ufcstats_events = scrape_completed_events()

    if ufcstats_events.empty:
        raise ValueError("No UFCStats completed events scraped.")

    if "ufcstats_event_url" not in ufcstats_events.columns:
        raise ValueError("UFCStats events missing ufcstats_event_url column.")

    ufcstats_events["ufcstats_event_id"] = (
        ufcstats_events["ufcstats_event_url"]
        .apply(extract_id_from_url)
    )

    ufcstats_events["exists_in_master"] = (
        ufcstats_events["ufcstats_event_id"]
        .astype(str)
        .isin(local_event_ids)
    )

    missing_events = ufcstats_events[
        ~ufcstats_events["exists_in_master"]
    ].copy()

    print("\n========== FIRST MISSING EVENT ==========\n")

    if len(missing_events):
        first_event_id = (
            missing_events.iloc[0]["ufcstats_event_id"]
        )

        print("Missing event id:", first_event_id)

        print(
            "Exists in local ids:",
            first_event_id in local_event_ids,
        )

    ufcstats_events.to_parquet(
        EVENT_CHECK_OUTPUT,
        index=False,
    )

    missing_events.to_parquet(
        MISSING_EVENTS_OUTPUT,
        index=False,
    )

    print("UFCStats completed events:", len(ufcstats_events))
    print("Saved event check:", EVENT_CHECK_OUTPUT)
    print("Saved missing events:", MISSING_EVENTS_OUTPUT)

    print("========== UFCSTATS EVENT CHECK SUMMARY ==========")
    print("UFCStats events:", len(ufcstats_events))
    print("Local events:", len(local_event_ids))
    print("Missing events:", len(missing_events))

    preview_cols = [
        c for c in [
            "ufcstats_event_name",
            "ufcstats_event_date",
            "ufcstats_event_id",
            "ufcstats_event_url",
        ]
        if c in missing_events.columns
    ]

    print(
        missing_events[preview_cols]
        .head(10)
        .to_string(index=False)
    )

    return ufcstats_events, missing_events


def main() -> None:
    run_ufcstats_event_check()


if __name__ == "__main__":
    main()
