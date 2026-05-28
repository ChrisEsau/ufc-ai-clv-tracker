from datetime import datetime, timezone

import pandas as pd

from scrapers.ufcstats_events import scrape_completed_events
from scrapers.ufcstats_utils import normalize_event_name


BASE_PATH = "."

DATASET_EVENT_STATUS_PATH = f"{BASE_PATH}/ufc_dataset_event_status.parquet"

EVENT_CHECK_OUTPUT = f"{BASE_PATH}/ufc_ufcstats_event_check.parquet"
MISSING_EVENTS_OUTPUT = f"{BASE_PATH}/ufc_missing_events.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


local_events = pd.read_parquet(DATASET_EVENT_STATUS_PATH)

local_events["event_name_norm"] = local_events["event_name"].apply(
    normalize_event_name
)

local_events["event_date"] = pd.to_datetime(
    local_events["event_date"],
    errors="coerce",
).dt.date

print("Local events:", len(local_events))

ufcstats_events = scrape_completed_events(
    run_id=RUN_ID,
    run_timestamp=RUN_TIMESTAMP,
)

print("UFCStats completed events:", len(ufcstats_events))

comparison = ufcstats_events.merge(
    local_events[
        [
            "event_name",
            "event_date",
            "fight_count",
            "event_name_norm",
        ]
    ],
    how="left",
    left_on=[
        "ufcstats_event_name_norm",
        "ufcstats_event_date",
    ],
    right_on=[
        "event_name_norm",
        "event_date",
    ],
)

comparison["in_local_dataset"] = comparison["event_name"].notna()

comparison["status"] = comparison["in_local_dataset"].map(
    {
        True: "present",
        False: "missing",
    }
)

comparison = comparison[
    [
        "run_id",
        "run_timestamp",
        "ufcstats_event_name",
        "ufcstats_event_date",
        "ufcstats_event_url",
        "status",
        "in_local_dataset",
        "event_name",
        "event_date",
        "fight_count",
    ]
].sort_values(
    "ufcstats_event_date",
    ascending=False,
)

missing_events = comparison[
    comparison["status"] == "missing"
].copy()

comparison.to_parquet(
    EVENT_CHECK_OUTPUT,
    index=False,
)

missing_events.to_parquet(
    MISSING_EVENTS_OUTPUT,
    index=False,
)

print("Saved event check:", EVENT_CHECK_OUTPUT)
print("Saved missing events:", MISSING_EVENTS_OUTPUT)

print("========== UFCSTATS EVENT CHECK SUMMARY ==========")
print("UFCStats events:", len(ufcstats_events))
print("Local events:", len(local_events))
print("Missing events:", len(missing_events))

if len(missing_events) > 0:
    print(
        missing_events[
            [
                "ufcstats_event_name",
                "ufcstats_event_date",
            ]
        ].head(10)
    )