# ============================================================
# run_ufcstats_event_check.py
# Detect completed UFCStats events missing from local dataset
# ============================================================

from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = "."

DATASET_EVENT_STATUS_PATH = f"{BASE_PATH}/ufc_dataset_event_status.parquet"

UFCSTATS_COMPLETED_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"

EVENT_CHECK_OUTPUT = f"{BASE_PATH}/ufc_ufcstats_event_check.parquet"
MISSING_EVENTS_OUTPUT = f"{BASE_PATH}/ufc_missing_events.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()

# ============================================================
# HELPERS
# ============================================================

def normalize_event_name(name):
    if pd.isna(name):
        return ""

    return (
        str(name)
        .lower()
        .replace(":", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )


# ============================================================
# LOAD LOCAL EVENT STATUS
# ============================================================

local_events = pd.read_parquet(DATASET_EVENT_STATUS_PATH)

local_events["event_name_norm"] = local_events["event_name"].apply(
    normalize_event_name
)

local_events["event_date"] = pd.to_datetime(
    local_events["event_date"],
    errors="coerce",
).dt.date

print("Local events:", len(local_events))

# ============================================================
# SCRAPE UFCSTATS COMPLETED EVENTS
# ============================================================

response = requests.get(
    UFCSTATS_COMPLETED_EVENTS_URL,
    timeout=30,
)

response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

rows = soup.select("tr.b-statistics__table-row")

ufcstats_rows = []

for row in rows:
    link = row.select_one("a.b-link_style_black")

    if link is None:
        continue

    event_name = link.get_text(strip=True)
    event_url = link.get("href")

    date_cell = row.select_one("span.b-statistics__date")

    event_date = (
        date_cell.get_text(strip=True)
        if date_cell is not None
        else None
    )

    ufcstats_rows.append({
        "run_id": RUN_ID,
        "run_timestamp": RUN_TIMESTAMP,
        "ufcstats_event_name": event_name,
        "ufcstats_event_url": event_url,
        "ufcstats_event_date": event_date,
        "ufcstats_event_name_norm": normalize_event_name(event_name),
    })

ufcstats_events = pd.DataFrame(ufcstats_rows)

ufcstats_events["ufcstats_event_date"] = pd.to_datetime(
    ufcstats_events["ufcstats_event_date"],
    errors="coerce",
).dt.date

print("UFCStats completed events:", len(ufcstats_events))

# ============================================================
# COMPARE UFCSTATS VS LOCAL DATASET
# ============================================================

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
]

comparison = comparison.sort_values(
    "ufcstats_event_date",
    ascending=False,
)

missing_events = comparison[
    comparison["status"] == "missing"
].copy()

# ============================================================
# SAVE OUTPUTS
# ============================================================

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

# ============================================================
# SUMMARY
# ============================================================

print("========== UFCSTATS EVENT CHECK SUMMARY ==========")
print("UFCStats events:", len(ufcstats_events))
print("Local events:", len(local_events))
print("Missing events:", len(missing_events))

if len(missing_events) > 0:
    print()
    print("Most recent missing events:")
    print(
        missing_events[
            [
                "ufcstats_event_name",
                "ufcstats_event_date",
            ]
        ].head(10)
    )