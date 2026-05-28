from datetime import datetime, timezone

import pandas as pd

from scrapers.ufcstats_fights import scrape_event_fights


BASE_PATH = "."

MISSING_EVENTS_PATH = f"{BASE_PATH}/ufc_missing_events.parquet"

STAGED_OUTPUT = f"{BASE_PATH}/ufc_staged_fight_rows.parquet"
AUDIT_OUTPUT = f"{BASE_PATH}/ufc_fight_scrape_audit.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


missing_events = pd.read_parquet(
    MISSING_EVENTS_PATH
)

print(
    "Missing events:",
    len(missing_events)
)

all_fights = []
audit_rows = []

for idx, row in missing_events.iterrows():

    event_name = row.get(
        "ufcstats_event_name"
    )

    event_date = row.get(
        "ufcstats_event_date"
    )

    event_url = row.get(
        "ufcstats_event_url"
    )

    print()
    print(
        f"[{idx+1}/{len(missing_events)}] "
        f"Scraping: {event_name}"
    )

    try:

        fights = scrape_event_fights(
            event_url=event_url,
            event_name=event_name,
            event_date=event_date,
        )

        fights["run_id"] = RUN_ID
        fights["run_timestamp"] = RUN_TIMESTAMP

        all_fights.append(
            fights
        )

        audit_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "event_name": event_name,
                "event_date": event_date,
                "event_url": event_url,
                "status": "success",
                "fight_count": len(fights),
                "error": None,
            }
        )

        print(
            f"Success: {len(fights)} fights"
        )

    except Exception as e:

        audit_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "event_name": event_name,
                "event_date": event_date,
                "event_url": event_url,
                "status": "failed",
                "fight_count": 0,
                "error": str(e),
            }
        )

        print(
            f"FAILED: {event_name}"
        )

        print(e)


if len(all_fights) > 0:

    staged_df = pd.concat(
        all_fights,
        ignore_index=True,
    )

else:

    staged_df = pd.DataFrame()


audit_df = pd.DataFrame(
    audit_rows
)

staged_df.to_parquet(
    STAGED_OUTPUT,
    index=False,
)

audit_df.to_parquet(
    AUDIT_OUTPUT,
    index=False,
)

print()
print("========== SCRAPE SUMMARY ==========")
print(
    "Total staged fights:",
    len(staged_df)
)
print(
    "Events attempted:",
    len(audit_df)
)
print(
    "Successful events:",
    (audit_df["status"] == "success").sum()
)
print(
    "Failed events:",
    (audit_df["status"] == "failed").sum()
)

print()
print(
    "Saved:",
    STAGED_OUTPUT
)

print(
    "Saved:",
    AUDIT_OUTPUT
)