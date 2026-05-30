from datetime import datetime, timezone

import pandas as pd

from scrapers.ufcstats_fight_details import (
    scrape_fight_details,
)

from pipeline.paths import (
    STAGED_FIGHT_ROWS_PATH,
    STAGED_FIGHT_DETAILS_PATH,
    FIGHT_DETAIL_SCRAPE_AUDIT_PATH,
)

STAGED_FIGHTS_PATH = STAGED_FIGHT_ROWS_PATH
DETAIL_OUTPUT = STAGED_FIGHT_DETAILS_PATH
AUDIT_OUTPUT = FIGHT_DETAIL_SCRAPE_AUDIT_PATH

RUN_ID = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)

RUN_TIMESTAMP = datetime.now(
    timezone.utc
).isoformat()

MAX_FIGHTS_TO_SCRAPE = 1
# set to 10 for testing if desired


staged = pd.read_parquet(
    STAGED_FIGHTS_PATH
)

staged["fight_url"] = staged["fight_url"].astype(str)

staged = staged[
    staged["fight_url"].notna()
    & (staged["fight_url"].str.strip() != "")
    & (staged["fight_url"].str.lower() != "nan")
].copy()

staged = staged.reset_index(drop=True)

if MAX_FIGHTS_TO_SCRAPE:
    staged = staged.head(
        MAX_FIGHTS_TO_SCRAPE
    )

print(
    "Fight rows to scrape:",
    len(staged)
)

detail_rows = []
audit_rows = []

for idx, row in staged.iterrows():

    fight_url = row.get("fight_url")

    print()
    print(
        f"[{idx+1}/{len(staged)}] "
        f"{fight_url}"
    )

    try:

        details = scrape_fight_details(
            fight_url=fight_url,
            event_name=row.get(
                "event_name"
            ),
            event_date=row.get(
                "event_date"
            ),
            fight_order=row.get(
                "fight_order"
            ),
        )

        details["run_id"] = RUN_ID
        details["run_timestamp"] = RUN_TIMESTAMP
        details["event_id"] = row.get("event_id")
        details["event_url"] = row.get("event_url")
        details["fight_id"] = row.get("fight_id")
        details["fight_url"] = row.get("fight_url")

        print("DEBUG event_id:", row.get("event_id"))
        print("DEBUG fight_id:", row.get("fight_id"))
        print("DEBUG fight_url:", row.get("fight_url"))

        detail_rows.append(
            details
        )

        audit_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "fight_url": fight_url,
                "event_name": row.get(
                    "event_name"
                ),
                "status": "success",
                "detail_rows": len(details),
                "error": None,
            }
        )

        print(
            f"Success: {len(details)} detail rows"
        )

    except Exception as e:

        audit_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "fight_url": fight_url,
                "event_name": row.get(
                    "event_name"
                ),
                "status": "failed",
                "detail_rows": 0,
                "error": str(e),
            }
        )

        print("FAILED")
        print(e)


if len(detail_rows) > 0:

    details_df = pd.concat(
        detail_rows,
        ignore_index=True,
    )

else:

    details_df = pd.DataFrame()


audit_df = pd.DataFrame(
    audit_rows
)

details_df.to_parquet(
    DETAIL_OUTPUT,
    index=False,
)

audit_df.to_parquet(
    AUDIT_OUTPUT,
    index=False,
)

print()
print("========== DETAIL SCRAPE SUMMARY ==========")
print(
    "Detail rows:",
    len(details_df)
)
print(
    "Audit rows:",
    len(audit_df)
)
print(
    "Successes:",
    (audit_df["status"] == "success").sum()
)
print(
    "Failures:",
    (audit_df["status"] == "failed").sum()
)

print()
print("Saved:", DETAIL_OUTPUT)
print("Saved:", AUDIT_OUTPUT)
