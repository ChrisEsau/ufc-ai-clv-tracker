from datetime import datetime, timezone

import pandas as pd

from scrapers.ufcstats_fight_details import (
    scrape_fight_details,
)

BASE_PATH = "."

STAGED_FIGHTS_PATH = (
    f"{BASE_PATH}/ufc_staged_fight_rows.parquet"
)

staged["fight_url"] = staged["fight_url"].astype(str)

staged = staged[
    staged["fight_url"].notna()
    & (staged["fight_url"].str.strip() != "")
    & (staged["fight_url"].str.lower() != "nan")
].copy()

staged = staged.reset_index(drop=True)

DETAIL_OUTPUT = (
    f"{BASE_PATH}/ufc_staged_fight_details.parquet"
)

AUDIT_OUTPUT = (
    f"{BASE_PATH}/ufc_fight_detail_scrape_audit.parquet"
)

RUN_ID = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)

RUN_TIMESTAMP = datetime.now(
    timezone.utc
).isoformat()

MAX_FIGHTS_TO_SCRAPE = 5
# set to 10 for testing if desired


staged = pd.read_parquet(
    STAGED_FIGHTS_PATH
)

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