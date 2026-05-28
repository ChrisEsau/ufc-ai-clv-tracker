from datetime import datetime, timezone

import pandas as pd

from scrapers.ufcstats_fighter_profiles import scrape_fighter_profile


BASE_PATH = "."

STAGED_FIGHTS_PATH = f"{BASE_PATH}/ufc_staged_fight_rows.parquet"
STAGED_MASTER_PATH = f"{BASE_PATH}/ufc_staged_master_rows_enriched.parquet"

PROFILE_OUTPUT = f"{BASE_PATH}/ufc_staged_fighter_profiles.parquet"
PROFILED_MASTER_OUTPUT = f"{BASE_PATH}/ufc_staged_master_rows_profiled.parquet"
AUDIT_OUTPUT = f"{BASE_PATH}/ufc_fighter_profile_scrape_audit.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()

MAX_FIGHTERS_TO_SCRAPE = 50


staged_fights = pd.read_parquet(STAGED_FIGHTS_PATH)
staged_master = pd.read_parquet(STAGED_MASTER_PATH)

fighter_rows = []

for _, row in staged_fights.iterrows():
    fighter_rows.append(
        {
            "fighter_role": "r",
            "fighter_name": row.get("red_fighter"),
            "fighter_url": row.get("red_fighter_url"),
        }
    )

    fighter_rows.append(
        {
            "fighter_role": "b",
            "fighter_name": row.get("blue_fighter"),
            "fighter_url": row.get("blue_fighter_url"),
        }
    )

fighters = pd.DataFrame(fighter_rows)

fighters = fighters.dropna(
    subset=["fighter_url"]
).drop_duplicates(
    subset=["fighter_url"]
).reset_index(drop=True)

fighters["fighter_id"] = (
    fighters["fighter_url"]
    .astype(str)
    .str.split("/")
    .str[-1]
)

if MAX_FIGHTERS_TO_SCRAPE:
    fighters = fighters.head(MAX_FIGHTERS_TO_SCRAPE)

print("Fighters to scrape:", len(fighters))

profile_dfs = []
audit_rows = []

for idx, row in fighters.iterrows():
    fighter_url = row["fighter_url"]
    fighter_id = row["fighter_id"]

    print()
    print(f"[{idx + 1}/{len(fighters)}] {fighter_url}")

    try:
        profile = scrape_fighter_profile(
            fighter_url=fighter_url,
            fighter_id=fighter_id,
        )

        profile["run_id"] = RUN_ID
        profile["run_timestamp"] = RUN_TIMESTAMP

        profile_dfs.append(profile)

        audit_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "fighter_id": fighter_id,
                "fighter_url": fighter_url,
                "status": "success",
                "error": None,
            }
        )

        print("Success")

    except Exception as e:
        audit_rows.append(
            {
                "run_id": RUN_ID,
                "run_timestamp": RUN_TIMESTAMP,
                "fighter_id": fighter_id,
                "fighter_url": fighter_url,
                "status": "failed",
                "error": str(e),
            }
        )

        print("FAILED")
        print(e)


profiles = (
    pd.concat(profile_dfs, ignore_index=True)
    if profile_dfs
    else pd.DataFrame()
)

profiles.to_parquet(PROFILE_OUTPUT, index=False)

audit = pd.DataFrame(audit_rows)
audit.to_parquet(AUDIT_OUTPUT, index=False)

# ============================================================
# Merge profiles onto staged master rows
# ============================================================

profile_lookup = profiles.set_index("fighter_name") if not profiles.empty else pd.DataFrame()

profile_cols = [
    "nick_name",
    "wins",
    "losses",
    "draws",
    "height",
    "weight",
    "reach",
    "stance",
    "dob",
    "splm",
    "str_acc",
    "sapm",
    "str_def",
    "td_avg",
    "td_avg_acc",
    "td_def",
    "sub_avg",
]

for side, name_col in [("r", "r_name"), ("b", "b_name")]:
    for profile_col in profile_cols:
        target_col = f"{side}_{profile_col}"

        def lookup_profile(name):
            if profiles.empty:
                return None

            matches = profiles[profiles["fighter_name"] == name]

            if matches.empty:
                return None

            return matches.iloc[0].get(profile_col)

        if target_col in staged_master.columns:
            staged_master[target_col] = staged_master[name_col].apply(lookup_profile)

    id_col = f"{side}_id"

    if id_col in staged_master.columns:
        def lookup_id(name):
            if profiles.empty:
                return None

            matches = profiles[profiles["fighter_name"] == name]

            if matches.empty:
                return None

            return matches.iloc[0].get("fighter_id")

        staged_master[id_col] = staged_master[name_col].apply(lookup_id)


if "winner" in staged_master.columns and "winner_id" in staged_master.columns:
    def get_winner_id(row):
        if row.get("winner") == row.get("r_name"):
            return row.get("r_id")

        if row.get("winner") == row.get("b_name"):
            return row.get("b_id")

        return None

    staged_master["winner_id"] = staged_master.apply(get_winner_id, axis=1)


staged_master.to_parquet(PROFILED_MASTER_OUTPUT, index=False)

print()
print("========== FIGHTER PROFILE ENRICHMENT ==========")
print("Profiles scraped:", len(profiles))
print("Audit rows:", len(audit))
print("Profiled staged rows:", len(staged_master))
print("Saved:", PROFILE_OUTPUT)
print("Saved:", PROFILED_MASTER_OUTPUT)
print("Saved:", AUDIT_OUTPUT)
