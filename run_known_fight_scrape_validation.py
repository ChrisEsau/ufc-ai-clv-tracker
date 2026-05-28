from datetime import datetime, timezone
import pandas as pd
import numpy as np

from scrapers.ufcstats_fight_details import scrape_fight_details


MASTER_PATH = "./ufc_master.parquet"
OUTPUT_PATH = "./ufc_known_fight_scrape_validation.parquet"

TARGET_EVENT_ID = "6e380a4d73ab4f0e"
TARGET_FIGHT_ID = "d14fea43712707f0"
TARGET_FIGHT_URL = f"http://ufcstats.com/fight-details/{TARGET_FIGHT_ID}"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


master = pd.read_parquet(MASTER_PATH)

master_row = master[
    (master["event_id"].astype(str) == TARGET_EVENT_ID)
    & (master["fight_id"].astype(str) == TARGET_FIGHT_ID)
]

if master_row.empty:
    raise ValueError("Known target fight not found in master parquet.")

master_row = master_row.iloc[0]

scraped = scrape_fight_details(
    fight_url=TARGET_FIGHT_URL,
    event_name=master_row["event_name"],
    event_date=master_row["date"],
    fight_order=None,
)

scraped["fight_id"] = TARGET_FIGHT_ID

compare_map = {
    "r_name": "red_fighter",
    "b_name": "blue_fighter",
    "method": "method",
    "finish_round": "round",
    "referee": "referee",

    "r_kd": "red_kd",
    "b_kd": "blue_kd",

    "r_sig_str_landed": "red_sig_str_landed",
    "r_sig_str_atmpted": "red_sig_str_attempted",
    "b_sig_str_landed": "blue_sig_str_landed",
    "b_sig_str_atmpted": "blue_sig_str_attempted",

    "r_total_str_landed": "red_total_str_landed",
    "r_total_str_atmpted": "red_total_str_attempted",
    "b_total_str_landed": "blue_total_str_landed",
    "b_total_str_atmpted": "blue_total_str_attempted",

    "r_td_landed": "red_td_landed",
    "r_td_atmpted": "red_td_attempted",
    "b_td_landed": "blue_td_landed",
    "b_td_atmpted": "blue_td_attempted",

    "r_sub_att": "red_sub_att",
    "b_sub_att": "blue_sub_att",

    "r_ctrl": "red_ctrl",
    "b_ctrl": "blue_ctrl",

    "r_head_landed": "r_head_landed",
    "r_head_atmpted": "r_head_atmpted",
    "b_head_landed": "b_head_landed",
    "b_head_atmpted": "b_head_atmpted",

    "r_body_landed": "r_body_landed",
    "r_body_atmpted": "r_body_atmpted",
    "b_body_landed": "b_body_landed",
    "b_body_atmpted": "b_body_atmpted",

    "r_leg_landed": "r_leg_landed",
    "r_leg_atmpted": "r_leg_atmpted",
    "b_leg_landed": "b_leg_landed",
    "b_leg_atmpted": "b_leg_atmpted",

    "r_dist_landed": "r_dist_landed",
    "r_dist_atmpted": "r_dist_atmpted",
    "b_dist_landed": "b_dist_landed",
    "b_dist_atmpted": "b_dist_atmpted",

    "r_clinch_landed": "r_clinch_landed",
    "r_clinch_atmpted": "r_clinch_atmpted",
    "b_clinch_landed": "b_clinch_landed",
    "b_clinch_atmpted": "b_clinch_atmpted",

    "r_ground_landed": "r_ground_landed",
    "r_ground_atmpted": "r_ground_atmpted",
    "b_ground_landed": "b_ground_landed",
    "b_ground_atmpted": "b_ground_atmpted",
}

scraped_row = scraped.iloc[0]

rows = []

for master_col, scraped_col in compare_map.items():

    master_val = master_row.get(master_col, np.nan)
    scraped_val = scraped_row.get(scraped_col, np.nan)

    try:
        match = np.isclose(
            float(master_val),
            float(scraped_val),
            equal_nan=True,
        )
    except Exception:
        match = str(master_val).strip() == str(scraped_val).strip()

    rows.append(
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "event_id": TARGET_EVENT_ID,
            "fight_id": TARGET_FIGHT_ID,
            "fight_url": TARGET_FIGHT_URL,
            "master_column": master_col,
            "scraped_column": scraped_col,
            "master_value": str(master_val),
            "scraped_value": str(scraped_val),
            "values_match": bool(match),
        }
    )

validation = pd.DataFrame(rows)

validation["overall_match_pass"] = validation["values_match"].all()

validation.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("========== KNOWN FIGHT SCRAPE VALIDATION ==========")
print("Event ID:", TARGET_EVENT_ID)
print("Fight ID:", TARGET_FIGHT_ID)
print("Compared fields:", len(validation))
print("Matched:", int(validation["values_match"].sum()))
print("Mismatched:", int((~validation["values_match"]).sum()))
print("OVERALL PASS:", bool(validation["overall_match_pass"].iloc[0]))

print()
print("Mismatches:")
print(
    validation[
        validation["values_match"] == False
    ][
        [
            "master_column",
            "master_value",
            "scraped_value",
        ]
    ]
)

print()
print("Saved:", OUTPUT_PATH)
