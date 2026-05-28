from datetime import datetime, timezone
import pandas as pd
import numpy as np

from scrapers.ufcstats_fight_details import scrape_fight_details
from scrapers.ufcstats_fighter_profiles import scrape_fighter_profile


MASTER_PATH = "./ufc_master.parquet"
OUTPUT_PATH = "./ufc_known_fight_scrape_validation.parquet"

TARGET_EVENT_ID = "6e380a4d73ab4f0e"
TARGET_FIGHT_ID = "d14fea43712707f0"
TARGET_FIGHT_URL = f"http://ufcstats.com/fight-details/{TARGET_FIGHT_ID}"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


def safe_pct(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")

    if pd.isna(den) or den == 0:
        return np.nan

    return round(num / den * 100, 0)


def clean_compare_value(x):
    if pd.isna(x):
        return "nan"

    return str(x).strip()


def values_match(master_val, mapped_val):
    if pd.isna(master_val) and pd.isna(mapped_val):
        return True

    try:
        return bool(
            np.isclose(
                float(master_val),
                float(mapped_val),
                equal_nan=True,
            )
        )
    except Exception:
        return clean_compare_value(master_val) == clean_compare_value(mapped_val)


def profile_to_side(mapped, side, profile):
    prefix = f"{side}_"

    mapped[f"{prefix}nick_name"] = profile.get("nick_name")
    mapped[f"{prefix}wins"] = profile.get("wins")
    mapped[f"{prefix}losses"] = profile.get("losses")
    mapped[f"{prefix}draws"] = profile.get("draws")
    mapped[f"{prefix}height"] = profile.get("height")
    mapped[f"{prefix}weight"] = profile.get("weight")
    mapped[f"{prefix}reach"] = profile.get("reach")
    mapped[f"{prefix}stance"] = profile.get("stance")
    mapped[f"{prefix}dob"] = profile.get("dob")
    mapped[f"{prefix}splm"] = profile.get("splm")
    mapped[f"{prefix}str_acc"] = profile.get("str_acc")
    mapped[f"{prefix}sapm"] = profile.get("sapm")
    mapped[f"{prefix}str_def"] = profile.get("str_def")
    mapped[f"{prefix}td_avg"] = profile.get("td_avg")
    mapped[f"{prefix}td_avg_acc"] = profile.get("td_avg_acc")
    mapped[f"{prefix}td_def"] = profile.get("td_def")
    mapped[f"{prefix}sub_avg"] = profile.get("sub_avg")


master = pd.read_parquet(MASTER_PATH)

master_row_df = master[
    (master["event_id"].astype(str) == TARGET_EVENT_ID)
    & (master["fight_id"].astype(str) == TARGET_FIGHT_ID)
]

if master_row_df.empty:
    raise ValueError("Known target fight not found in master parquet.")

master_row = master_row_df.iloc[0]

scraped = scrape_fight_details(
    fight_url=TARGET_FIGHT_URL,
    event_name=master_row["event_name"],
    event_date=master_row["date"],
    fight_order=None,
)

scraped_row = scraped.iloc[0]

mapped = pd.Series(index=master.columns, dtype="object")

# Master-context fields not currently scraped from fight page
for col in [
    "event_id",
    "event_name",
    "date",
    "location",
    "fight_id",
    "division",
    "title_fight",
    "total_rounds",
]:
    mapped[col] = master_row[col]

mapped["method"] = scraped_row.get("method")
mapped["finish_round"] = pd.to_numeric(scraped_row.get("round"), errors="coerce")
mapped["referee"] = scraped_row.get("referee")

# If fight time exists, convert M:SS to seconds
time_val = scraped_row.get("time")
if isinstance(time_val, str) and ":" in time_val:
    m, s = time_val.split(":")
    mapped["match_time_sec"] = int(m) * 60 + int(s)

# Fighter IDs from master for known validation
mapped["r_id"] = master_row["r_id"]
mapped["b_id"] = master_row["b_id"]

mapped["r_name"] = scraped_row.get("red_fighter")
mapped["b_name"] = scraped_row.get("blue_fighter")

# Core stats
field_map = {
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

for master_col, scrape_col in field_map.items():
    mapped[master_col] = pd.to_numeric(scraped_row.get(scrape_col), errors="coerce")

# Derived accuracy / percentage fields
for side in ["r", "b"]:
    mapped[f"{side}_sig_str_acc"] = safe_pct(
        mapped[f"{side}_sig_str_landed"],
        mapped[f"{side}_sig_str_atmpted"],
    )

    mapped[f"{side}_total_str_acc"] = safe_pct(
        mapped[f"{side}_total_str_landed"],
        mapped[f"{side}_total_str_atmpted"],
    )

    mapped[f"{side}_td_acc"] = safe_pct(
        mapped[f"{side}_td_landed"],
        mapped[f"{side}_td_atmpted"],
    )

    for zone in ["head", "body", "leg", "dist", "clinch", "ground"]:
        mapped[f"{side}_{zone}_acc"] = safe_pct(
            mapped[f"{side}_{zone}_landed"],
            mapped[f"{side}_{zone}_atmpted"],
        )

        mapped[f"{side}_landed_{zone}_per"] = safe_pct(
            mapped[f"{side}_{zone}_landed"],
            mapped[f"{side}_sig_str_landed"],
        )

# Winner
if scraped_row.get("red_result") == "W":
    mapped["winner"] = mapped["r_name"]
    mapped["winner_id"] = mapped["r_id"]
elif scraped_row.get("blue_result") == "W":
    mapped["winner"] = mapped["b_name"]
    mapped["winner_id"] = mapped["b_id"]

# Fighter profiles
r_url = f"http://ufcstats.com/fighter-details/{mapped['r_id']}"
b_url = f"http://ufcstats.com/fighter-details/{mapped['b_id']}"

r_profile = scrape_fighter_profile(r_url, fighter_id=mapped["r_id"]).iloc[0]
b_profile = scrape_fighter_profile(b_url, fighter_id=mapped["b_id"]).iloc[0]

profile_to_side(mapped, "r", r_profile)
profile_to_side(mapped, "b", b_profile)

# Merge-artifact columns
for col in ["r_name_x", "b_name_x", "r_name_y", "b_name_y"]:
    mapped[col] = np.nan

rows = []

for col in master.columns:
    master_val = master_row[col]
    mapped_val = mapped[col]

    match = values_match(master_val, mapped_val)

    rows.append(
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "event_id": TARGET_EVENT_ID,
            "fight_id": TARGET_FIGHT_ID,
            "column_name": col,
            "master_value": clean_compare_value(master_val),
            "mapped_value": clean_compare_value(mapped_val),
            "values_match": bool(match),
        }
    )

validation = pd.DataFrame(rows)
validation["overall_match_pass"] = validation["values_match"].all()

validation.to_parquet(OUTPUT_PATH, index=False)

print("========== KNOWN FIGHT FULL 128-COLUMN VALIDATION ==========")
print("Event ID:", TARGET_EVENT_ID)
print("Fight ID:", TARGET_FIGHT_ID)
print("Compared columns:", len(validation))
print("Matched:", int(validation["values_match"].sum()))
print("Mismatched:", int((~validation["values_match"]).sum()))
print("OVERALL PASS:", bool(validation["overall_match_pass"].iloc[0]))

print()
print("Mismatches:")
print(
    validation[validation["values_match"] == False][
        ["column_name", "master_value", "mapped_value"]
    ]
)

print()
print("Saved:", OUTPUT_PATH)
