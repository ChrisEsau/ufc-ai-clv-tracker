from datetime import datetime, timezone
import pandas as pd
import numpy as np

from pipeline.common.paths import MASTER_PATH, STAGED_MASTER_ROWS_PATH


BASE_PATH = "."

MAPPED_PATH = STAGED_MASTER_ROWS_PATH

OUTPUT_PATH = f"{BASE_PATH}/ufc_append_row_match_validation.parquet"

TARGET_EVENT_ID = "6e380a4d73ab4f0e"
TARGET_FIGHT_ID = "d14fea43712707f0"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


master = pd.read_parquet(MASTER_PATH)
mapped = pd.read_parquet(MAPPED_PATH)

master_row = master[
    (master["event_id"].astype(str) == TARGET_EVENT_ID)
    & (master["fight_id"].astype(str) == TARGET_FIGHT_ID)
]

common_fight_ids = set(master["fight_id"].astype(str)).intersection(
    set(mapped["fight_id"].astype(str))
)

if not common_fight_ids:
    raise ValueError(
        "No overlapping fight_id values found between master and mapped staged parquet."
    )

TARGET_FIGHT_ID = sorted(common_fight_ids)[0]

master_row = master[
    master["fight_id"].astype(str) == TARGET_FIGHT_ID
]

mapped_row = mapped[
    mapped["fight_id"].astype(str) == TARGET_FIGHT_ID
]

if master_row.empty:
    raise ValueError("Target fight not found in master parquet.")

if mapped_row.empty:
    raise ValueError("Target fight not found in mapped staged parquet.")

master_row = master_row.iloc[0]
mapped_row = mapped_row.iloc[0]

compare_cols = [
    "event_name",
    "date",
    "fight_id",
    "method",
    "finish_round",
    "referee",
    "r_name",
    "b_name",
    "winner",
    "r_kd",
    "b_kd",
    "r_sig_str_landed",
    "r_sig_str_atmpted",
    "b_sig_str_landed",
    "b_sig_str_atmpted",
    "r_total_str_landed",
    "r_total_str_atmpted",
    "b_total_str_landed",
    "b_total_str_atmpted",
    "r_td_landed",
    "r_td_atmpted",
    "b_td_landed",
    "b_td_atmpted",
    "r_sub_att",
    "b_sub_att",
    "r_ctrl",
    "b_ctrl",
    "r_head_landed",
    "r_head_atmpted",
    "b_head_landed",
    "b_head_atmpted",
    "r_body_landed",
    "r_body_atmpted",
    "b_body_landed",
    "b_body_atmpted",
    "r_leg_landed",
    "r_leg_atmpted",
    "b_leg_landed",
    "b_leg_atmpted",
    "r_dist_landed",
    "r_dist_atmpted",
    "b_dist_landed",
    "b_dist_atmpted",
    "r_clinch_landed",
    "r_clinch_atmpted",
    "b_clinch_landed",
    "b_clinch_atmpted",
    "r_ground_landed",
    "r_ground_atmpted",
    "b_ground_landed",
    "b_ground_atmpted",
]

rows = []

for col in compare_cols:

    if col not in master.columns or col not in mapped.columns:
        continue

    master_val = master_row[col]
    mapped_val = mapped_row[col]

    master_str = str(master_val)
    mapped_str = str(mapped_val)

    if pd.isna(master_val) and pd.isna(mapped_val):
        match = True
    else:
        match = master_str == mapped_str

        try:
            if pd.notna(master_val) and pd.notna(mapped_val):
                match = np.isclose(
                    float(master_val),
                    float(mapped_val),
                    equal_nan=True,
                )
        except Exception:
            pass

    rows.append(
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "event_id": TARGET_EVENT_ID,
            "fight_id": TARGET_FIGHT_ID,
            "column_name": col,
            "master_value": master_val,
            "mapped_value": mapped_val,
            "values_match": bool(match),
        }
    )

validation = pd.DataFrame(rows)

validation["overall_match_pass"] = validation["values_match"].all()

validation.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("========== APPEND ROW MATCH VALIDATION ==========")
print("Event ID:", TARGET_EVENT_ID)
print("Fight ID:", TARGET_FIGHT_ID)
print("Compared columns:", len(validation))
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
            "column_name",
            "master_value",
            "mapped_value",
        ]
    ]
)

print()
print("Saved:", OUTPUT_PATH)
