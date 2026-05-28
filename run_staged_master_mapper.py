import pandas as pd
import numpy as np
from datetime import datetime

print("========== STAGED MASTER MAPPER ==========")

def time_to_seconds(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if ":" not in x:
        return np.nan

    mins, secs = x.split(":")
    return int(mins) * 60 + int(secs)
    
# =========================
# LOAD DATA
# =========================

staged = pd.read_parquet("ufc_staged_fight_details.parquet")
master = pd.read_parquet("ufc_master.parquet")

print(f"Staged rows: {len(staged)}")
print(f"Master cols: {len(master.columns)}")

# =========================
# CREATE OUTPUT FRAME
# =========================

mapped = pd.DataFrame(columns=master.columns)

# =========================
# DIRECT FIELD MAPS
# =========================

mapped["event_name"] = staged["event_name"]
mapped["date"] = pd.to_datetime(
    staged["event_date"],
    errors="coerce"
).dt.strftime("%-m/%-d/%Y")

mapped["fight_id"] = (
    staged["fight_url"]
    .astype(str)
    .str.split("/")
    .str[-1]
)

mapped["method"] = staged["method"]

mapped["finish_round"] = pd.to_numeric(
    staged["round"],
    errors="coerce"
)

mapped["match_time_sec"] = staged["time"].apply(time_to_seconds)

mapped["referee"] = staged["referee"]

# =========================
# RED FIGHTER MAPS
# =========================

mapped["r_name"] = staged["red_fighter"]

mapped["r_kd"] = pd.to_numeric(
    staged["red_kd"],
    errors="coerce"
)

mapped["r_sig_str_landed"] = pd.to_numeric(
    staged["red_sig_str_landed"],
    errors="coerce"
)

mapped["r_sig_str_atmpted"] = pd.to_numeric(
    staged["red_sig_str_attempted"],
    errors="coerce"
)

mapped["r_total_str_landed"] = pd.to_numeric(
    staged["red_total_str_landed"],
    errors="coerce"
)

mapped["r_total_str_atmpted"] = pd.to_numeric(
    staged["red_total_str_attempted"],
    errors="coerce"
)

mapped["r_td_landed"] = pd.to_numeric(
    staged["red_td_landed"],
    errors="coerce"
)

mapped["r_td_atmpted"] = pd.to_numeric(
    staged["red_td_attempted"],
    errors="coerce"
)

mapped["r_sub_att"] = pd.to_numeric(
    staged["red_sub_att"],
    errors="coerce"
)

mapped["r_ctrl"] = staged["red_ctrl"]

# =========================
# BLUE FIGHTER MAPS
# =========================

mapped["b_name"] = staged["blue_fighter"]

mapped["b_kd"] = pd.to_numeric(
    staged["blue_kd"],
    errors="coerce"
)

mapped["b_sig_str_landed"] = pd.to_numeric(
    staged["blue_sig_str_landed"],
    errors="coerce"
)

mapped["b_sig_str_atmpted"] = pd.to_numeric(
    staged["blue_sig_str_attempted"],
    errors="coerce"
)

mapped["b_total_str_landed"] = pd.to_numeric(
    staged["blue_total_str_landed"],
    errors="coerce"
)

mapped["b_total_str_atmpted"] = pd.to_numeric(
    staged["blue_total_str_attempted"],
    errors="coerce"
)

mapped["b_td_landed"] = pd.to_numeric(
    staged["blue_td_landed"],
    errors="coerce"
)

mapped["b_td_atmpted"] = pd.to_numeric(
    staged["blue_td_attempted"],
    errors="coerce"
)

mapped["b_sub_att"] = pd.to_numeric(
    staged["blue_sub_att"],
    errors="coerce"
)

mapped["b_ctrl"] = staged["blue_ctrl"]

# =========================
# PERCENTAGE DERIVATIONS
# =========================

mapped["r_sig_str_acc"] = np.where(
    mapped["r_sig_str_atmpted"] > 0,
    (
        mapped["r_sig_str_landed"]
        / mapped["r_sig_str_atmpted"]
        * 100
    ).round(0),
    np.nan
)

mapped["b_sig_str_acc"] = np.where(
    mapped["b_sig_str_atmpted"] > 0,
    (
        mapped["b_sig_str_landed"]
        / mapped["b_sig_str_atmpted"]
        * 100
    ).round(0),
    np.nan
)

mapped["r_td_acc"] = np.where(
    mapped["r_td_atmpted"] > 0,
    (
        mapped["r_td_landed"]
        / mapped["r_td_atmpted"]
        * 100
    ).round(0),
    np.nan
)

mapped["b_td_acc"] = np.where(
    mapped["b_td_atmpted"] > 0,
    (
        mapped["b_td_landed"]
        / mapped["b_td_atmpted"]
        * 100
    ).round(0),
    np.nan
)

# =========================
# WINNER DERIVATION
# =========================

mapped["winner"] = np.where(
    staged["red_result"].astype(str).str.lower() == "w",
    staged["red_fighter"],
    np.where(
        staged["blue_result"].astype(str).str.lower() == "w",
        staged["blue_fighter"],
        np.nan
    )
)

# =========================
# RUN METADATA
# =========================

mapped["event_id"] = np.nan
mapped["winner_id"] = np.nan

mapped["run_id"] = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
mapped["run_timestamp"] = datetime.utcnow()

# =========================
# ALIGN COLUMN ORDER
# =========================

mapped = mapped.reindex(columns=master.columns)

# =========================
# SAVE OUTPUTS
# =========================

mapped.to_parquet(
    "ufc_staged_master_rows.parquet",
    index=False
)

audit = pd.DataFrame({
    "column_name": master.columns,
    "non_null_count": [
        mapped[c].notna().sum()
        for c in master.columns
    ]
})

audit.to_parquet(
    "ufc_staged_master_mapping_audit.parquet",
    index=False
)

print()
print("========== MAPPER SUMMARY ==========")
print(f"Mapped rows: {len(mapped)}")
print(f"Mapped cols: {len(mapped.columns)}")

filled_cols = (
    audit["non_null_count"] > 0
).sum()

print(f"Populated cols: {filled_cols}")

print()
print("Saved: ./ufc_staged_master_rows.parquet")
print("Saved: ./ufc_staged_master_mapping_audit.parquet")
