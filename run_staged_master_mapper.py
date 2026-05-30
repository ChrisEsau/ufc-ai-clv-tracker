import pandas as pd
import numpy as np
from datetime import datetime, timezone

from pipeline.paths import (
    STAGED_FIGHT_DETAILS_PATH,
    MASTER_PATH,
    STAGED_MASTER_ROWS_PATH,
    STAGED_MASTER_MAPPING_AUDIT_PATH,
)

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

staged = pd.read_parquet(STAGED_FIGHT_DETAILS_PATH)
master = pd.read_parquet(MASTER_PATH)

print(f"Staged rows: {len(staged)}")
print(f"Master cols: {len(master.columns)}")

print()
print("========== STAGED COLUMNS ==========")
print(list(staged.columns))

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

if "fight_id" in staged.columns:
    mapped["fight_id"] = staged["fight_id"]
else:
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
# ZONE STRIKING MAPS
# =========================
for side, prefix in [("r", "red"), ("b", "blue")]:
    for zone in ["head", "body", "leg", "dist", "clinch", "ground"]:
        mapped[f"{side}_{zone}_landed"] = pd.to_numeric(
            staged[f"{side}_{zone}_landed"],
            errors="coerce",
        )
        mapped[f"{side}_{zone}_atmpted"] = pd.to_numeric(
            staged[f"{side}_{zone}_atmpted"],
            errors="coerce",
        )


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

def extract_id_from_url(url):
    if pd.isna(url):
        return None

    return (
        str(url)
        .strip()
        .rstrip("/")
        .split("/")[-1]
    )


if "event_id" not in staged.columns:
    raise ValueError(
        "Missing event_id column in staged fight details."
    )

mapped["event_id"] = staged["event_id"]


mapped["winner_id"] = np.nan

mapped["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
mapped["run_timestamp"] = datetime.now(timezone.utc)

# =========================
# ALIGN COLUMN ORDER
# =========================

mapped = mapped.reindex(columns=master.columns)

# =========================
# SAVE OUTPUTS
# =========================

mapped.to_parquet(
    STAGED_MASTER_ROWS_PATH,
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
    STAGED_MASTER_MAPPING_AUDIT_PATH,
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
print("Saved:", STAGED_MASTER_ROWS_PATH)
print("Saved:", STAGED_MASTER_MAPPING_AUDIT_PATH)
