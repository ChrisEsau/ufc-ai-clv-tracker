from datetime import datetime, timezone
import pandas as pd


BASE_PATH = "."

MASTER_PATH = f"{BASE_PATH}/ufc_master.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()

BACKUP_PATH = (
    f"{BASE_PATH}/ufc_master_backup_before_date_repair_{RUN_ID}.parquet"
)


# ============================================================
# LOAD
# ============================================================

master = pd.read_parquet(MASTER_PATH)

before_invalid = pd.to_datetime(
    master["date"],
    errors="coerce"
).isna().sum()

print("========== MASTER DATE FORMAT REPAIR ==========")
print("Rows:", len(master))
print("Invalid dates before:", before_invalid)


# ============================================================
# BACKUP
# ============================================================

master.to_parquet(
    BACKUP_PATH,
    index=False
)

print("Backup saved:", BACKUP_PATH)


# ============================================================
# REPAIR DATE FORMAT
# ============================================================

parsed_dates = pd.to_datetime(
    master["date"],
    errors="coerce"
)

master["date"] = parsed_dates.dt.strftime("%-m/%-d/%Y")


# ============================================================
# VALIDATE
# ============================================================

after_invalid = pd.to_datetime(
    master["date"],
    errors="coerce"
).isna().sum()

print("Invalid dates after:", after_invalid)


# ============================================================
# SAVE
# ============================================================

master.to_parquet(
    MASTER_PATH,
    index=False
)

print("Saved:", MASTER_PATH)
