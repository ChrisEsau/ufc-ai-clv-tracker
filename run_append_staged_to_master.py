from datetime import datetime, timezone
import shutil
import pandas as pd


BASE_PATH = "."

MASTER_PATH = f"{BASE_PATH}/ufc_master.parquet"
STAGED_PATH = f"{BASE_PATH}/ufc_staged_master_rows_profiled.parquet"
PRECHECK_PATH = f"{BASE_PATH}/ufc_append_precheck.parquet"

APPEND_AUDIT_OUTPUT = f"{BASE_PATH}/ufc_append_audit.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()

BACKUP_PATH = f"{BASE_PATH}/ufc_master_backup_before_append_{RUN_ID}.parquet"


# ============================================================
# LOAD PRECHECK
# ============================================================

precheck = pd.read_parquet(PRECHECK_PATH)

append_ready = bool(precheck["append_ready"].iloc[0])

if not append_ready:
    print("Append precheck failed. Refusing to append.")
    print(precheck[["check_name", "status", "failure_count", "details"]])
    raise SystemExit(1)


# ============================================================
# LOAD DATA
# ============================================================

master = pd.read_parquet(MASTER_PATH)
staged = pd.read_parquet(STAGED_PATH)

before_rows = len(master)
staged_rows = len(staged)

print("========== APPEND STAGED TO MASTER ==========")
print("Master rows before:", before_rows)
print("Staged rows:", staged_rows)


# ============================================================
# FINAL SAFETY CHECKS
# ============================================================

if list(master.columns) != list(staged.columns):
    raise ValueError("Column mismatch. Refusing to append.")

master_ids = set(master["fight_id"].astype(str))
staged_ids = set(staged["fight_id"].astype(str))

overlap = master_ids.intersection(staged_ids)

if overlap:
    raise ValueError(
        f"Refusing to append. {len(overlap)} staged fight_ids already exist in master."
    )


# ============================================================
# BACKUP MASTER
# ============================================================

shutil.copyfile(
    MASTER_PATH,
    BACKUP_PATH,
)

print("Backup saved:", BACKUP_PATH)


# ============================================================
# APPEND
# ============================================================

updated = pd.concat(
    [master, staged],
    ignore_index=True,
)

# Normalize object/date columns before parquet write
for col in updated.columns:
    if updated[col].dtype == "object":
        updated[col] = updated[col].apply(
            lambda x: x.isoformat() if hasattr(x, "isoformat") else x
        )
  
# ============================================================
# ALIGN STAGED DTYPES TO MASTER
# ============================================================

for col in master.columns:

    master_dtype = master[col].dtype

    try:

        if pd.api.types.is_numeric_dtype(master_dtype):

            updated[col] = pd.to_numeric(
                updated[col],
                errors="coerce"
            )

        elif pd.api.types.is_datetime64_any_dtype(master_dtype):

            updated[col] = pd.to_datetime(
                updated[col],
                errors="coerce"
            )

        else:

            updated[col] = updated[col].astype(str)

    except Exception as e:

        print(f"WARNING dtype harmonization failed for {col}: {e}")   
        
updated.to_parquet(
    MASTER_PATH,
    index=False,
)

after_rows = len(updated)


# ============================================================
# AUDIT
# ============================================================

audit = pd.DataFrame(
    [
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "master_path": MASTER_PATH,
            "backup_path": BACKUP_PATH,
            "before_rows": before_rows,
            "staged_rows": staged_rows,
            "after_rows": after_rows,
            "expected_after_rows": before_rows + staged_rows,
            "row_count_pass": after_rows == before_rows + staged_rows,
            "append_status": (
                "success"
                if after_rows == before_rows + staged_rows
                else "row_count_mismatch"
            ),
        }
    ]
)

audit.to_parquet(
    APPEND_AUDIT_OUTPUT,
    index=False,
)

print()
print("========== APPEND SUMMARY ==========")
print("Rows before:", before_rows)
print("Rows appended:", staged_rows)
print("Rows after:", after_rows)
print("Expected rows after:", before_rows + staged_rows)
print("Append status:", audit["append_status"].iloc[0])
print()
print("Saved:", MASTER_PATH)
print("Saved:", APPEND_AUDIT_OUTPUT)
