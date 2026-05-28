from datetime import datetime, timezone
import pandas as pd
import numpy as np


BASE_PATH = "."

MASTER_PATH = f"{BASE_PATH}/ufc_master.parquet"
STAGED_PATH = f"{BASE_PATH}/ufc_staged_master_rows_profiled.parquet"

PRECHECK_OUTPUT = f"{BASE_PATH}/ufc_append_precheck.parquet"
DUPLICATE_OUTPUT = f"{BASE_PATH}/ufc_append_duplicate_check.parquet"
REQUIRED_FIELD_OUTPUT = f"{BASE_PATH}/ufc_append_required_field_audit.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


master = pd.read_parquet(MASTER_PATH)
staged = pd.read_parquet(STAGED_PATH)

checks = []

master_cols = list(master.columns)
staged_cols = list(staged.columns)

column_count_match = len(master_cols) == len(staged_cols)
column_order_match = master_cols == staged_cols

checks.append({
    "check_name": "column_count_match",
    "status": "pass" if column_count_match else "fail",
    "failure_count": 0 if column_count_match else 1,
    "details": f"master={len(master_cols)}, staged={len(staged_cols)}",
})

checks.append({
    "check_name": "column_order_match",
    "status": "pass" if column_order_match else "fail",
    "failure_count": 0 if column_order_match else 1,
    "details": "staged column order matches master" if column_order_match else "staged column order does not match master",
})


# ============================================================
# DUPLICATE CHECKS
# ============================================================

def clean_id_series(s):
    return (
        s.astype(str)
        .str.strip()
        .replace(["", "nan", "None", "NaN"], pd.NA)
    )

staged_fight_ids = clean_id_series(staged["fight_id"])
master_fight_ids = clean_id_series(master["fight_id"])

valid_staged_fight_ids = staged_fight_ids.dropna()
valid_master_fight_ids = master_fight_ids.dropna()

duplicate_in_staged = staged[
    staged_fight_ids.notna()
    & staged_fight_ids.duplicated(keep=False)
].copy()

already_in_master = staged[
    staged_fight_ids.notna()
    & staged_fight_ids.isin(set(valid_master_fight_ids))
].copy()

duplicate_in_staged_count = len(duplicate_in_staged)
already_in_master_count = len(already_in_master)

checks.append({
    "check_name": "duplicate_fight_ids_in_staged",
    "status": "pass" if duplicate_in_staged_count == 0 else "fail",
    "failure_count": duplicate_in_staged_count,
    "details": f"{duplicate_in_staged_count} duplicate staged fight_id rows",
})

checks.append({
    "check_name": "fight_ids_already_in_master",
    "status": "pass" if already_in_master_count == 0 else "fail",
    "failure_count": already_in_master_count,
    "details": f"{already_in_master_count} staged fight_id rows already exist in master",
})

duplicate_check = pd.concat(
    [
        duplicate_in_staged.assign(duplicate_type="duplicate_in_staged"),
        already_in_master.assign(duplicate_type="already_in_master"),
    ],
    ignore_index=True,
)

duplicate_check.to_parquet(DUPLICATE_OUTPUT, index=False)


# ============================================================
# REQUIRED FIELD CHECKS
# ============================================================

required_fields = [
    "event_name",
    "date",
    "fight_id",
    "r_name",
    "b_name",
    "method",
    "finish_round",
    "match_time_sec",
    "r_sig_str_landed",
    "b_sig_str_landed",
    "r_total_str_landed",
    "b_total_str_landed",
    "winner",
]

required_rows = []

for col in required_fields:
    if col not in staged.columns:
        missing_count = len(staged)
    else:
        series = staged[col]
        missing_count = int(
            series.isna().sum()
            + (series.astype(str).str.strip().isin(["", "nan", "None"])).sum()
        )

    required_rows.append({
        "column_name": col,
        "missing_count": missing_count,
        "status": "pass" if missing_count == 0 else "fail",
    })

required_audit = pd.DataFrame(required_rows)
required_audit.to_parquet(REQUIRED_FIELD_OUTPUT, index=False)

failed_required = required_audit[
    required_audit["status"] == "fail"
]

print()
print("========== FAILED REQUIRED FIELDS ==========")

if failed_required.empty:
    print("None")
else:
    print(
        failed_required[
            ["column_name", "missing_count"]
        ].to_string(index=False)
    )

required_failures = int((required_audit["status"] == "fail").sum())

checks.append({
    "check_name": "required_fields_populated",
    "status": "pass" if required_failures == 0 else "fail",
    "failure_count": required_failures,
    "details": f"{required_failures} required fields have missing values",
})


# ============================================================
# NEGATIVE STAT CHECK
# ============================================================

stat_cols = [
    c for c in staged.columns
    if any(
        token in c
        for token in [
            "_kd",
            "_landed",
            "_atmpted",
            "_acc",
            "_td_",
            "_sub_att",
            "_ctrl",
            "_per",
            "match_time_sec",
        ]
    )
]

negative_count = 0

for col in stat_cols:
    vals = pd.to_numeric(staged[col], errors="coerce")
    negative_count += int((vals < 0).sum())

checks.append({
    "check_name": "negative_stat_check",
    "status": "pass" if negative_count == 0 else "fail",
    "failure_count": negative_count,
    "details": f"{negative_count} negative numeric stat values found",
})


# ============================================================
# FINAL PRECHECK
# ============================================================

precheck = pd.DataFrame(checks)

append_ready = bool((precheck["status"] == "pass").all())

precheck["run_id"] = RUN_ID
precheck["run_timestamp"] = RUN_TIMESTAMP
precheck["staged_rows"] = len(staged)
precheck["master_rows"] = len(master)
precheck["append_ready"] = append_ready

precheck.to_parquet(PRECHECK_OUTPUT, index=False)

print("========== APPEND PRECHECK VALIDATION ==========")
print("Master rows:", len(master))
print("Staged rows:", len(staged))
print("Append ready:", append_ready)
print()
print(precheck[["check_name", "status", "failure_count", "details"]])
print()
print("Saved:", PRECHECK_OUTPUT)
print("Saved:", DUPLICATE_OUTPUT)
print("Saved:", REQUIRED_FIELD_OUTPUT)
print()
print("========== FIGHT ID DEBUG ==========")
print("Staged fight_id nulls:", int(staged_fight_ids.isna().sum()))
print("Staged unique valid fight_ids:", int(valid_staged_fight_ids.nunique()))
print("Master unique valid fight_ids:", int(valid_master_fight_ids.nunique()))
print("Overlap count:", int(staged_fight_ids.isin(set(valid_master_fight_ids)).sum()))
print()
print("========== OVERLAPPING FIGHTS ==========")

overlap_preview = already_in_master[
    ["event_name", "date", "fight_id", "r_name", "b_name"]
].head(25)

print(overlap_preview.to_string(index=False))
