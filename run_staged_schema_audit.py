from datetime import datetime, timezone

import pandas as pd


BASE_PATH = "."

MASTER_PATH = f"{BASE_PATH}/ufc_master.parquet"
STAGED_PATH = f"{BASE_PATH}/ufc_staged_fight_rows.parquet"

SCHEMA_AUDIT_OUTPUT = f"{BASE_PATH}/ufc_staged_schema_audit.parquet"
STAGED_QUALITY_OUTPUT = f"{BASE_PATH}/ufc_staged_fight_quality_audit.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


master = pd.read_parquet(MASTER_PATH)
staged = pd.read_parquet(STAGED_PATH)

master_cols = set(master.columns)
staged_cols = set(staged.columns)

all_cols = sorted(master_cols.union(staged_cols))

schema_rows = []

for col in all_cols:
    in_master = col in master_cols
    in_staged = col in staged_cols

    master_dtype = str(master[col].dtype) if in_master else None
    staged_dtype = str(staged[col].dtype) if in_staged else None

    schema_rows.append(
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "column_name": col,
            "in_master": in_master,
            "in_staged": in_staged,
            "master_dtype": master_dtype,
            "staged_dtype": staged_dtype,
            "dtype_match": (
                master_dtype == staged_dtype
                if in_master and in_staged
                else None
            ),
        }
    )

schema_audit = pd.DataFrame(schema_rows)

schema_audit["column_status"] = schema_audit.apply(
    lambda r: (
        "matched"
        if r["in_master"] and r["in_staged"]
        else "missing_from_staged"
        if r["in_master"] and not r["in_staged"]
        else "extra_in_staged"
    ),
    axis=1,
)

schema_audit.to_parquet(
    SCHEMA_AUDIT_OUTPUT,
    index=False,
)


# ============================================================
# STAGED QUALITY AUDIT
# ============================================================

required_staged_cols = [
    "event_name",
    "event_date",
    "event_url",
    "fight_url",
    "red_fighter",
    "blue_fighter",
    "red_fighter_url",
    "blue_fighter_url",
    "method",
    "round",
    "time",
    "weight_class",
]

quality = {
    "run_id": RUN_ID,
    "run_timestamp": RUN_TIMESTAMP,
    "staged_row_count": len(staged),
    "staged_column_count": len(staged.columns),
    "master_row_count": len(master),
    "master_column_count": len(master.columns),
}

for col in required_staged_cols:
    if col in staged.columns:
        quality[f"missing_{col}_count"] = int(
            staged[col].isna().sum()
            + (staged[col].astype(str).str.strip() == "").sum()
        )
    else:
        quality[f"missing_{col}_count"] = None

if "fight_url" in staged.columns:
    quality["duplicate_fight_url_count"] = int(
        staged["fight_url"].duplicated().sum()
    )
    quality["unique_fight_url_count"] = int(
        staged["fight_url"].nunique(dropna=True)
    )
else:
    quality["duplicate_fight_url_count"] = None
    quality["unique_fight_url_count"] = None

fallback_key = [
    c for c in [
        "event_name",
        "event_date",
        "red_fighter",
        "blue_fighter",
    ]
    if c in staged.columns
]

if fallback_key:
    quality["duplicate_fallback_key_count"] = int(
        staged.duplicated(subset=fallback_key).sum()
    )
else:
    quality["duplicate_fallback_key_count"] = None

matched_cols = int((schema_audit["column_status"] == "matched").sum())
missing_from_staged = int(
    (schema_audit["column_status"] == "missing_from_staged").sum()
)
extra_in_staged = int(
    (schema_audit["column_status"] == "extra_in_staged").sum()
)

quality["matched_column_count"] = matched_cols
quality["missing_from_staged_count"] = missing_from_staged
quality["extra_in_staged_count"] = extra_in_staged

identity_ready = (
    quality.get("missing_fight_url_count", 999) == 0
    and quality.get("duplicate_fight_url_count", 999) == 0
)

fallback_ready = (
    quality.get("duplicate_fallback_key_count", 999) == 0
)

quality["fight_url_identity_ready"] = identity_ready
quality["fallback_identity_ready"] = fallback_ready

quality["append_ready"] = (
    identity_ready
    and quality.get("missing_red_fighter_count", 999) == 0
    and quality.get("missing_blue_fighter_count", 999) == 0
    and quality.get("missing_event_name_count", 999) == 0
    and quality.get("missing_event_date_count", 999) == 0
)

quality_audit = pd.DataFrame([quality])

quality_audit.to_parquet(
    STAGED_QUALITY_OUTPUT,
    index=False,
)

print("========== STAGED SCHEMA AUDIT ==========")
print("Master rows:", len(master))
print("Staged rows:", len(staged))
print("Matched columns:", matched_cols)
print("Missing from staged:", missing_from_staged)
print("Extra in staged:", extra_in_staged)
print("Append ready:", quality["append_ready"])
print()
print("Saved:", SCHEMA_AUDIT_OUTPUT)
print("Saved:", STAGED_QUALITY_OUTPUT)