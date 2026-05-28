from datetime import datetime, timezone
import pandas as pd


BASE_PATH = "."

MASTER_PATH = f"{BASE_PATH}/ufc_master.parquet"
STAGED_PATH = f"{BASE_PATH}/ufc_staged_fight_rows.parquet"

MAPPING_AUDIT_OUTPUT = f"{BASE_PATH}/ufc_staged_mapping_audit.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


master = pd.read_parquet(MASTER_PATH)
staged = pd.read_parquet(STAGED_PATH)

manual_mapping = {
    "event_name": ["event_name"],
    "event_date": ["event_date"],
    "date": ["event_date"],
    "red_fighter": ["red_fighter"],
    "blue_fighter": ["blue_fighter"],
    "r_fighter": ["red_fighter"],
    "b_fighter": ["blue_fighter"],
    "winner": ["winner"],
    "method": ["method"],
    "round": ["round"],
    "time": ["time"],
    "weight_class": ["weight_class"],
    "fight_url": ["fight_url"],
    "event_url": ["event_url"],
    "red_fighter_url": ["red_fighter_url"],
    "blue_fighter_url": ["blue_fighter_url"],
}

rows = []

for master_col in master.columns:
    direct_match = master_col in staged.columns

    mapped_source = None
    mapping_type = "unmapped"

    if direct_match:
        mapped_source = master_col
        mapping_type = "direct"
    elif master_col in manual_mapping:
        for candidate in manual_mapping[master_col]:
            if candidate in staged.columns:
                mapped_source = candidate
                mapping_type = "manual"
                break

    if mapped_source:
        staged_non_null_pct = (
            staged[mapped_source].notna().mean() * 100
        )
    else:
        staged_non_null_pct = None

    rows.append(
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "master_column": master_col,
            "master_dtype": str(master[master_col].dtype),
            "mapped_source_column": mapped_source,
            "mapping_type": mapping_type,
            "append_populatable": mapped_source is not None,
            "staged_non_null_pct": staged_non_null_pct,
            "requires_fight_detail_scrape": mapping_type == "unmapped",
        }
    )

mapping_df = pd.DataFrame(rows)

mapping_df.to_parquet(
    MAPPING_AUDIT_OUTPUT,
    index=False,
)

print("========== STAGED MAPPING AUDIT ==========")
print("Master columns:", len(master.columns))
print("Staged columns:", len(staged.columns))
print("Direct mappings:", (mapping_df["mapping_type"] == "direct").sum())
print("Manual mappings:", (mapping_df["mapping_type"] == "manual").sum())
print("Unmapped:", (mapping_df["mapping_type"] == "unmapped").sum())
print("Populatable:", mapping_df["append_populatable"].sum())
print("Saved:", MAPPING_AUDIT_OUTPUT)