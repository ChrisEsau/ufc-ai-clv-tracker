from datetime import datetime, timezone

import pandas as pd

from pipeline.common.paths import STAGED_FIGHT_DETAILS_PATH


BASE_PATH = "."

DETAILS_PATH = STAGED_FIGHT_DETAILS_PATH

STRUCTURE_AUDIT_OUTPUT = f"{BASE_PATH}/ufc_fight_detail_structure_audit.parquet"
RAW_SAMPLE_OUTPUT = f"{BASE_PATH}/ufc_fight_detail_raw_sample.parquet"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


details = pd.read_parquet(DETAILS_PATH)

raw_cols = [
    c for c in details.columns
    if c.startswith("raw_detail_col_")
]

audit_rows = []

for col in raw_cols:
    non_null_count = int(details[col].notna().sum())

    sample_values = (
        details[col]
        .dropna()
        .astype(str)
        .head(10)
        .tolist()
    )

    audit_rows.append(
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "column_name": col,
            "non_null_count": non_null_count,
            "non_null_pct": (
                non_null_count / len(details) * 100
                if len(details) > 0
                else 0
            ),
            "sample_values": " | ".join(sample_values),
        }
    )

structure_audit = pd.DataFrame(audit_rows)

structure_audit.to_parquet(
    STRUCTURE_AUDIT_OUTPUT,
    index=False,
)

sample_cols = [
    "event_name",
    "event_date",
    "fight_url",
    "red_fighter",
    "blue_fighter",
    "red_result",
    "blue_result",
    "method",
    "round",
    "time",
    "time_format",
    "referee",
    "parse_status",
    "raw_col_count",
] + raw_cols

sample_cols = [
    c for c in sample_cols
    if c in details.columns
]

raw_sample = details[sample_cols].head(25)

raw_sample.to_parquet(
    RAW_SAMPLE_OUTPUT,
    index=False,
)

print("========== FIGHT DETAIL STRUCTURE AUDIT ==========")
print("Detail rows:", len(details))
print("Raw detail columns:", len(raw_cols))
print("Saved:", STRUCTURE_AUDIT_OUTPUT)
print("Saved:", RAW_SAMPLE_OUTPUT)

if not structure_audit.empty:
    print()
    print(structure_audit[["column_name", "non_null_count", "sample_values"]])