from datetime import datetime, timezone

import pandas as pd


BASE_PATH = "."

from pipeline.common.paths import (
    MASTER_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
    MASTER_COLUMN_VALIDATION_PATH,
)

MAPPED_PATH = STAGED_MASTER_ROWS_PROFILED_PATH
VALIDATION_OUTPUT = MASTER_COLUMN_VALIDATION_PATH

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


master = pd.read_parquet(MASTER_PATH)
mapped = pd.read_parquet(MAPPED_PATH)

master_cols = list(master.columns)
mapped_cols = list(mapped.columns)

rows = []

max_len = max(len(master_cols), len(mapped_cols))

for i in range(max_len):

    master_col = master_cols[i] if i < len(master_cols) else None
    mapped_col = mapped_cols[i] if i < len(mapped_cols) else None

    master_dtype = (
        str(master[master_col].dtype)
        if master_col in master.columns
        else None
    )

    mapped_dtype = (
        str(mapped[mapped_col].dtype)
        if mapped_col in mapped.columns
        else None
    )

    rows.append(
        {
            "run_id": RUN_ID,
            "run_timestamp": RUN_TIMESTAMP,
            "position": i,
            "master_column": master_col,
            "mapped_column": mapped_col,
            "column_name_match": master_col == mapped_col,
            "master_dtype": master_dtype,
            "mapped_dtype": mapped_dtype,
            "dtype_match": master_dtype == mapped_dtype,
        }
    )

validation = pd.DataFrame(rows)

exact_column_count_match = len(master_cols) == len(mapped_cols)
exact_column_order_match = validation["column_name_match"].all()
duplicate_mapped_columns = mapped.columns.duplicated().sum()
missing_from_mapped = sorted(set(master_cols) - set(mapped_cols))
extra_in_mapped = sorted(set(mapped_cols) - set(master_cols))

validation["exact_column_count_match"] = exact_column_count_match
validation["exact_column_order_match"] = exact_column_order_match
validation["duplicate_mapped_columns"] = duplicate_mapped_columns
validation["missing_from_mapped_count"] = len(missing_from_mapped)
validation["extra_in_mapped_count"] = len(extra_in_mapped)

validation_pass = (
    exact_column_count_match
    and exact_column_order_match
    and duplicate_mapped_columns == 0
    and len(missing_from_mapped) == 0
    and len(extra_in_mapped) == 0
)

validation["validation_pass"] = validation_pass

validation.to_parquet(
    VALIDATION_OUTPUT,
    index=False,
)

print("========== MASTER COLUMN VALIDATION ==========")
print("Master columns:", len(master_cols))
print("Mapped columns:", len(mapped_cols))
print("Column count match:", exact_column_count_match)
print("Column order match:", exact_column_order_match)
print("Duplicate mapped columns:", duplicate_mapped_columns)
print("Missing from mapped:", len(missing_from_mapped))
print("Extra in mapped:", len(extra_in_mapped))
print("VALIDATION PASS:", validation_pass)
print()
print("Saved:", VALIDATION_OUTPUT)
