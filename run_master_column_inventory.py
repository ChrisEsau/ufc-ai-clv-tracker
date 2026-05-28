# run_master_column_inventory.py

import pandas as pd

MASTER_PATH = "./ufc_master.parquet"

df = pd.read_parquet(MASTER_PATH)

print("========== MASTER COLUMN INVENTORY ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

for col in df.columns:
    print(col)