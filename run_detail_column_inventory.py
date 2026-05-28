import pandas as pd

DETAILS_PATH = "./ufc_staged_fight_details.parquet"

df = pd.read_parquet(DETAILS_PATH)

print("========== DETAIL COLUMN INVENTORY ==========")
print()

print("Column count:", len(df.columns))
print()

for col in df.columns:
    print(col)