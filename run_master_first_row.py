import pandas as pd

MASTER_PATH = "./ufc_master.parquet"

df = pd.read_parquet(MASTER_PATH)

print("========== MASTER FIRST ROW ==========")
print()

row = df.iloc[0]

for col in df.columns:
    print(f"{col}: {row[col]}")