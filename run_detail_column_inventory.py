import pandas as pd

from pipeline.common.paths import STAGED_FIGHT_DETAILS_PATH

DETAILS_PATH = STAGED_FIGHT_DETAILS_PATH

df = pd.read_parquet(DETAILS_PATH)

print("========== DETAIL COLUMN INVENTORY ==========")
print()

print("Column count:", len(df.columns))
print()

for col in df.columns:
    print(col)