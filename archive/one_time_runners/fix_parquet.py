import pandas as pd

MASTER_PATH = "data/master/ufc_master.parquet"

df = pd.read_parquet(MASTER_PATH)

print("Before:", len(df))

date_check = pd.to_datetime(df["date"], errors="coerce")

bad_rows = df[date_check.isna()]

print("Rows being removed:", len(bad_rows))
print(bad_rows[["event_name", "fight_id", "date"]])

df = df[~date_check.isna()]

print("After:", len(df))

df.to_parquet(MASTER_PATH, index=False)