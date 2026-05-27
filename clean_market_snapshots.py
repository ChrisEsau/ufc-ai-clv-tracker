import pandas as pd

PATH = "ufc_market_snapshots.parquet"

df = pd.read_parquet(PATH)

print("Before cleanup:", len(df))

required_cols = [
    "fight_id",
    "red_fighter",
    "blue_fighter",
    "red_american_odds",
    "blue_american_odds",
    "bookmaker",
]

df = df.dropna(subset=[c for c in required_cols if c in df.columns]).copy()

if "odds_match_type" in df.columns:
    df = df[df["odds_match_type"] == "matched"].copy()

df = df.drop_duplicates(
    subset=[
        "snapshot_timestamp",
        "fight_id",
    ],
    keep="last",
)

df.to_parquet(PATH, index=False)

print("After cleanup:", len(df))
