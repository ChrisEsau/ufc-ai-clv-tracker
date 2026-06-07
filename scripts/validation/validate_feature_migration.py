import pandas as pd

legacy = pd.read_parquet(
    "data/features/UFC_enhanced_rolling_features_EWM.parquet"
)

print("\nLEGACY DATASET")
print("Rows:", len(legacy))
print("Cols:", len(legacy.columns))

print("\nCOLUMN CHECKS")

print("r_pre columns:",
      len([c for c in legacy.columns if c.startswith("r_pre_")]))

print("b_pre columns:",
      len([c for c in legacy.columns if c.startswith("b_pre_")]))

print("diff columns:",
      len([c for c in legacy.columns if c.endswith("_diff")]))

print("r_ewm columns:",
      len([c for c in legacy.columns if c.startswith("r_ewm_")]))

print("b_ewm columns:",
      len([c for c in legacy.columns if c.startswith("b_ewm_")]))

print("recent form columns:",
      len([c for c in legacy.columns if "recent_form" in c]))

print("\nTOTAL FEATURE COLUMNS")

feature_cols = [
    c for c in legacy.columns
    if c not in [
        "target",
        "fight_id",
        "event_id",
        "date"
    ]
]

print(len(feature_cols))