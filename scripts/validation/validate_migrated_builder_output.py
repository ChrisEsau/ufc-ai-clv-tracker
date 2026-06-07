import pandas as pd

from pipeline.features.base.build_rolling_features import (
    build_rolling_base_features,
)
from pipeline.features.base.ewm_features import (
    add_ewm_feature_layer,
)

from ufc_feature_engineering import add_v5_engineered_features

# ============================================================
# LOAD SOURCE DATA
# ============================================================

MASTER_PATH = "data/master/ufc_master.parquet"

LEGACY_PATH = (
    "data/features/UFC_enhanced_rolling_features_EWM.parquet"
)

print("=" * 80)
print("LOAD DATA")
print("=" * 80)

master_df = pd.read_parquet(MASTER_PATH)
legacy_df = pd.read_parquet(LEGACY_PATH)

print("Master rows :", len(master_df))
print("Legacy rows :", len(legacy_df))
print()

# ============================================================
# BASIC PREP
# ============================================================

df = master_df.copy()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "event_id", "fight_id"]
).reset_index(drop=True)

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

df["target"] = (
    df["winner"]
    .fillna("")
    .astype(str)
    .str.lower()
    .eq("red")
).astype(int)

print("=" * 80)
print("BUILD BASE FEATURES")
print("=" * 80)

candidate_df = build_rolling_base_features(df)

print("Base shape:", candidate_df.shape)

print()
print("=" * 80)
print("ADD EWM FEATURES")
print("=" * 80)

candidate_df = add_ewm_feature_layer(candidate_df)
candidate_df = add_v5_engineered_features(candidate_df)

print("Candidate shape:", candidate_df.shape)

# ============================================================
# COMPARE
# ============================================================

print()
print("=" * 80)
print("PARITY CHECK")
print("=" * 80)

legacy_cols = set(legacy_df.columns)
candidate_cols = set(candidate_df.columns)

missing_cols = sorted(legacy_cols - candidate_cols)
extra_cols = sorted(candidate_cols - legacy_cols)

print("Legacy columns    :", len(legacy_cols))
print("Candidate columns :", len(candidate_cols))

print()
print("Missing columns :", len(missing_cols))
print("Extra columns   :", len(extra_cols))

if missing_cols:
    print("\nFIRST 50 MISSING")
    print(missing_cols[:50])

if extra_cols:
    print("\nFIRST 50 EXTRA")
    print(extra_cols[:50])

print()
print("=" * 80)
print("ROW CHECK")
print("=" * 80)

print("Legacy rows    :", len(legacy_df))
print("Candidate rows :", len(candidate_df))

print()
print("=" * 80)
print("COLUMN MATCH")
print("=" * 80)

print(
    "Exact column parity:",
    len(missing_cols) == 0 and len(extra_cols) == 0
)

print()
print("DONE")