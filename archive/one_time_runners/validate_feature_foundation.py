from pathlib import Path
import pandas as pd

MASTER_PATH = Path("data/master/ufc_master.parquet")
ROLLING_PATH = Path("data/features/UFC_enhanced_rolling_features_EWM.parquet")
INVENTORY_PATH = Path("configs/features/full_rolling_feature_inventory.yaml")

ENGINEERED_FEATURES = [
    "age_diff",
    "height_diff",
    "reach_diff",
    "weight_diff",
    "striking_edge",
    "grappling_edge",
    "finish_volatility",
    "wrestling_pressure_vs_defense",
    "reach_striking_combo",
    "chin_risk_diff",
    "experience_ratio_diff",
    "aggression_index_diff",
    "age_squared_diff",
    "pressure_striking_adv_diff",
    "wrestling_mismatch_diff",
    "submission_mismatch_diff",
]

print("=" * 80)
print("UFC FEATURE FOUNDATION VALIDATION")
print("=" * 80)

# ------------------------------------------------------------------
# MASTER
# ------------------------------------------------------------------

master_df = pd.read_parquet(MASTER_PATH)

print("\nMASTER DATASET")
print(f"Rows    : {len(master_df):,}")
print(f"Columns : {len(master_df.columns)}")

# ------------------------------------------------------------------
# ROLLING
# ------------------------------------------------------------------

rolling_df = pd.read_parquet(ROLLING_PATH)

print("\nROLLING FEATURE DATASET")
print(f"Rows    : {len(rolling_df):,}")
print(f"Columns : {len(rolling_df.columns)}")

# ------------------------------------------------------------------
# INVENTORY
# ------------------------------------------------------------------

inventory_columns = []

if INVENTORY_PATH.exists():
    for line in INVENTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line.startswith('name: "') and line.endswith('"'):
            inventory_columns.append(line[7:-1])

print("\nINVENTORY")
print(f"Columns : {len(inventory_columns)}")

# ------------------------------------------------------------------
# COMPARE INVENTORY VS ROLLING
# ------------------------------------------------------------------

rolling_cols = set(rolling_df.columns)
inventory_cols = set(inventory_columns)

missing_from_rolling = sorted(inventory_cols - rolling_cols)
extra_in_rolling = sorted(rolling_cols - inventory_cols)

print("\nINVENTORY VALIDATION")
print(f"Missing from rolling : {len(missing_from_rolling)}")
print(f"Extra in rolling     : {len(extra_in_rolling)}")

# ------------------------------------------------------------------
# CURRENT MONEYLINE CONTRACT
# ------------------------------------------------------------------

moneyline_features = []

for col in rolling_df.columns:
    if col.endswith("_diff"):
        moneyline_features.append(col)

for feature in ENGINEERED_FEATURES:
    if feature in rolling_df.columns:
        moneyline_features.append(feature)

moneyline_features = sorted(set(moneyline_features))

print("\nCURRENT MONEYLINE CONTRACT")
print(f"Feature count : {len(moneyline_features)}")

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------

print("\nEXPECTED VALUES")
print("Rolling columns expected     : 483")
print("Moneyline features expected  : 124")

print("\nRESULTS")
print(
    f"Rolling columns match?      : "
    f"{len(rolling_df.columns) == 483}"
)

print(
    f"Moneyline count match?      : "
    f"{len(moneyline_features) == 124}"
)

print(
    f"Inventory match?            : "
    f"{len(missing_from_rolling) == 0 and len(extra_in_rolling) == 0}"
)

print("\nDONE")