# ============================================================
# run_model_predictions.py
# Model-only live prediction runner
# ============================================================

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from pipeline_config import *
from ufc_pipeline_utils import *
from ufc_feature_engineering import (
    add_v5_engineered_features,
    get_engineered_feature_list,
)

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = "."

MODEL_VERSION = "UFC_Model_v5_Experiment"
PRODUCTION_DIR = f"{BASE_PATH}/{MODEL_VERSION}"

LIVE_CARD_PATH = f"{BASE_PATH}/ufc_live_card.parquet"
CURRENT_FIGHTER_FEATURES_PATH = f"{BASE_PATH}/ufc_current_fighter_features.parquet"

MODEL_PREDICTIONS_OUTPUT = f"{BASE_PATH}/ufc_model_predictions.parquet"
FEATURE_AUDIT_OUTPUT = f"{BASE_PATH}/ufc_live_feature_audit.parquet"
MATCH_AUDIT_OUTPUT = f"{BASE_PATH}/ufc_live_match_audit.parquet"

PREDICTION_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
PREDICTION_TIMESTAMP = datetime.now(timezone.utc).isoformat()

CLIP_LOW = 0.02
CLIP_HIGH = 0.98

# ============================================================
# LOAD ARTIFACTS
# ============================================================

print("Loading model artifacts...")

model = joblib.load(f"{PRODUCTION_DIR}/calibrated_model.pkl")
feature_columns = joblib.load(f"{PRODUCTION_DIR}/feature_columns.pkl")
BEST_THRESHOLD = joblib.load(f"{PRODUCTION_DIR}/best_threshold.pkl")
production_config = joblib.load(f"{PRODUCTION_DIR}/production_config.pkl")

print("Model version:", production_config.get("version", MODEL_VERSION))
print("Feature count:", len(feature_columns))
print("Best threshold:", BEST_THRESHOLD)

# ============================================================
# LOAD INPUTS
# ============================================================

live_card_df = pd.read_parquet(LIVE_CARD_PATH)
fighter_features_df = pd.read_parquet(CURRENT_FIGHTER_FEATURES_PATH)

print("Live card rows:", len(live_card_df))
print("Current fighter feature rows:", len(fighter_features_df))

fighter_features_df["fighter_id"] = fighter_features_df["fighter_id"].astype(str)
fighter_features_df["fighter_norm"] = fighter_features_df["fighter_norm"].astype(str)

# ============================================================
# MATCH HELPERS
# ============================================================

def find_fighter_row(fighter_id, fighter_name, fighter_features_df):
    fighter_id = str(fighter_id).strip()
    fighter_norm = normalize_name(fighter_name)

    if fighter_id:
        id_match = fighter_features_df[
            fighter_features_df["fighter_id"].astype(str) == fighter_id
        ]

        if len(id_match) > 0:
            return id_match.iloc[0], "id_match"

    exact_name_match = fighter_features_df[
        fighter_features_df["fighter_norm"] == fighter_norm
    ]

    if len(exact_name_match) > 0:
        return exact_name_match.iloc[0], "exact_name_match"

    temp = fighter_features_df.copy()

    temp["name_score"] = temp["fighter_norm"].apply(
        lambda x: token_set_score(x, fighter_norm)
    )

    temp = temp.sort_values("name_score", ascending=False)

    best = temp.iloc[0]

    if best["name_score"] >= 0.88:
        return best, f"fuzzy_name_match_{round(best['name_score'], 3)}"

    return None, "missing"


# ============================================================
# MATCH AUDIT
# ============================================================

match_audit_rows = []

for _, fight in live_card_df.iterrows():
    red_row, red_source = find_fighter_row(
        fight["red_fighter_id"],
        fight["red_fighter"],
        fighter_features_df,
    )

    blue_row, blue_source = find_fighter_row(
        fight["blue_fighter_id"],
        fight["blue_fighter"],
        fighter_features_df,
    )

    match_audit_rows.append({
        "prediction_run_id": PREDICTION_RUN_ID,
        "prediction_timestamp": PREDICTION_TIMESTAMP,
        "event_name": fight.get("event_name"),
        "fight_id": fight.get("fight_id"),
        "red_fighter": fight["red_fighter"],
        "blue_fighter": fight["blue_fighter"],
        "red_fighter_id": fight["red_fighter_id"],
        "blue_fighter_id": fight["blue_fighter_id"],
        "red_feature_match": red_source,
        "blue_feature_match": blue_source,
        "matched_red_name": None if red_row is None else red_row["fighter_name"],
        "matched_blue_name": None if blue_row is None else blue_row["fighter_name"],
        "red_latest_fight_date": None if red_row is None else red_row.get("latest_fight_date"),
        "blue_latest_fight_date": None if blue_row is None else blue_row.get("latest_fight_date"),
    })

match_audit_df = pd.DataFrame(match_audit_rows)

match_audit_df.to_parquet(
    MATCH_AUDIT_OUTPUT,
    index=False,
)

print("Match audit saved:", MATCH_AUDIT_OUTPUT)

# ============================================================
# BUILD LIVE FEATURE ROWS
# ============================================================

ENGINEERED_FEATURE_SET = set(get_engineered_feature_list())


def build_live_feature_row(fight):
    red_row, red_match = find_fighter_row(
        fight["red_fighter_id"],
        fight["red_fighter"],
        fighter_features_df,
    )

    blue_row, blue_match = find_fighter_row(
        fight["blue_fighter_id"],
        fight["blue_fighter"],
        fighter_features_df,
    )

    out = fight.to_dict()

    out["prediction_run_id"] = PREDICTION_RUN_ID
    out["prediction_timestamp"] = PREDICTION_TIMESTAMP
    out["model_version"] = production_config.get("version", MODEL_VERSION)

    out["red_feature_match"] = red_match
    out["blue_feature_match"] = blue_match

    # --------------------------------------------------------
    # Standard model diff features
    # --------------------------------------------------------

    for feature in feature_columns:

        if feature in ENGINEERED_FEATURE_SET:
            continue

        if feature.endswith("_diff"):
            base = feature[:-5]

            red_value = safe_value(red_row, base)
            blue_value = safe_value(blue_row, base)

            out[feature] = red_value - blue_value

    # --------------------------------------------------------
    # Raw prefixed columns needed by centralized engineering
    # --------------------------------------------------------

    raw_feature_map = {
        "age": "age",
        "height": "height",
        "reach": "reach",
        "weight": "weight",

        "pre_str_def": "str_def",
        "pre_td_def": "td_def",
        "pre_sapm": "sapm",
        "pre_splm": "splm",
        "pre_td_avg": "td_avg",
        "pre_sub_avg": "sub_avg",
        "pre_fights": "fights",
        "pre_losses": "losses",
        "pre_ctrl_against_per_min": "ctrl_against_per_min",
    }

    for prefixed_name, source_name in raw_feature_map.items():
        out[f"r_{prefixed_name}"] = safe_value(red_row, source_name)
        out[f"b_{prefixed_name}"] = safe_value(blue_row, source_name)

    return out


live_rows = []

for _, fight in live_card_df.iterrows():
    live_rows.append(build_live_feature_row(fight))

live_df = pd.DataFrame(live_rows)

live_df = add_v5_engineered_features(live_df)

live_df = live_df.loc[
    :,
    ~live_df.columns.duplicated(),
].copy()

print("Live feature rows:", len(live_df))

# ============================================================
# FEATURE VALIDATION
# ============================================================

missing_features = [
    col for col in feature_columns
    if col not in live_df.columns
]

for col in missing_features:
    live_df[col] = 0

X_live = live_df[feature_columns].copy()

for col in X_live.columns:
    X_live[col] = pd.to_numeric(
        X_live[col],
        errors="coerce",
    ).fillna(0)

feature_audit_rows = []

for idx, row in X_live.iterrows():
    zero_pct = (row == 0).mean() * 100
    nonzero_count = (row != 0).sum()

    feature_audit_rows.append({
        "prediction_run_id": PREDICTION_RUN_ID,
        "prediction_timestamp": PREDICTION_TIMESTAMP,
        "event_name": live_df.loc[idx, "event_name"],
        "fight_id": live_df.loc[idx, "fight_id"],
        "red_fighter": live_df.loc[idx, "red_fighter"],
        "blue_fighter": live_df.loc[idx, "blue_fighter"],
        "red_feature_match": live_df.loc[idx, "red_feature_match"],
        "blue_feature_match": live_df.loc[idx, "blue_feature_match"],
        "feature_count_expected": len(feature_columns),
        "feature_count_actual": X_live.shape[1],
        "missing_feature_count": len(missing_features),
        "nonzero_feature_count": nonzero_count,
        "zero_feature_pct": zero_pct,
        "passes_feature_validation": (
            len(missing_features) == 0
            and nonzero_count >= 90
            and zero_pct <= 35
        ),
    })

feature_audit_df = pd.DataFrame(feature_audit_rows)

feature_audit_df.to_parquet(
    FEATURE_AUDIT_OUTPUT,
    index=False,
)

print("Feature audit saved:", FEATURE_AUDIT_OUTPUT)

# ============================================================
# MODEL PREDICTIONS
# ============================================================

live_probs = model.predict_proba(X_live)[:, 1]

live_probs = np.clip(
    live_probs,
    CLIP_LOW,
    CLIP_HIGH,
)

live_df["red_model_prob"] = live_probs
live_df["blue_model_prob"] = 1 - live_probs

live_df["model_pick"] = np.where(
    live_df["red_model_prob"] >= BEST_THRESHOLD,
    live_df["red_fighter"],
    live_df["blue_fighter"],
)

live_df["model_pick_fighter_id"] = np.where(
    live_df["red_model_prob"] >= BEST_THRESHOLD,
    live_df["red_fighter_id"],
    live_df["blue_fighter_id"],
)

live_df["model_confidence"] = np.maximum(
    live_df["red_model_prob"],
    live_df["blue_model_prob"],
)

live_df["passes_model_data_quality"] = (
    (live_df["red_feature_match"] != "missing")
    &
    (live_df["blue_feature_match"] != "missing")
)

# ============================================================
# FINAL MODEL PREDICTION OUTPUT
# ============================================================

prediction_cols = [
    "prediction_run_id",
    "prediction_timestamp",
    "model_version",

    "event_name",
    "commence_time",
    "fight_id",

    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",

    "red_model_prob",
    "blue_model_prob",

    "model_pick",
    "model_pick_fighter_id",
    "model_confidence",

    "red_feature_match",
    "blue_feature_match",
    "passes_model_data_quality",
]

prediction_cols = [
    col for col in prediction_cols
    if col in live_df.columns
]

model_predictions_df = live_df[prediction_cols].copy()

model_predictions_df = model_predictions_df.merge(
    feature_audit_df[
        [
            "fight_id",
            "nonzero_feature_count",
            "zero_feature_pct",
            "passes_feature_validation",
        ]
    ],
    on="fight_id",
    how="left",
)

model_predictions_df.to_parquet(
    MODEL_PREDICTIONS_OUTPUT,
    index=False,
)

print("Model predictions saved:", MODEL_PREDICTIONS_OUTPUT)
print("Prediction rows:", len(model_predictions_df))
