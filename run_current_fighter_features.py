# ============================================================
# run_current_fighter_features.py
# ============================================================

import pandas as pd

from pipeline.common.paths import (
    CURRENT_FIGHTER_FEATURES_PATH,
    ROLLING_FEATURES_PATH,
    ensure_data_dirs,
)
from ufc_pipeline_utils import normalize_name


CURRENT_FEATURES_OUTPUT = CURRENT_FIGHTER_FEATURES_PATH


def first_existing_column(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col
    return None


def build_current_fighter_features(rolling_df):
    """
    Convert historical fight-level rolling features into a current
    one-row-per-fighter feature store for live predictions.
    """

    rolling_df = rolling_df.copy()

    rolling_df["date"] = pd.to_datetime(
        rolling_df["date"],
        errors="coerce"
    )

    rolling_df = rolling_df.sort_values("date").reset_index(drop=True)

    possible_red_id_cols = [
        "r_fighter_id",
        "red_fighter_id",
        "R_fighter_id",
        "R_ID",
        "r_id",
    ]

    possible_blue_id_cols = [
        "b_fighter_id",
        "blue_fighter_id",
        "B_fighter_id",
        "B_ID",
        "b_id",
    ]

    possible_red_name_cols = [
        "r_name",
        "red_fighter",
        "R_fighter",
        "R",
    ]

    possible_blue_name_cols = [
        "b_name",
        "blue_fighter",
        "B_fighter",
        "B",
    ]

    red_id_col = first_existing_column(
        rolling_df,
        possible_red_id_cols,
    )

    blue_id_col = first_existing_column(
        rolling_df,
        possible_blue_id_cols,
    )

    red_name_col = first_existing_column(
        rolling_df,
        possible_red_name_cols,
    )

    blue_name_col = first_existing_column(
        rolling_df,
        possible_blue_name_cols,
    )

    if red_name_col is None or blue_name_col is None:
        raise ValueError("Could not find fighter name columns.")

    long_rows = []

    for _, row in rolling_df.iterrows():

        red_long = {
            "fighter_name": row.get(red_name_col),
            "fighter_norm": normalize_name(row.get(red_name_col)),
            "fighter_id": str(row.get(red_id_col, "")) if red_id_col else "",
            "latest_fight_date": row["date"],
            "source_side": "red",
        }

        blue_long = {
            "fighter_name": row.get(blue_name_col),
            "fighter_norm": normalize_name(row.get(blue_name_col)),
            "fighter_id": str(row.get(blue_id_col, "")) if blue_id_col else "",
            "latest_fight_date": row["date"],
            "source_side": "blue",
        }

        for col in rolling_df.columns:

            if col.startswith("r_pre_"):
                neutral = col.replace("r_pre_", "")
                red_long[neutral] = row[col]

            elif col.startswith("b_pre_"):
                neutral = col.replace("b_pre_", "")
                blue_long[neutral] = row[col]

            elif col.startswith("r_ewm_"):
                neutral = col.replace("r_ewm_", "ewm_")
                red_long[neutral] = row[col]

            elif col.startswith("b_ewm_"):
                neutral = col.replace("b_ewm_", "ewm_")
                blue_long[neutral] = row[col]

            elif col.startswith("r_recent_form_"):
                neutral = col.replace("r_recent_form_", "recent_form_")
                red_long[neutral] = row[col]

            elif col.startswith("b_recent_form_"):
                neutral = col.replace("b_recent_form_", "recent_form_")
                blue_long[neutral] = row[col]

            elif col.startswith("r_") and not col.startswith("r_pre_"):
                neutral = col.replace("r_", "")
                red_long[neutral] = row[col]

            elif col.startswith("b_") and not col.startswith("b_pre_"):
                neutral = col.replace("b_", "")
                blue_long[neutral] = row[col]

        long_rows.append(red_long)
        long_rows.append(blue_long)

    fighter_long_df = pd.DataFrame(long_rows)
    fighter_long_df = fighter_long_df[fighter_long_df["fighter_id"].astype(str).str.strip().ne("")].copy()

    # Fighter IDs are canonical. Names are display fields only, and UFCStats can
    # change display names over time (for example suffixes or shortened names).
    # Grouping by fighter_norm creates duplicate fighter_id rows and multiplies
    # live-card feature joins, so collapse to the latest row per fighter_id only.
    current_fighter_features = (
        fighter_long_df
        .sort_values(["fighter_id", "latest_fight_date"])
        .groupby(
            "fighter_id",
            as_index=False,
            group_keys=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    current_fighter_features["feature_store_updated_at"] = (
        pd.Timestamp.now("UTC").isoformat()
    )

    return current_fighter_features


def main():
    print("Building current fighter feature store...")
    ensure_data_dirs()

    rolling_df = pd.read_parquet(ROLLING_FEATURES_PATH)

    print("Rolling feature rows:", len(rolling_df))
    print("Rolling feature cols:", len(rolling_df.columns))

    current_fighter_features = build_current_fighter_features(
        rolling_df
    )

    current_fighter_features.to_parquet(
        CURRENT_FEATURES_OUTPUT,
        index=False,
    )

    print("Current fighter feature rows:", len(current_fighter_features))
    print("Saved:")
    print(CURRENT_FEATURES_OUTPUT)


if __name__ == "__main__":
    main()
