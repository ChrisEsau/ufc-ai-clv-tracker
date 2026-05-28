# run_staged_derived_stats_transformer.py

import pandas as pd
import numpy as np


INPUT_PATH = "./ufc_staged_master_rows.parquet"
OUTPUT_PATH = "./ufc_staged_master_rows_enriched.parquet"
AUDIT_PATH = "./ufc_staged_derived_stats_audit.parquet"


df = pd.read_parquet(INPUT_PATH)


def safe_pct(num, den):
    return np.where(
        pd.to_numeric(den, errors="coerce") > 0,
        (
            pd.to_numeric(num, errors="coerce")
            / pd.to_numeric(den, errors="coerce")
            * 100
        ).round(0),
        np.nan,
    )


def time_to_seconds(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if ":" not in x:
        return np.nan

    mins, secs = x.split(":")
    return int(mins) * 60 + int(secs)


# =========================
# Match time
# =========================

if "time" in df.columns:
    df["match_time_sec"] = df["time"].apply(time_to_seconds)


# =========================
# Accuracy columns
# =========================

for side in ["r", "b"]:

    df[f"{side}_sig_str_acc"] = safe_pct(
        df[f"{side}_sig_str_landed"],
        df[f"{side}_sig_str_atmpted"],
    )

    df[f"{side}_total_str_acc"] = safe_pct(
        df[f"{side}_total_str_landed"],
        df[f"{side}_total_str_atmpted"],
    )

    df[f"{side}_td_acc"] = safe_pct(
        df[f"{side}_td_landed"],
        df[f"{side}_td_atmpted"],
    )

    for zone in [
        "head",
        "body",
        "leg",
        "dist",
        "clinch",
        "ground",
    ]:
        df[f"{side}_{zone}_acc"] = safe_pct(
            df[f"{side}_{zone}_landed"],
            df[f"{side}_{zone}_atmpted"],
        )


# =========================
# Landed distribution percentages
# =========================

for side in ["r", "b"]:

    sig_landed = pd.to_numeric(
        df[f"{side}_sig_str_landed"],
        errors="coerce",
    )

    for zone in ["head", "body", "leg"]:
        df[f"{side}_landed_{zone}_per"] = safe_pct(
            df[f"{side}_{zone}_landed"],
            sig_landed,
        )

    for zone in ["dist", "clinch", "ground"]:
        df[f"{side}_landed_{zone}_per"] = safe_pct(
            df[f"{side}_{zone}_landed"],
            sig_landed,
        )


# =========================
# Save
# =========================

df.to_parquet(OUTPUT_PATH, index=False)

audit_rows = []

for col in df.columns:
    audit_rows.append(
        {
            "column_name": col,
            "non_null_count": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
        }
    )

audit = pd.DataFrame(audit_rows)
audit.to_parquet(AUDIT_PATH, index=False)

print("========== DERIVED STATS TRANSFORMER ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved:", OUTPUT_PATH)
print("Saved:", AUDIT_PATH)
