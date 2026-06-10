import pandas as pd
import numpy as np
from datetime import datetime, timezone

from pipeline.common.fight_context import (
    clean_division,
    title_fight_flag,
    total_rounds_from_time_format,
)
from pipeline.common.paths import (
    STAGED_FIGHT_DETAILS_PATH,
    MASTER_PATH,
    STAGED_MASTER_ROWS_PATH,
    STAGED_MASTER_MAPPING_AUDIT_PATH,
)


def first_existing_series(df, column_names, default=None):
    for column_name in column_names:
        if column_name in df.columns:
            return df[column_name]

    return pd.Series(default, index=df.index)


def safe_pct(num, den):
    numerator = pd.to_numeric(num, errors="coerce").fillna(0)
    denominator = pd.to_numeric(den, errors="coerce").fillna(0)

    return np.where(
        denominator > 0,
        (numerator / denominator * 100).round(0),
        0,
    )


def time_to_seconds(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if ":" not in x:
        return np.nan

    mins, secs = x.split(":")
    return int(mins) * 60 + int(secs)


def run_staged_master_mapper():
    print("========== STAGED MASTER MAPPER ==========")

    staged = pd.read_parquet(STAGED_FIGHT_DETAILS_PATH)
    master = pd.read_parquet(MASTER_PATH)

    print(f"Staged rows: {len(staged)}")
    print(f"Master cols: {len(master.columns)}")

    print()
    print("========== STAGED COLUMNS ==========")
    print(list(staged.columns))

    mapped = pd.DataFrame(columns=master.columns)

    mapped["event_name"] = staged["event_name"]
    mapped["date"] = pd.to_datetime(
        staged["event_date"],
        errors="coerce",
    ).dt.strftime("%-m/%-d/%Y")

    weight_class = first_existing_series(staged, ["weight_class", "division"])
    event_location = first_existing_series(staged, ["event_location", "location"])
    time_format = first_existing_series(staged, ["time_format"])

    mapped["location"] = event_location
    mapped["division"] = weight_class.apply(clean_division)
    mapped["title_fight"] = weight_class.apply(title_fight_flag)
    mapped["total_rounds"] = [
        total_rounds_from_time_format(value, title_fight)
        for value, title_fight in zip(time_format, mapped["title_fight"])
    ]

    if "fight_id" in staged.columns:
        mapped["fight_id"] = staged["fight_id"]
    else:
        mapped["fight_id"] = (
            staged["fight_url"]
            .astype(str)
            .str.rstrip("/")
            .str.split("/")
            .str[-1]
        )

    mapped["method"] = staged["method"]

    mapped["finish_round"] = pd.to_numeric(
        staged["round"],
        errors="coerce",
    )

    mapped["match_time_sec"] = staged["time"].apply(time_to_seconds)
    mapped["referee"] = staged["referee"]

    mapped["r_name"] = staged["red_fighter"]
    mapped["b_name"] = staged["blue_fighter"]

    red_maps = {
        "r_kd": "red_kd",
        "r_sig_str_landed": "red_sig_str_landed",
        "r_sig_str_atmpted": "red_sig_str_attempted",
        "r_total_str_landed": "red_total_str_landed",
        "r_total_str_atmpted": "red_total_str_attempted",
        "r_td_landed": "red_td_landed",
        "r_td_atmpted": "red_td_attempted",
        "r_sub_att": "red_sub_att",
    }

    blue_maps = {
        "b_kd": "blue_kd",
        "b_sig_str_landed": "blue_sig_str_landed",
        "b_sig_str_atmpted": "blue_sig_str_attempted",
        "b_total_str_landed": "blue_total_str_landed",
        "b_total_str_atmpted": "blue_total_str_attempted",
        "b_td_landed": "blue_td_landed",
        "b_td_atmpted": "blue_td_attempted",
        "b_sub_att": "blue_sub_att",
    }

    for target, source in {**red_maps, **blue_maps}.items():
        mapped[target] = pd.to_numeric(staged[source], errors="coerce")

    mapped["r_ctrl"] = staged["red_ctrl"]
    mapped["b_ctrl"] = staged["blue_ctrl"]

    for side in ["r", "b"]:
        for zone in ["head", "body", "leg", "dist", "clinch", "ground"]:
            mapped[f"{side}_{zone}_landed"] = pd.to_numeric(
                staged[f"{side}_{zone}_landed"],
                errors="coerce",
            )
            mapped[f"{side}_{zone}_atmpted"] = pd.to_numeric(
                staged[f"{side}_{zone}_atmpted"],
                errors="coerce",
            )

    mapped["r_sig_str_acc"] = safe_pct(
        mapped["r_sig_str_landed"],
        mapped["r_sig_str_atmpted"],
    )

    mapped["b_sig_str_acc"] = safe_pct(
        mapped["b_sig_str_landed"],
        mapped["b_sig_str_atmpted"],
    )

    mapped["r_td_acc"] = safe_pct(
        mapped["r_td_landed"],
        mapped["r_td_atmpted"],
    )

    mapped["b_td_acc"] = safe_pct(
        mapped["b_td_landed"],
        mapped["b_td_atmpted"],
    )

    mapped["winner"] = np.where(
        staged["red_result"].astype(str).str.lower() == "w",
        staged["red_fighter"],
        np.where(
            staged["blue_result"].astype(str).str.lower() == "w",
            staged["blue_fighter"],
            np.nan,
        ),
    )

    if "event_id" not in staged.columns:
        raise ValueError("Missing event_id column in staged fight details.")

    mapped["event_id"] = staged["event_id"]
    mapped["winner_id"] = np.nan

    mapped["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mapped["run_timestamp"] = datetime.now(timezone.utc)

    mapped = mapped.reindex(columns=master.columns)

    mapped.to_parquet(STAGED_MASTER_ROWS_PATH, index=False)

    audit = pd.DataFrame(
        {
            "column_name": master.columns,
            "non_null_count": [
                mapped[c].notna().sum()
                for c in master.columns
            ],
        }
    )

    audit.to_parquet(STAGED_MASTER_MAPPING_AUDIT_PATH, index=False)

    print()
    print("========== MAPPER SUMMARY ==========")
    print(f"Mapped rows: {len(mapped)}")
    print(f"Mapped cols: {len(mapped.columns)}")
    print(f"Populated cols: {(audit['non_null_count'] > 0).sum()}")

    print()
    print("Saved:", STAGED_MASTER_ROWS_PATH)
    print("Saved:", STAGED_MASTER_MAPPING_AUDIT_PATH)

    return mapped, audit


if __name__ == "__main__":
    run_staged_master_mapper()
