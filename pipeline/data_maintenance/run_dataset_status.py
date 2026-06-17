# ============================================================
# run_dataset_status.py
# Build dataset status artifact for Data Maintenance dashboard
# ============================================================

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    MASTER_PATH,
    DATASET_STATUS_PATH,
    DATASET_EVENT_STATUS_PATH,
)

MASTER_DATASET_PATH = MASTER_PATH
DATASET_STATUS_OUTPUT = DATASET_STATUS_PATH
DATASET_EVENT_STATUS_OUTPUT = DATASET_EVENT_STATUS_PATH


def first_existing(df: pd.DataFrame, possible_cols: list[str]) -> str | None:
    for col in possible_cols:
        if col in df.columns:
            return col
    return None


def run_dataset_status() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_timestamp = datetime.now(timezone.utc).isoformat()

    if not Path(MASTER_DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Missing master dataset: {MASTER_DATASET_PATH}"
        )

    df = pd.read_parquet(MASTER_DATASET_PATH)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    date_col = first_existing(df, [
        "date",
        "event_date",
        "fight_date",
        "Date",
    ])

    event_col = first_existing(df, [
        "event_name",
        "event",
        "Event",
    ])

    red_fighter_col = first_existing(df, [
        "red_fighter",
        "r_fighter",
        "r_name",
        "R_fighter",
        "R",
    ])

    blue_fighter_col = first_existing(df, [
        "blue_fighter",
        "b_fighter",
        "b_name",
        "B_fighter",
        "B",
    ])

    red_id_col = first_existing(df, [
        "red_fighter_id",
        "r_fighter_id",
        "R_fighter_id",
    ])

    blue_id_col = first_existing(df, [
        "blue_fighter_id",
        "b_fighter_id",
        "B_fighter_id",
    ])

    result_col = first_existing(df, [
        "winner",
        "Winner",
        "result",
        "Result",
    ])

    raw_dates = df["date"].astype(str)

    parsed_dates = pd.to_datetime(
        raw_dates,
        errors="coerce",
    )

    invalid_date_cols = [
        c for c in ["event_name", "date", "fight_id", "r_name", "b_name"]
        if c in df.columns
    ]
    invalid_dates = df[parsed_dates.isna()][invalid_date_cols]

    print()
    print("========== INVALID DATE ROWS ==========")

    if invalid_dates.empty:
        print("None")
    else:
        print(invalid_dates.head(50).to_string(index=False))

    status = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "dataset_path": str(MASTER_PATH),
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "date_column": date_col,
        "event_column": event_col,
        "red_fighter_column": red_fighter_col,
        "blue_fighter_column": blue_fighter_col,
        "red_id_column": red_id_col,
        "blue_id_column": blue_id_col,
        "result_column": result_col,
    }

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        status["latest_fight_date"] = df[date_col].max()
        status["earliest_fight_date"] = df[date_col].min()
        status["invalid_date_count"] = int(df[date_col].isna().sum())
    else:
        status["latest_fight_date"] = None
        status["earliest_fight_date"] = None
        status["invalid_date_count"] = None

    if event_col:
        status["unique_events"] = int(df[event_col].nunique(dropna=True))
    else:
        status["unique_events"] = None

    fighter_cols = [
        c for c in [red_fighter_col, blue_fighter_col]
        if c is not None
    ]

    if fighter_cols:
        fighters = pd.concat(
            [df[c].dropna().astype(str) for c in fighter_cols],
            ignore_index=True,
        )
        status["unique_fighters"] = int(fighters.nunique())
    else:
        status["unique_fighters"] = None

    if result_col:
        status["missing_result_count"] = int(df[result_col].isna().sum())
    else:
        status["missing_result_count"] = None

    if red_id_col:
        status["missing_red_id_count"] = int(
            df[red_id_col].isna().sum()
            + (df[red_id_col].astype(str).str.strip() == "").sum()
        )
    else:
        status["missing_red_id_count"] = None

    if blue_id_col:
        status["missing_blue_id_count"] = int(
            df[blue_id_col].isna().sum()
            + (df[blue_id_col].astype(str).str.strip() == "").sum()
        )
    else:
        status["missing_blue_id_count"] = None

    duplicate_key_cols = [
        c for c in [
            date_col,
            event_col,
            red_fighter_col,
            blue_fighter_col,
        ]
        if c is not None
    ]

    if duplicate_key_cols:
        status["duplicate_fight_count"] = int(
            df.duplicated(subset=duplicate_key_cols).sum()
        )
    else:
        status["duplicate_fight_count"] = None

    if date_col and event_col:
        latest_date = df[date_col].max()

        latest_event_rows = df[df[date_col] == latest_date]

        if not latest_event_rows.empty:
            status["latest_event_name"] = (
                latest_event_rows[event_col]
                .dropna()
                .astype(str)
                .mode()
                .iloc[0]
            )
            status["latest_event_fight_count"] = len(latest_event_rows)
        else:
            status["latest_event_name"] = None
            status["latest_event_fight_count"] = None
    else:
        status["latest_event_name"] = None
        status["latest_event_fight_count"] = None

    status_df = pd.DataFrame([status])

    status_df.to_parquet(
        DATASET_STATUS_OUTPUT,
        index=False,
    )

    print("Saved dataset status:", DATASET_STATUS_OUTPUT)

    if date_col and event_col:
        event_status = (
            df.groupby([event_col, date_col], dropna=False)
            .size()
            .reset_index(name="fight_count")
            .rename(
                columns={
                    event_col: "event_name",
                    date_col: "event_date",
                }
            )
            .sort_values("event_date", ascending=False)
            .reset_index(drop=True)
        )

        event_status["run_id"] = run_id
        event_status["run_timestamp"] = run_timestamp

    else:
        event_status = pd.DataFrame()

    event_status.to_parquet(
        DATASET_EVENT_STATUS_OUTPUT,
        index=False,
    )

    print("Saved event status:", DATASET_EVENT_STATUS_OUTPUT)

    print("========== DATASET STATUS SUMMARY ==========")
    for k, v in status.items():
        print(f"{k}: {v}")

    return status_df, event_status


def main() -> None:
    run_dataset_status()


if __name__ == "__main__":
    main()
