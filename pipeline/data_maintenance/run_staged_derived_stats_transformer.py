import pandas as pd
import numpy as np

from pipeline.common.fight_time import clock_time_to_seconds, elapsed_fight_time_seconds
from pipeline.common.paths import (
    STAGED_MASTER_ROWS_PATH,
    STAGED_MASTER_ROWS_ENRICHED_PATH,
    STAGED_DERIVED_STATS_AUDIT_PATH,
)

INPUT_PATH = STAGED_MASTER_ROWS_PATH
OUTPUT_PATH = STAGED_MASTER_ROWS_ENRICHED_PATH
AUDIT_PATH = STAGED_DERIVED_STATS_AUDIT_PATH


def safe_pct(num, den):
    numerator = pd.to_numeric(num, errors="coerce").fillna(0)
    denominator = pd.to_numeric(den, errors="coerce").fillna(0)

    return np.where(
        denominator > 0,
        (numerator / denominator * 100).round(0),
        0,
    )


def time_to_seconds(x):
    """Backward-compatible alias for final-round clock conversion."""
    return clock_time_to_seconds(x)


def build_elapsed_match_time_seconds(df: pd.DataFrame) -> pd.Series:
    """Build total elapsed fight time from finish_round and final-round clock.

    UFCStats stores the displayed fight clock as time elapsed inside the final
    round. The model feature stack needs total cage time:

        ((finish_round - 1) * 300) + final_round_clock_seconds
    """
    if "time" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    final_round_clock_seconds = df["time"].apply(clock_time_to_seconds)
    if "finish_round" not in df.columns:
        return final_round_clock_seconds

    return pd.Series(
        [
            elapsed_fight_time_seconds(finish_round, clock_seconds)
            for finish_round, clock_seconds in zip(df["finish_round"], final_round_clock_seconds)
        ],
        index=df.index,
    )


def run_staged_derived_stats_transformer(debug=True):
    df = pd.read_parquet(INPUT_PATH)

    if "time" in df.columns:
        df["match_time_sec"] = build_elapsed_match_time_seconds(df)

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

    if debug:
        derived_cols = [
            c for c in df.columns
            if c.endswith("_acc") or c.endswith("_per")
        ]

        print()
        print("========== DERIVED STATS SAMPLE ==========")

        debug_cols = [
            "event_name",
            "fight_id",
            "finish_round",
            "time",
            "match_time_sec",
        ] + derived_cols

        debug_cols = [
            c for c in debug_cols
            if c in df.columns
        ]

        print(
            df[debug_cols]
            .head(5)
            .to_string(index=False)
        )

        print()
        print("========== INPUT COLUMN COMPLETENESS ==========")

        for col in [
            "fight_id",
            "finish_round",
            "time",
            "match_time_sec",
            "r_head_landed",
            "r_head_atmpted",
            "r_body_landed",
            "r_body_atmpted",
            "r_leg_landed",
            "r_leg_atmpted",
        ]:
            if col in df.columns:
                print(
                    f"{col:<20} "
                    f"non_null={df[col].notna().sum()} "
                    f"sample={df[col].dropna().head(3).tolist()}"
                )

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

    return df, audit


if __name__ == "__main__":
    run_staged_derived_stats_transformer(debug=True)
