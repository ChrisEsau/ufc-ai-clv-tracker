"""Historical R1/R2/R3 output decay on matched survivor cohorts."""

from __future__ import annotations

import pandas as pd
import numpy as np

from pipeline.common.paths import ROUND_STATS_PATH
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import observed_duration_seconds


METRICS = [
    "sig_str_attempted",
    "sig_str_landed",
    "total_str_attempted",
    "total_str_landed",
    "td_attempted",
    "td_landed",
    "ctrl_sec",
    "distance_attempted",
    "clinch_attempted",
    "ground_attempted",
]


def round_exposure(total_seconds, round_no):
    start = (round_no - 1) * 300.0
    return max(0.0, min(300.0, total_seconds - start))


def build_rows(cohort, stats):
    fight_ids = set(cohort["fight_id"].astype(str))

    stats = stats[
        stats["fight_id"].astype(str).isin(fight_ids)
        & stats["round"].isin([1, 2, 3])
    ].copy()

    duration = {}

    for _, fight in cohort.iterrows():
        duration[str(fight["fight_id"])] = float(
            observed_duration_seconds(fight)
        )

    stats["fight_id"] = stats["fight_id"].astype(str)

    stats["exposure_seconds"] = stats.apply(
        lambda r: round_exposure(
            duration[r["fight_id"]],
            int(r["round"]),
        ),
        axis=1,
    )

    return stats[stats["exposure_seconds"] > 0].copy()


def round_summary(frame):
    rows = []

    for round_no in (1, 2, 3):
        x = frame[frame["round"] == round_no]

        if x.empty:
            continue

        exposure = x["exposure_seconds"].sum()

        row = {
            "round": round_no,
            "fighter_rounds": len(x),
            "exposure_hours": exposure / 3600.0,
        }

        for metric in METRICS:
            total = x[metric].sum()
            row[metric + "_per5"] = total / exposure * 300.0

        sig_att = x["sig_str_attempted"].sum()
        sig_land = x["sig_str_landed"].sum()

        total_att = x["total_str_attempted"].sum()
        total_land = x["total_str_landed"].sum()

        row["sig_accuracy"] = sig_land / sig_att
        row["total_accuracy"] = total_land / total_att

        rows.append(row)

    return pd.DataFrame(rows)


def matched_fights(frame, max_round):
    eligible = (
        frame.groupby("fight_id")["round"]
        .max()
    )

    ids = set(eligible[eligible >= max_round].index)

    return frame[frame["fight_id"].isin(ids)].copy()


def print_ratios(summary):
    indexed = summary.set_index("round")

    base = indexed.loc[1]

    print("\nROUND RELATIVE TO R1")
    print("-" * 100)

    ratio_metrics = [
        "sig_str_attempted_per5",
        "sig_str_landed_per5",
        "total_str_attempted_per5",
        "total_str_landed_per5",
        "td_attempted_per5",
        "td_landed_per5",
        "ctrl_sec_per5",
        "sig_accuracy",
        "total_accuracy",
    ]

    out = []

    for round_no in indexed.index:
        row = {"round": round_no}

        for metric in ratio_metrics:
            row[metric + "_vs_r1"] = (
                indexed.loc[round_no, metric] / base[metric]
            )

        out.append(row)

    print(
        pd.DataFrame(out).to_string(
            index=False,
            float_format=lambda x: f"{x:8.4f}",
        )
    )


def evaluate(name, cohort, stats):
    frame = build_rows(cohort, stats)

    print("\n" + "=" * 150)
    print(f"{name} — FIGHTS REACHING R2")
    print("=" * 150)

    r2 = round_summary(matched_fights(frame, 2))

    print(
        r2.to_string(
            index=False,
            float_format=lambda x: f"{x:8.4f}",
        )
    )
    print_ratios(r2)

    print("\n" + "=" * 150)
    print(f"{name} — FIGHTS REACHING R3")
    print("=" * 150)

    r3 = round_summary(matched_fights(frame, 3))

    print(
        r3.to_string(
            index=False,
            float_format=lambda x: f"{x:8.4f}",
        )
    )
    print_ratios(r3)


def run():
    train, holdout, _ = temporal_cohorts(100, 50)

    stats = pd.read_parquet(ROUND_STATS_PATH)

    evaluate("TRAIN", train, stats)
    evaluate("HOLDOUT", holdout, stats)


if __name__ == "__main__":
    run()
