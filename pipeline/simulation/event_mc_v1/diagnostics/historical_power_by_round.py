"""Historical round-by-round effective striking-power proxies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from pipeline.common.paths import ROUND_STATS_PATH
from .phase7b_kd_calibration import temporal_cohorts


def prepare(cohort, stats):
    fight_ids = set(cohort["fight_id"].astype(str))

    x = stats[
        stats["fight_id"].astype(str).isin(fight_ids)
        & stats["round"].isin([1, 2, 3])
    ].copy()

    x["fight_id"] = x["fight_id"].astype(str)

    meta = cohort[
        ["fight_id", "winner", "method", "finish_round"]
    ].copy()

    meta["fight_id"] = meta["fight_id"].astype(str)
    meta["finish_round"] = pd.to_numeric(
        meta["finish_round"], errors="coerce"
    )

    x = x.merge(
        meta,
        on="fight_id",
        how="left",
        validate="many_to_one",
    )

    # Prior knockdowns inflicted by this fighter in earlier rounds.
    x = x.sort_values(["fight_id", "fighter_id", "round"])

    x["prior_kd"] = (
        x.groupby(["fight_id", "fighter_id"])["kd"]
        .cumsum()
        .shift(fill_value=0)
    )

    x["first_kd_eligible"] = x["prior_kd"].eq(0)

    method = x["method"].astype(str).str.upper()

    x["ko_win"] = (
        (method.str.contains("KO") | method.str.contains("TKO"))
        & x["round"].eq(x["finish_round"])
        & x["fighter_name"].astype(str).eq(x["winner"].astype(str))
    ).astype(int)

    return x


def summarize(name, cohort, stats):
    x = prepare(cohort, stats)

    rows = []

    for r in (1, 2, 3):
        q = x[x["round"] == r]

        sig = q["sig_str_landed"].sum()
        head = q["head_landed"].sum()
        kd = q["kd"].sum()
        ko = q["ko_win"].sum()

        eligible = q[q["first_kd_eligible"]]
        eligible_head = eligible["head_landed"].sum()
        eligible_sig = eligible["sig_str_landed"].sum()
        eligible_kd = eligible["kd"].sum()

        rows.append({
            "round": r,
            "fighter_rounds": len(q),
            "sig_landed": sig,
            "head_landed": head,
            "knockdowns": kd,
            "ko_tko_wins": ko,

            "kd_per_100_sig": kd / sig * 100 if sig else np.nan,
            "kd_per_100_head": kd / head * 100 if head else np.nan,

            "first_kd_eligible_rounds": len(eligible),
            "first_kd_per_100_sig": (
                eligible_kd / eligible_sig * 100
                if eligible_sig else np.nan
            ),
            "first_kd_per_100_head": (
                eligible_kd / eligible_head * 100
                if eligible_head else np.nan
            ),

            "ko_per_100_sig": ko / sig * 100 if sig else np.nan,
            "ko_per_100_head": ko / head * 100 if head else np.nan,
        })

    out = pd.DataFrame(rows)

    base = out.iloc[0]

    for col in (
        "kd_per_100_sig",
        "kd_per_100_head",
        "first_kd_per_100_sig",
        "first_kd_per_100_head",
        "ko_per_100_sig",
        "ko_per_100_head",
    ):
        out[col + "_vs_r1"] = out[col] / base[col]

    print("\n" + "=" * 170)
    print(name)
    print("=" * 170)

    print(
        out.to_string(
            index=False,
            float_format=lambda z: f"{z:8.4f}",
        )
    )


def main():
    # No limits: use the entire temporal cohorts.
    train, holdout, _ = temporal_cohorts(None, None)

    stats = pd.read_parquet(ROUND_STATS_PATH)

    print(f"TRAIN fights:   {len(train):,}")
    print(f"HOLDOUT fights: {len(holdout):,}")

    summarize("TRAIN — HISTORICAL POWER PROXIES", train, stats)
    summarize("HOLDOUT — HISTORICAL POWER PROXIES", holdout, stats)


if __name__ == "__main__":
    main()
