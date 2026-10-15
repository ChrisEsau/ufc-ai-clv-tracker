from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.common.paths import (
    FSR_V2_PREFIGHT_SNAPSHOTS_PATH,
    MASTER_PATH,
)
from pipeline.simulation.event_mc_v1.diagnostics.fresh_100_fight_predictive_replay import (
    select_fresh_cohort,
)


FIGHTS = 500


def is_ko(method) -> bool:
    s = str(method).upper()
    return "KO" in s or "TKO" in s


def safe_auc(y, x):
    y = np.asarray(y, int)
    x = np.asarray(x, float)
    keep = np.isfinite(x)

    if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
        return np.nan

    return roc_auc_score(y[keep], x[keep])


def build_long_master(master):
    rows = []

    for _, r in master.iterrows():
        date = pd.Timestamp(r["date"])
        winner = str(r.get("winner_id", ""))
        ko = is_ko(r.get("method", ""))

        for side, prefix, opp_prefix in (
            ("red", "r", "b"),
            ("blue", "b", "r"),
        ):
            fighter_id = str(r.get(f"{prefix}_id", ""))
            opponent_id = str(r.get(f"{opp_prefix}_id", ""))

            if not fighter_id or fighter_id == "nan":
                continue

            dob = r.get(f"{prefix}_dob", pd.NaT)

            rows.append(
                {
                    "fight_id": str(r["fight_id"]),
                    "event_date": date,
                    "side": side,
                    "fighter_id": fighter_id,
                    "opponent_id": opponent_id,
                    "dob": dob,
                    "actual_win": int(fighter_id == winner),
                    "actual_ko_win": int(ko and fighter_id == winner),
                    "actual_ko_loss": int(ko and fighter_id != winner),
                    "fight_is_ko": int(ko),
                }
            )

    long = pd.DataFrame(rows).sort_values(
        ["fighter_id", "event_date", "fight_id"]
    )

    pieces = []

    for _, g in long.groupby("fighter_id", sort=False):
        g = g.copy()

        prior_ko_dates = (
            g["event_date"]
            .where(g["actual_ko_win"].eq(1))
            .shift()
            .ffill()
        )

        g["last_prior_ko_win_date"] = prior_ko_dates

        g["years_since_prior_ko_win"] = (
            (g["event_date"] - g["last_prior_ko_win_date"])
            .dt.days
            / 365.2425
        )

        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


def print_trait(label, frame, col, ycol="actual_ko_win"):
    y = frame[ycol].astype(int)
    x = frame[col].astype(float)

    pos = x[y.eq(1)]
    neg = x[y.eq(0)]

    print(
        f"{label:<42}"
        f"AUC={safe_auc(y, x):.4f} | "
        f"KO winners={pos.mean():.2f} | "
        f"others={neg.mean():.2f} | "
        f"edge={pos.mean()-neg.mean():+.2f}"
    )


def main():
    print("=" * 145)
    print("FSR V2 — KO PHYSICAL TRAIT AUDIT")
    print("=" * 145)

    cohort, _, metadata = select_fresh_cohort(
        FIGHTS,
        offset=0,
    )

    cohort["fight_id"] = cohort["fight_id"].astype(str)

    fresh_ids = set(cohort["fight_id"])

    master = (
        pd.read_parquet(MASTER_PATH)
        .drop_duplicates("fight_id")
        .copy()
    )

    master["fight_id"] = master["fight_id"].astype(str)

    history = build_long_master(master)

    fsr = pd.read_parquet(
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH
    ).copy()

    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["event_date"] = pd.to_datetime(
        fsr["event_date"]
    ).dt.normalize()

    eval_rows = history[
        history["fight_id"].isin(fresh_ids)
    ].copy()

    eval_rows["event_date"] = pd.to_datetime(
        eval_rows["event_date"]
    ).dt.normalize()

    phys = fsr[
        fsr["fight_id"].isin(fresh_ids)
    ][
        [
            "fight_id",
            "fighter_id",
            "fighter_name",
            "event_date",
            "striking_power",
            "damage_durability",
            "knockdown_resistance",
        ]
    ].copy()

    frame = eval_rows.merge(
        phys,
        on=[
            "fight_id",
            "fighter_id",
            "event_date",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(frame) != 1000:
        raise RuntimeError(
            f"Expected 1000 fresh fighter rows, got {len(frame)}"
        )

    # Opponent physical traits.
    opp = frame[
        [
            "fight_id",
            "fighter_id",
            "striking_power",
            "damage_durability",
            "knockdown_resistance",
        ]
    ].rename(
        columns={
            "fighter_id": "opponent_id",
            "striking_power": "opp_striking_power",
            "damage_durability": "opp_damage_durability",
            "knockdown_resistance": "opp_knockdown_resistance",
        }
    )

    frame = frame.merge(
        opp,
        on=["fight_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )

    frame["age"] = (
        (
            frame["event_date"]
            - pd.to_datetime(frame["dob"])
        ).dt.days
        / 365.2425
    )

    # Simple matchup signals — diagnostic only.
    frame["power_edge"] = (
        frame["striking_power"]
        - frame["opp_striking_power"]
    )

    frame["power_vs_durability"] = (
        frame["striking_power"]
        - frame["opp_damage_durability"]
    )

    frame["power_vs_kd_resistance"] = (
        frame["striking_power"]
        - frame["opp_knockdown_resistance"]
    )

    frame["power_vs_combined_resistance"] = (
        frame["striking_power"]
        - 0.5 * frame["opp_damage_durability"]
        - 0.5 * frame["opp_knockdown_resistance"]
    )

    print()
    print(
        f"Fresh fights: {len(frame)//2} | "
        f"actual KO fights: "
        f"{frame.groupby('fight_id')['fight_is_ko'].max().sum()}"
    )

    print()
    print("=" * 145)
    print("FRESH-500 DISTRIBUTIONS")
    print("=" * 145)

    for col in (
        "striking_power",
        "damage_durability",
        "knockdown_resistance",
    ):
        x = frame[col]

        print(
            f"{col:<28} "
            f"mean={x.mean():.2f} | "
            f"std={x.std():.2f} | "
            f"p10={x.quantile(.10):.2f} | "
            f"p50={x.quantile(.50):.2f} | "
            f"p90={x.quantile(.90):.2f} | "
            f"min={x.min():.2f} | "
            f"max={x.max():.2f}"
        )

    print()
    print("=" * 145)
    print("ACTUAL KO-WINNER DISCRIMINATION")
    print("=" * 145)

    print_trait(
        "Striking power",
        frame,
        "striking_power",
    )

    print_trait(
        "Power edge vs opponent",
        frame,
        "power_edge",
    )

    print_trait(
        "Power - opponent durability",
        frame,
        "power_vs_durability",
    )

    print_trait(
        "Power - opponent KD resistance",
        frame,
        "power_vs_kd_resistance",
    )

    print_trait(
        "Power - combined resistance",
        frame,
        "power_vs_combined_resistance",
    )

    print()
    print("=" * 145)
    print("DEFENSIVE TRAITS — ACTUAL KO LOSSES")
    print("=" * 145)

    # Lower rating should predict KO loss, so negate for AUC.
    frame["negative_durability"] = (
        -frame["damage_durability"]
    )

    frame["negative_kd_resistance"] = (
        -frame["knockdown_resistance"]
    )

    print_trait(
        "Low damage durability",
        frame,
        "negative_durability",
        "actual_ko_loss",
    )

    print_trait(
        "Low knockdown resistance",
        frame,
        "negative_kd_resistance",
        "actual_ko_loss",
    )

    print()
    print("=" * 145)
    print("HIGH-POWER PREVALENCE")
    print("=" * 145)

    for threshold in (
        70,
        75,
        80,
        85,
    ):
        subset = frame[
            frame["striking_power"] >= threshold
        ]

        if len(subset) == 0:
            continue

        print(
            f"POWER >= {threshold}: "
            f"{len(subset)}/{len(frame)} "
            f"({len(subset)/len(frame):.2%}) | "
            f"actual KO-win rate="
            f"{subset['actual_ko_win'].mean():.2%} | "
            f"mean age={subset['age'].mean():.1f}"
        )

    print()
    print("=" * 145)
    print("POWER BY AGE")
    print("=" * 145)

    age_bins = pd.cut(
        frame["age"],
        bins=[
            0,
            27,
            30,
            33,
            36,
            39,
            100,
        ],
        right=False,
    )

    age_table = (
        frame.assign(age_bin=age_bins)
        .groupby(
            "age_bin",
            observed=True,
        )
        .agg(
            fighters=("fighter_id", "size"),
            mean_power=("striking_power", "mean"),
            p75_power=("striking_power", lambda x: x.quantile(.75)),
            ko_win_rate=("actual_ko_win", "mean"),
        )
    )

    print(age_table.to_string())

    print()
    print("=" * 145)
    print("POWER STALENESS — YEARS SINCE PRIOR KO WIN")
    print("=" * 145)

    stale = frame[
        frame["years_since_prior_ko_win"].notna()
    ].copy()

    stale_bins = pd.cut(
        stale["years_since_prior_ko_win"],
        bins=[
            0,
            1,
            2,
            3,
            5,
            100,
        ],
        right=False,
    )

    stale_table = (
        stale.assign(stale_bin=stale_bins)
        .groupby(
            "stale_bin",
            observed=True,
        )
        .agg(
            fighters=("fighter_id", "size"),
            mean_power=("striking_power", "mean"),
            ko_win_rate=("actual_ko_win", "mean"),
            mean_age=("age", "mean"),
        )
    )

    print(stale_table.to_string())

    # -----------------------------------------------------------------
    # Structural monotonicity of published power snapshots.
    # -----------------------------------------------------------------

    print()
    print("=" * 145)
    print("CAREER POWER MONOTONICITY")
    print("=" * 145)

    all_power = fsr[
        [
            "fighter_id",
            "event_date",
            "striking_power",
        ]
    ].dropna().sort_values(
        [
            "fighter_id",
            "event_date",
        ]
    )

    fighters_over_50 = 0
    fighters_with_decline_after_50 = 0
    comparisons = 0
    declines = 0

    for _, g in all_power.groupby(
        "fighter_id",
        sort=False,
    ):
        values = g[
            "striking_power"
        ].to_numpy(float)

        if len(values) < 2:
            continue

        above = np.flatnonzero(
            values > 50.0
        )

        if not len(above):
            continue

        fighters_over_50 += 1

        tail = values[
            above[0]:
        ]

        d = np.diff(tail)

        comparisons += len(d)
        declines += int(
            (d < -1e-9).sum()
        )

        if np.any(
            d < -1e-9
        ):
            fighters_with_decline_after_50 += 1

    print(
        f"fighters ever above 50 power: "
        f"{fighters_over_50}"
    )

    print(
        f"fighters with ANY later power decline: "
        f"{fighters_with_decline_after_50} "
        f"({fighters_with_decline_after_50/max(fighters_over_50,1):.2%})"
    )

    print(
        f"negative snapshot transitions after >50: "
        f"{declines}/{comparisons} "
        f"({declines/max(comparisons,1):.4%})"
    )

    print()
    print("=" * 145)
    print("EXTREME HIGH-POWER NON-KO RESULTS")
    print("=" * 145)

    extreme = frame[
        (frame["striking_power"] >= 75)
        & (frame["actual_ko_win"] == 0)
    ].copy()

    print(
        extreme[
            [
                "event_date",
                "fighter_name",
                "age",
                "striking_power",
                "damage_durability",
                "knockdown_resistance",
                "years_since_prior_ko_win",
                "actual_win",
                "fight_is_ko",
            ]
        ]
        .sort_values(
            [
                "striking_power",
                "age",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(40)
        .to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()
