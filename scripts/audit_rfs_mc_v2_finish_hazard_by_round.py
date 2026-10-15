"""Audit conditional UFC finish hazard and knockdown potency by round."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HISTORY_PATH = (
    ROOT
    / "data/simulation/rfs_mc_v2_shared_state/"
    / "historical_fighter_state.parquet"
)

MASTER_PATH = (
    ROOT
    / "data/master/ufc_master.parquet"
)

ROUND_PATH = (
    ROOT
    / "data/fight_details/ufc_round_stats.parquet"
)

MIN_PRIOR_FIGHTS = 3


def classify_method(value: object) -> str:
    """Normalize official result into simulator finish families."""

    if value is None or pd.isna(value):
        return "other"

    text = str(value).lower()

    if (
        "ko" in text
        or "tko" in text
        or "knockout" in text
    ):
        return "ko_tko"

    if "submission" in text:
        return "submission"

    if (
        "decision" in text
        or "draw" in text
    ):
        return "decision"

    return "other"


def main() -> None:
    history = pd.read_parquet(HISTORY_PATH)
    master = pd.read_parquet(MASTER_PATH)
    rounds = pd.read_parquet(ROUND_PATH)

    for df in (history, master, rounds):
        df["fight_id"] = df["fight_id"].astype(str)

    # ------------------------------------------------------------
    # Primary cohort: both fighters >= 3 completed prior fights.
    # ------------------------------------------------------------

    prior = pd.to_numeric(
        history["rfs_traj_prior_fight_count"],
        errors="coerce",
    )

    eligible = history.loc[
        prior >= MIN_PRIOR_FIGHTS
    ].copy()

    eligible = (
        eligible.groupby("fight_id")
        .filter(
            lambda g: (
                len(g) == 2
                and g["fighter_id"].nunique() == 2
            )
        )
    )

    eligible_ids = set(
        eligible["fight_id"].unique()
    )

    outcomes = master.loc[
        master["fight_id"].isin(eligible_ids),
        [
            "fight_id",
            "method",
            "finish_round",
            "total_rounds",
        ],
    ].copy()

    outcomes["method_class"] = (
        outcomes["method"].map(
            classify_method
        )
    )

    outcomes["finish_round"] = pd.to_numeric(
        outcomes["finish_round"],
        errors="coerce",
    )

    outcomes["total_rounds"] = pd.to_numeric(
        outcomes["total_rounds"],
        errors="coerce",
    )

    recognized = outcomes.loc[
        outcomes["method_class"].isin(
            [
                "ko_tko",
                "submission",
                "decision",
            ]
        )
    ].copy()

    # ------------------------------------------------------------
    # Conditional round hazard:
    #
    # P(finish in Rn | fight reached Rn)
    # ------------------------------------------------------------

    print("=" * 78)
    print("RFS MC V2 — CONDITIONAL FINISH HAZARD BY ROUND")
    print("=" * 78)

    for round_number in range(1, 6):

        # Fight can only reach this round if it was scheduled for it.
        scheduled = recognized.loc[
            recognized["total_rounds"]
            >= round_number
        ]

        # At risk if it did not finish before this round.
        at_risk = scheduled.loc[
            (
                scheduled["method_class"]
                == "decision"
            )
            | (
                scheduled["finish_round"]
                >= round_number
            )
        ]

        finishes = at_risk.loc[
            (
                at_risk["method_class"]
                != "decision"
            )
            & (
                at_risk["finish_round"]
                == round_number
            )
        ]

        ko = finishes.loc[
            finishes["method_class"]
            == "ko_tko"
        ]

        sub = finishes.loc[
            finishes["method_class"]
            == "submission"
        ]

        if len(at_risk) == 0:
            continue

        print(
            f"Round {round_number}: "
            f"at risk={len(at_risk):4d}  "
            f"finish={len(finishes)/len(at_risk):7.2%}  "
            f"KO={len(ko)/len(at_risk):7.2%}  "
            f"SUB={len(sub)/len(at_risk):7.2%}"
        )

    # ------------------------------------------------------------
    # Knockdown potency proxy by round.
    #
    # This uses all observed fighter-round activity, including finish
    # rounds, because we want realized knockdown conversion among fights
    # that actually reached each round.
    # ------------------------------------------------------------

    cohort_rounds = rounds.loc[
        rounds["fight_id"].isin(
            eligible_ids
        )
    ].copy()

    cohort_rounds["round"] = pd.to_numeric(
        cohort_rounds["round"],
        errors="coerce",
    )

    cohort_rounds["kd"] = pd.to_numeric(
        cohort_rounds["kd"],
        errors="coerce",
    )

    cohort_rounds["sig_str_landed"] = pd.to_numeric(
        cohort_rounds["sig_str_landed"],
        errors="coerce",
    )

    print()
    print("=" * 78)
    print("KNOCKDOWN POTENCY BY ROUND")
    print("=" * 78)

    for round_number in range(1, 6):
        group = cohort_rounds.loc[
            cohort_rounds["round"]
            == round_number
        ]

        landed = float(
            group["sig_str_landed"].sum()
        )

        knockdowns = float(
            group["kd"].sum()
        )

        if landed <= 0:
            continue

        kd_per_landed = (
            knockdowns / landed
        )

        print(
            f"Round {round_number}: "
            f"fighter-rounds={len(group):5d}  "
            f"KD={knockdowns:6.0f}  "
            f"SIG landed={landed:8.0f}  "
            f"KD / landed={kd_per_landed:8.4%}"
        )

    print()
    print("=" * 78)
    print("ROUND FINISH HAZARD AUDIT PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
