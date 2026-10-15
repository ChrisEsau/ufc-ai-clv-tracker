"""Audit historical finish targets for RFS Monte Carlo V2.

This establishes the empirical outcome distribution for the primary
evaluation cohort:

    both fighters have at least 3 completed prior UFC fights

The output is descriptive only. It does not fit simulator calibration
parameters yet.
"""

from pathlib import Path
import sys

import pandas as pd


# Support:
#     python scripts/audit_rfs_mc_v2_finish_targets.py
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


HISTORY_PATH = (
    REPO_ROOT
    / "data/simulation/rfs_mc_v2_shared_state/"
    / "historical_fighter_state.parquet"
)

MASTER_PATH = (
    REPO_ROOT
    / "data/master/ufc_master.parquet"
)

MIN_PRIOR_FIGHTS = 3


def classify_method(
    method: object,
) -> str:
    """Normalize master outcome method into calibration classes."""

    if method is None or pd.isna(method):
        return "unknown"

    text = str(method).strip().lower()

    # KO and TKO are one simulator method family.
    if (
        "ko" in text
        or "tko" in text
        or "technical knockout" in text
        or "knockout" in text
    ):
        return "ko_tko"

    if "submission" in text:
        return "submission"

    # Everything else that reached an official result without
    # KO/TKO or submission belongs to the scheduled-distance family.
    if (
        "decision" in text
        or "draw" in text
    ):
        return "decision"

    return "other"


def main() -> None:
    history = pd.read_parquet(
        HISTORY_PATH
    )

    master = pd.read_parquet(
        MASTER_PATH
    )

    history["fight_id"] = (
        history["fight_id"].astype(str)
    )

    master["fight_id"] = (
        master["fight_id"].astype(str)
    )

    # ---------------------------------------------------------------
    # Determine the approved >=3-prior-fights historical cohort.
    # ---------------------------------------------------------------

    prior_counts = pd.to_numeric(
        history["rfs_traj_prior_fight_count"],
        errors="coerce",
    )

    eligible_rows = history.loc[
        prior_counts >= MIN_PRIOR_FIGHTS
    ].copy()

    eligible_fights = (
        eligible_rows.groupby(
            "fight_id"
        )
        .filter(
            lambda group: (
                len(group) == 2
                and group["fighter_id"].nunique() == 2
            )
        )
    )

    eligible_ids = (
        eligible_fights["fight_id"]
        .drop_duplicates()
    )

    outcomes = master.loc[
        master["fight_id"].isin(
            eligible_ids
        )
    ].copy()

    if outcomes.empty:
        raise RuntimeError(
            "No eligible historical outcomes found."
        )

    # One authoritative result per fight.
    if outcomes["fight_id"].duplicated().any():
        duplicates = (
            outcomes.loc[
                outcomes["fight_id"].duplicated(
                    keep=False
                ),
                "fight_id",
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Duplicate master fight outcomes found: "
            f"{duplicates[:10]}"
        )

    outcomes["method_class"] = (
        outcomes["method"].map(
            classify_method
        )
    )

    outcomes["finish_round_numeric"] = (
        pd.to_numeric(
            outcomes["finish_round"],
            errors="coerce",
        )
    )

    outcomes["scheduled_rounds_numeric"] = (
        pd.to_numeric(
            outcomes["total_rounds"],
            errors="coerce",
        )
    )

    total = len(outcomes)

    method_counts = (
        outcomes["method_class"]
        .value_counts()
        .reindex(
            [
                "ko_tko",
                "submission",
                "decision",
                "other",
                "unknown",
            ],
            fill_value=0,
        )
    )

    ko_count = int(
        method_counts["ko_tko"]
    )

    sub_count = int(
        method_counts["submission"]
    )

    decision_count = int(
        method_counts["decision"]
    )

    finish_count = (
        ko_count
        + sub_count
    )

    recognized_count = (
        finish_count
        + decision_count
    )

    print("=" * 78)
    print("RFS MONTE CARLO V2 — HISTORICAL FINISH TARGET AUDIT")
    print("=" * 78)

    print(
        "Minimum prior fights per fighter:",
        MIN_PRIOR_FIGHTS,
    )

    print(
        "Eligible fights:",
        f"{total:,}",
    )

    print()

    print("OUTCOME DISTRIBUTION")

    for method_class, count in method_counts.items():
        rate = (
            count / total
            if total
            else 0.0
        )

        print(
            f"  {method_class:12s}"
            f"{int(count):6d}"
            f"  {rate:7.2%}"
        )

    print()

    print(
        "Recognized KO/SUB/decision fights:",
        f"{recognized_count:,}",
    )

    if recognized_count:
        print(
            "KO/TKO rate among recognized:",
            f"{ko_count / recognized_count:.2%}",
        )

        print(
            "Submission rate among recognized:",
            f"{sub_count / recognized_count:.2%}",
        )

        print(
            "Decision rate among recognized:",
            f"{decision_count / recognized_count:.2%}",
        )

        print(
            "Overall finish rate among recognized:",
            f"{finish_count / recognized_count:.2%}",
        )

    # ---------------------------------------------------------------
    # Finish-round distribution.
    # ---------------------------------------------------------------

    finishes = outcomes.loc[
        outcomes["method_class"].isin(
            [
                "ko_tko",
                "submission",
            ]
        )
    ].copy()

    round_counts = (
        finishes["finish_round_numeric"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    print()
    print("FINISH ROUND DISTRIBUTION")

    valid_finish_rounds = int(
        round_counts.sum()
    )

    for round_number, count in round_counts.items():
        rate = (
            count / valid_finish_rounds
            if valid_finish_rounds
            else 0.0
        )

        print(
            f"  Round {round_number}:"
            f" {int(count):5d}"
            f"  {rate:7.2%}"
        )

    # ---------------------------------------------------------------
    # Method-specific round profiles.
    # ---------------------------------------------------------------

    print()
    print("KO/TKO ROUND DISTRIBUTION")

    ko_rounds = (
        outcomes.loc[
            outcomes["method_class"] == "ko_tko",
            "finish_round_numeric",
        ]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    ko_round_total = int(
        ko_rounds.sum()
    )

    for round_number, count in ko_rounds.items():
        print(
            f"  Round {round_number}:"
            f" {int(count):5d}"
            f"  {count / ko_round_total:7.2%}"
        )

    print()
    print("SUBMISSION ROUND DISTRIBUTION")

    sub_rounds = (
        outcomes.loc[
            outcomes["method_class"] == "submission",
            "finish_round_numeric",
        ]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    sub_round_total = int(
        sub_rounds.sum()
    )

    for round_number, count in sub_rounds.items():
        print(
            f"  Round {round_number}:"
            f" {int(count):5d}"
            f"  {count / sub_round_total:7.2%}"
        )

    # ---------------------------------------------------------------
    # Scheduled-distance mix.
    # ---------------------------------------------------------------

    print()
    print("SCHEDULED ROUND MIX")

    scheduled_counts = (
        outcomes[
            "scheduled_rounds_numeric"
        ]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    for scheduled_rounds, count in scheduled_counts.items():
        print(
            f"  {scheduled_rounds}-round fights:"
            f" {int(count):5d}"
            f"  {count / total:7.2%}"
        )

    print()
    print("=" * 78)
    print("FINISH CALIBRATION TARGET AUDIT PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
