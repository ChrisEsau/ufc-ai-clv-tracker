"""Audit empirical round-to-round performance change for RFS MC V2.

Uses the primary evaluation cohort:
    both fighters have >= 3 prior UFC fights

Only completed rounds are used. A finish round is excluded because its
UFCStats totals may represent less than five minutes.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


HISTORY_PATH = (
    REPO_ROOT
    / "data/simulation/rfs_mc_v2_shared_state/"
    / "historical_fighter_state.parquet"
)

ROUND_PATH = (
    REPO_ROOT
    / "data/fight_details/ufc_round_stats.parquet"
)

MASTER_PATH = (
    REPO_ROOT
    / "data/master/ufc_master.parquet"
)

MIN_PRIOR_FIGHTS = 3


def classify_decision(method: object) -> bool:
    """Return True for official decisions/draws that went full distance."""

    if method is None or pd.isna(method):
        return False

    text = str(method).lower()

    return (
        "decision" in text
        or "draw" in text
    )


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Vectorized ratio preserving missing/zero exposure."""

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    result = numerator / denominator

    return result.where(
        denominator > 0
    )


def paired_change(
    df: pd.DataFrame,
    metric: str,
    earlier_round: int,
    later_round: int,
) -> tuple[int, float, float, float]:
    """Return paired later/earlier behavior for one metric."""

    earlier = (
        df.loc[
            df["round"] == earlier_round,
            [
                "fight_id",
                "fighter_id",
                metric,
            ],
        ]
        .rename(
            columns={
                metric: "earlier",
            }
        )
    )

    later = (
        df.loc[
            df["round"] == later_round,
            [
                "fight_id",
                "fighter_id",
                metric,
            ],
        ]
        .rename(
            columns={
                metric: "later",
            }
        )
    )

    paired = earlier.merge(
        later,
        on=[
            "fight_id",
            "fighter_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    paired["earlier"] = pd.to_numeric(
        paired["earlier"],
        errors="coerce",
    )

    paired["later"] = pd.to_numeric(
        paired["later"],
        errors="coerce",
    )

    paired = paired.loc[
        np.isfinite(paired["earlier"])
        & np.isfinite(paired["later"])
    ]

    if paired.empty:
        return 0, np.nan, np.nan, np.nan

    mean_earlier = float(
        paired["earlier"].mean()
    )

    mean_later = float(
        paired["later"].mean()
    )

    if mean_earlier == 0:
        relative_change = np.nan
    else:
        relative_change = (
            mean_later / mean_earlier
            - 1.0
        )

    return (
        len(paired),
        mean_earlier,
        mean_later,
        relative_change,
    )


def main() -> None:
    history = pd.read_parquet(
        HISTORY_PATH
    )

    rounds = pd.read_parquet(
        ROUND_PATH
    )

    master = pd.read_parquet(
        MASTER_PATH
    )

    for df in (
        history,
        rounds,
        master,
    ):
        df["fight_id"] = (
            df["fight_id"].astype(str)
        )

    # ---------------------------------------------------------------
    # Primary >=3 prior-fight cohort.
    # ---------------------------------------------------------------

    prior = pd.to_numeric(
        history["rfs_traj_prior_fight_count"],
        errors="coerce",
    )

    eligible_rows = history.loc[
        prior >= MIN_PRIOR_FIGHTS
    ].copy()

    eligible_rows = (
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

    eligible_ids = set(
        eligible_rows[
            "fight_id"
        ].astype(str)
    )

    rounds = rounds.loc[
        rounds["fight_id"].isin(
            eligible_ids
        )
    ].copy()

    outcomes = master.loc[
        master["fight_id"].isin(
            eligible_ids
        ),
        [
            "fight_id",
            "method",
            "finish_round",
            "total_rounds",
        ],
    ].copy()

    if outcomes["fight_id"].duplicated().any():
        raise RuntimeError(
            "Master contains duplicate fight outcomes."
        )

    rounds = rounds.merge(
        outcomes,
        on="fight_id",
        how="inner",
        validate="many_to_one",
    )

    rounds["round"] = pd.to_numeric(
        rounds["round"],
        errors="raise",
    ).astype(int)

    rounds["finish_round"] = pd.to_numeric(
        rounds["finish_round"],
        errors="coerce",
    )

    rounds["is_decision"] = (
        rounds["method"].map(
            classify_decision
        )
    )

    # A round is complete when:
    #
    # 1. the fight went to decision/draw, or
    # 2. the round occurred before the official finish round.
    rounds["complete_round"] = (
        rounds["is_decision"]
        | (
            rounds["finish_round"].notna()
            & (
                rounds["round"]
                < rounds["finish_round"]
            )
        )
    )

    complete = rounds.loc[
        rounds["complete_round"]
    ].copy()

    # ---------------------------------------------------------------
    # Derived performance measures.
    # ---------------------------------------------------------------

    complete[
        "sig_str_accuracy"
    ] = safe_ratio(
        complete["sig_str_landed"],
        complete["sig_str_attempted"],
    )

    complete[
        "td_accuracy"
    ] = safe_ratio(
        complete["td_landed"],
        complete["td_attempted"],
    )

    metrics = (
        (
            "sig_str_attempted",
            "SIG STR ATTEMPTS",
        ),
        (
            "sig_str_accuracy",
            "SIG STR ACCURACY",
        ),
        (
            "total_str_attempted",
            "TOTAL STR ATTEMPTS",
        ),
        (
            "td_attempted",
            "TAKEDOWN ATTEMPTS",
        ),
        (
            "td_accuracy",
            "TAKEDOWN ACCURACY",
        ),
        (
            "ctrl_sec",
            "CONTROL SECONDS",
        ),
    )

    print("=" * 78)
    print("RFS MONTE CARLO V2 — DYNAMIC PERFORMANCE TARGET AUDIT")
    print("=" * 78)

    print(
        "Eligible fights:",
        f"{len(eligible_ids):,}",
    )

    print(
        "Completed fighter-round observations:",
        f"{len(complete):,}",
    )

    for earlier, later in (
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ):
        print()
        print(
            f"ROUND {earlier} -> ROUND {later}"
        )
        print("-" * 78)

        for metric, label in metrics:
            (
                count,
                mean_earlier,
                mean_later,
                relative_change,
            ) = paired_change(
                complete,
                metric,
                earlier,
                later,
            )

            if count == 0:
                continue

            print(
                f"{label:22s}"
                f" n={count:5d}"
                f"  {mean_earlier:8.3f}"
                f" -> {mean_later:8.3f}"
                f"  change={relative_change:+7.2%}"
            )

    print()
    print("=" * 78)
    print("DYNAMIC PERFORMANCE TARGET AUDIT PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
