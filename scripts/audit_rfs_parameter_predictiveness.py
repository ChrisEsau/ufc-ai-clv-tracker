"""Audit how RFS MC V2 simulator parameters relate to historical winners.

Purpose
-------
This is a shadow diagnostic only. It does not modify simulator behavior.

For every eligible historical matchup:

1. Load the leakage-safe pre-fight fighter profiles.
2. Build the population using ONLY fights before that matchup date.
3. Resolve the same 37 calibrated parameters used by RFS MC V2.
4. Compare Red vs Blue parameter values to the observed winner.
5. Measure whether parameter advantages historically correspond to winning.

Important
---------
Fights after 2024-12-31 are deliberately excluded so that 2025+ can remain
an untouched chronological holdout for later simulator validation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.simulation.rfs_mc_v2_shared_state.historical_matchup_loader import (
    HistoricalMatchupLoadError,
    load_historical_matchup,
)
from pipeline.simulation.rfs_mc_v2_shared_state.rfs_parameter_resolver import (
    ALL_TARGETS,
    RFSParameterResolutionError,
    _calibration_family,
    resolve_fighter_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

SHADOW_HISTORY_PATH = (
    REPO_ROOT
    / "data/simulation/rfs_mc_v2_shared_state/historical_fighter_state.parquet"
)

CANONICAL_HISTORY_PATH = (
    REPO_ROOT
    / "data/features/round_fighter_state_history.parquet"
)

MASTER_PATH = (
    REPO_ROOT
    / "data/master/ufc_master.parquet"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "data/simulation/rfs_mc_v2_shared_state"
)

MATCHUP_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_parameter_matchups_v1.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_parameter_predictiveness_v1.csv"
)

FAMILY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_parameter_family_summary_v1.csv"
)


# Primary historical cohort.
MIN_PRIOR_FIGHTS = 3

# Chronological calibration boundaries.
TRAIN_END = pd.Timestamp("2022-12-31")
DEVELOPMENT_END = pd.Timestamp("2024-12-31")


def _history_path() -> Path:
    """Prefer simulator shadow history, with canonical RFS as fallback."""

    if SHADOW_HISTORY_PATH.exists():
        return SHADOW_HISTORY_PATH

    return CANONICAL_HISTORY_PATH


def _method_group(method: object) -> str:
    """Collapse observed UFC result method into broad method families."""

    value = str(method or "").strip().upper()

    if "KO" in value or "TKO" in value:
        return "KO_TKO"

    if "SUB" in value:
        return "SUB"

    if "DEC" in value:
        return "DEC"

    return "OTHER"


def _split_name(date: pd.Timestamp) -> str:
    """Return chronological development split."""

    if date <= TRAIN_END:
        return "TRAIN"

    if date <= DEVELOPMENT_END:
        return "DEVELOPMENT"

    return "HOLDOUT"


def _finite(value: Any) -> float | None:
    """Return a finite float or None."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def _auc_from_scores(
    outcome: pd.Series,
    score: pd.Series,
) -> float:
    """Compute binary ROC AUC using the rank-sum definition.

    This avoids adding a dependency on sklearn for this diagnostic.
    """

    frame = pd.DataFrame(
        {
            "outcome": pd.to_numeric(outcome, errors="coerce"),
            "score": pd.to_numeric(score, errors="coerce"),
        }
    ).dropna()

    if frame.empty:
        return float("nan")

    positives = frame["outcome"] == 1
    negatives = frame["outcome"] == 0

    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = frame["score"].rank(method="average")

    positive_rank_sum = float(ranks.loc[positives].sum())

    u_statistic = (
        positive_rank_sum
        - n_pos * (n_pos + 1) / 2.0
    )

    return float(
        u_statistic
        / (n_pos * n_neg)
    )


def _parameter_metrics(
    frame: pd.DataFrame,
    target: str,
    prefix: str,
) -> dict[str, object]:
    """Calculate winner-association metrics for one target and cohort."""

    red_col = f"red__{target}"
    blue_col = f"blue__{target}"
    diff_col = f"diff__{target}"
    aligned_col = f"winner_diff__{target}"

    work = frame[
        [
            "red_win",
            red_col,
            blue_col,
            diff_col,
            aligned_col,
        ]
    ].copy()

    for column in work.columns:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work = work.dropna()

    if work.empty:
        return {
            f"{prefix}_fights": 0,
            f"{prefix}_higher_value_win_rate": np.nan,
            f"{prefix}_auc": np.nan,
            f"{prefix}_spearman": np.nan,
            f"{prefix}_winner_diff_mean": np.nan,
            f"{prefix}_winner_diff_median": np.nan,
            f"{prefix}_winner_diff_effect": np.nan,
        }

    non_ties = work[
        work[diff_col] != 0
    ].copy()

    if non_ties.empty:
        higher_value_win_rate = np.nan
    else:
        higher_value_won = np.where(
            non_ties[diff_col] > 0,
            non_ties["red_win"] == 1,
            non_ties["red_win"] == 0,
        )

        higher_value_win_rate = float(
            np.mean(higher_value_won)
        )

    auc = _auc_from_scores(
        work["red_win"],
        work[diff_col],
    )

    if (
        work[diff_col].nunique() > 1
        and work["red_win"].nunique() > 1
    ):
        spearman = float(
            work[diff_col].corr(
                work["red_win"],
                method="spearman",
            )
        )
    else:
        spearman = np.nan

    winner_diff_mean = float(
        work[aligned_col].mean()
    )

    winner_diff_median = float(
        work[aligned_col].median()
    )

    diff_std = float(
        work[diff_col].std(ddof=0)
    )

    if diff_std > 0:
        winner_diff_effect = (
            winner_diff_mean
            / diff_std
        )
    else:
        winner_diff_effect = np.nan

    return {
        f"{prefix}_fights": len(work),
        f"{prefix}_higher_value_win_rate": higher_value_win_rate,
        f"{prefix}_auc": auc,
        f"{prefix}_spearman": spearman,
        f"{prefix}_winner_diff_mean": winner_diff_mean,
        f"{prefix}_winner_diff_median": winner_diff_median,
        f"{prefix}_winner_diff_effect": winner_diff_effect,
    }


def _observed_direction(
    train_auc: float,
    development_auc: float,
) -> str:
    """Classify historical direction without assuming higher is always better."""

    values = [
        value
        for value in (
            train_auc,
            development_auc,
        )
        if np.isfinite(value)
    ]

    if not values:
        return "UNKNOWN"

    mean_auc = float(
        np.mean(values)
    )

    if mean_auc >= 0.525:
        return "HIGHER_HELPFUL"

    if mean_auc <= 0.475:
        return "LOWER_HELPFUL"

    return "WEAK"


def main() -> None:
    """Build historical parameter/winner audit."""

    history_path = _history_path()

    print("=" * 78)
    print("RFS MC V2 HISTORICAL PARAMETER PREDICTIVENESS AUDIT")
    print("=" * 78)
    print()
    print("History :", history_path)
    print("Outcomes:", MASTER_PATH)
    print("Targets :", len(ALL_TARGETS))
    print("Minimum prior fights:", MIN_PRIOR_FIGHTS)
    print("Calibration cutoff  :", DEVELOPMENT_END.date())
    print("2025+                : RESERVED HOLDOUT")
    print()

    history = pd.read_parquet(
        history_path
    )

    outcomes = pd.read_parquet(
        MASTER_PATH
    )

    history = history.copy()

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce",
    )

    if "fight_id" not in history.columns:
        raise RuntimeError(
            "Historical RFS artifact has no fight_id column."
        )

    if "rfs_traj_prior_fight_count" not in history.columns:
        raise RuntimeError(
            "Historical RFS artifact is missing "
            "rfs_traj_prior_fight_count."
        )

    # Derive one chronological candidate record per historical fight.
    fight_index = (
        history[
            [
                "fight_id",
                "date",
            ]
        ]
        .dropna(
            subset=[
                "fight_id",
                "date",
            ]
        )
        .drop_duplicates(
            subset=["fight_id"]
        )
        .sort_values(
            [
                "date",
                "fight_id",
            ]
        )
        .reset_index(drop=True)
    )

    holdout_candidates = int(
        (
            fight_index["date"]
            > DEVELOPMENT_END
        ).sum()
    )

    # Do not resolve the reserved holdout.
    fight_index = fight_index.loc[
        fight_index["date"]
        <= DEVELOPMENT_END
    ].copy()

    rows: list[dict[str, object]] = []

    skipped_loader = 0
    skipped_resolver = 0
    skipped_outcome = 0

    for sequence, record in enumerate(
        fight_index.itertuples(index=False),
        start=1,
    ):
        fight_id = str(
            record.fight_id
        )

        try:
            matchup = load_historical_matchup(
                history,
                outcomes,
                fight_id,
                min_prior_fights=MIN_PRIOR_FIGHTS,
            )
        except (
            HistoricalMatchupLoadError,
            ValueError,
            KeyError,
        ):
            skipped_loader += 1
            continue

        red_id = str(
            matchup.red.fighter_id
        )

        blue_id = str(
            matchup.blue.fighter_id
        )

        winner_id = (
            None
            if matchup.actual.winner_id is None
            else str(matchup.actual.winner_id)
        )

        if winner_id not in {
            red_id,
            blue_id,
        }:
            skipped_outcome += 1
            continue

        # Leakage-safe population:
        # strictly before this historical fight date.
        population = history.loc[
            history["date"]
            < matchup.date
        ].copy()

        prior_counts = pd.to_numeric(
            population[
                "rfs_traj_prior_fight_count"
            ],
            errors="coerce",
        )

        population = population.loc[
            prior_counts > 0
        ].copy()

        try:
            red_parameters = resolve_fighter_parameters(
                profile=matchup.red.features,
                prior_fight_count=(
                    matchup.red.prior_fight_count
                ),
                population_history=population,
            )

            blue_parameters = resolve_fighter_parameters(
                profile=matchup.blue.features,
                prior_fight_count=(
                    matchup.blue.prior_fight_count
                ),
                population_history=population,
            )

        except (
            RFSParameterResolutionError,
            ValueError,
            KeyError,
        ):
            skipped_resolver += 1
            continue

        red_win = int(
            winner_id == red_id
        )

        method_group = _method_group(
            matchup.actual.method
        )

        row: dict[str, object] = {
            "fight_id": matchup.fight_id,
            "date": matchup.date,
            "split": _split_name(
                matchup.date
            ),
            "event_name": matchup.event_name,
            "division": matchup.division,
            "scheduled_rounds": matchup.scheduled_rounds,
            "red_fighter_id": red_id,
            "red_fighter_name": matchup.red.fighter_name,
            "blue_fighter_id": blue_id,
            "blue_fighter_name": matchup.blue.fighter_name,
            "winner_id": winner_id,
            "red_win": red_win,
            "method": matchup.actual.method,
            "method_group": method_group,
            "finish_round": matchup.actual.finish_round,
            "red_prior_fights": matchup.red.prior_fight_count,
            "blue_prior_fights": matchup.blue.prior_fight_count,
        }

        missing_target = False

        for target in sorted(
            ALL_TARGETS
        ):
            red_estimate = (
                red_parameters
                .estimates
                .get(target)
            )

            blue_estimate = (
                blue_parameters
                .estimates
                .get(target)
            )

            if (
                red_estimate is None
                or blue_estimate is None
            ):
                missing_target = True
                break

            red_value = _finite(
                red_estimate.shrunk_estimate
            )

            blue_value = _finite(
                blue_estimate.shrunk_estimate
            )

            if (
                red_value is None
                or blue_value is None
            ):
                missing_target = True
                break

            difference = (
                red_value
                - blue_value
            )

            winner_difference = (
                difference
                if red_win == 1
                else -difference
            )

            row[
                f"red__{target}"
            ] = red_value

            row[
                f"blue__{target}"
            ] = blue_value

            row[
                f"diff__{target}"
            ] = difference

            row[
                f"winner_diff__{target}"
            ] = winner_difference

        if missing_target:
            skipped_resolver += 1
            continue

        rows.append(
            row
        )

        if (
            len(rows) % 250 == 0
        ):
            print(
                f"Resolved eligible fights: {len(rows):,}"
            )

    if not rows:
        raise RuntimeError(
            "No eligible historical matchup rows were produced."
        )

    matchup_df = pd.DataFrame(
        rows
    ).sort_values(
        [
            "date",
            "fight_id",
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matchup_df.to_csv(
        MATCHUP_OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 78)
    print("MATCHUP DATASET")
    print("=" * 78)
    print("Eligible fights      :", len(matchup_df))
    print("Train fights         :", int((matchup_df["split"] == "TRAIN").sum()))
    print(
        "Development fights   :",
        int(
            (
                matchup_df["split"]
                == "DEVELOPMENT"
            ).sum()
        ),
    )
    print("Reserved 2025+ fights:", holdout_candidates)
    print("Skipped loader       :", skipped_loader)
    print("Skipped resolver     :", skipped_resolver)
    print("Skipped outcome      :", skipped_outcome)

    print()
    print("Actual methods")
    print(
        matchup_df[
            "method_group"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
        .to_string()
    )

    train_df = matchup_df.loc[
        matchup_df["split"]
        == "TRAIN"
    ].copy()

    development_df = matchup_df.loc[
        matchup_df["split"]
        == "DEVELOPMENT"
    ].copy()

    train_dev_df = matchup_df.copy()

    summary_rows: list[
        dict[str, object]
    ] = []

    for target in sorted(
        ALL_TARGETS
    ):
        result: dict[str, object] = {
            "target": target,
            "family": _calibration_family(
                target
            ),
        }

        result.update(
            _parameter_metrics(
                train_df,
                target,
                "train",
            )
        )

        result.update(
            _parameter_metrics(
                development_df,
                target,
                "development",
            )
        )

        result.update(
            _parameter_metrics(
                train_dev_df,
                target,
                "train_dev",
            )
        )

        # Within-method winner ranking.
        for method_name in (
            "KO_TKO",
            "SUB",
            "DEC",
        ):
            method_df = (
                train_dev_df.loc[
                    train_dev_df[
                        "method_group"
                    ]
                    == method_name
                ]
            )

            result.update(
                _parameter_metrics(
                    method_df,
                    target,
                    method_name.lower(),
                )
            )

        train_auc = float(
            result[
                "train_auc"
            ]
        )

        development_auc = float(
            result[
                "development_auc"
            ]
        )

        result[
            "observed_direction"
        ] = _observed_direction(
            train_auc,
            development_auc,
        )

        if (
            np.isfinite(train_auc)
            and np.isfinite(development_auc)
        ):
            train_sign = np.sign(
                train_auc - 0.5
            )

            development_sign = np.sign(
                development_auc - 0.5
            )

            result[
                "train_development_direction_consistent"
            ] = bool(
                train_sign
                == development_sign
            )
        else:
            result[
                "train_development_direction_consistent"
            ] = False

        result[
            "predictive_edge"
        ] = abs(
            float(
                result[
                    "train_dev_auc"
                ]
            )
            - 0.5
        )

        summary_rows.append(
            result
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        [
            "predictive_edge",
            "target",
        ],
        ascending=[
            False,
            True,
        ],
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    family_rows = []

    for family, group in summary_df.groupby(
        "family",
        sort=True,
    ):
        family_rows.append(
            {
                "family": family,
                "targets": len(group),
                "mean_train_dev_auc": float(
                    group[
                        "train_dev_auc"
                    ].mean()
                ),
                "mean_predictive_edge": float(
                    group[
                        "predictive_edge"
                    ].mean()
                ),
                "mean_higher_value_win_rate": float(
                    group[
                        "train_dev_higher_value_win_rate"
                    ].mean()
                ),
                "higher_helpful_targets": int(
                    (
                        group[
                            "observed_direction"
                        ]
                        == "HIGHER_HELPFUL"
                    ).sum()
                ),
                "lower_helpful_targets": int(
                    (
                        group[
                            "observed_direction"
                        ]
                        == "LOWER_HELPFUL"
                    ).sum()
                ),
                "weak_targets": int(
                    (
                        group[
                            "observed_direction"
                        ]
                        == "WEAK"
                    ).sum()
                ),
                "train_dev_consistent_targets": int(
                    group[
                        "train_development_direction_consistent"
                    ].sum()
                ),
            }
        )

    family_df = pd.DataFrame(
        family_rows
    ).sort_values(
        "mean_predictive_edge",
        ascending=False,
    )

    family_df.to_csv(
        FAMILY_OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 78)
    print("TOP PARAMETER WINNER ASSOCIATIONS")
    print("=" * 78)

    display_columns = [
        "family",
        "target",
        "train_dev_auc",
        "train_dev_higher_value_win_rate",
        "train_dev_winner_diff_effect",
        "train_auc",
        "development_auc",
        "observed_direction",
        "train_development_direction_consistent",
    ]

    print(
        summary_df[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("=" * 78)
    print("FAMILY SUMMARY")
    print("=" * 78)

    print(
        family_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)
    print(MATCHUP_OUTPUT_PATH)
    print(SUMMARY_OUTPUT_PATH)
    print(FAMILY_OUTPUT_PATH)
    print()
    print("2025+ outcomes were NOT analyzed.")
    print("HISTORICAL PARAMETER PREDICTIVENESS AUDIT COMPLETE")


if __name__ == "__main__":
    main()
