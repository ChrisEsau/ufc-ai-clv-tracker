"""Build leakage-safe submission-result history for RFS Monte Carlo V1.

The output contains one post-fight snapshot per fighter and fight.

Because the profile builder selects rows strictly before the target date,
the selected snapshot contains only fights completed before the target
fight. The target fight itself cannot enter the simulation profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    MASTER_PATH,
    RFS_MC_V1_SUBMISSION_HISTORY_PATH,
)


BETA_PRIOR_ALPHA = 1.0
BETA_PRIOR_BETA = 9.0
CONVERSION_PRIOR_STRENGTH = (
    BETA_PRIOR_ALPHA + BETA_PRIOR_BETA
)


REQUIRED_COLUMNS = {
    "date",
    "fight_id",
    "method",
    "winner",
    "winner_id",
    "r_id",
    "r_name",
    "r_sub_att",
    "b_id",
    "b_name",
    "b_sub_att",
}


@dataclass(frozen=True)
class SubmissionHistoryBuildResult:
    """Submission-result history artifact."""

    history: pd.DataFrame


class SubmissionHistoryBuildError(RuntimeError):
    """Raised when submission history cannot be built safely."""


def _safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """Convert a series to nonnegative numeric values."""

    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide while returning zero for unavailable denominators."""

    denominator = denominator.replace(0.0, np.nan)

    return (
        numerator.div(denominator)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _normalize_text(
    series: pd.Series,
) -> pd.Series:
    """Normalize identifiers and names for comparison."""

    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.casefold()
    )


def build_submission_history(
    master: pd.DataFrame,
) -> SubmissionHistoryBuildResult:
    """Build post-fight submission snapshots for both fighters."""

    missing = REQUIRED_COLUMNS - set(master.columns)

    if missing:
        raise SubmissionHistoryBuildError(
            "Master dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    fights = master[
        sorted(REQUIRED_COLUMNS)
    ].copy()

    fights["date"] = pd.to_datetime(
        fights["date"],
        errors="coerce",
    ).dt.normalize()

    fights = fights.loc[
        fights["date"].notna()
        & fights["fight_id"].notna()
        & fights["r_id"].notna()
        & fights["b_id"].notna()
    ].copy()

    # Defensive deduplication in case the master contains repeated rows.
    fights = (
        fights.sort_values(
            ["date", "fight_id"],
            kind="stable",
        )
        .drop_duplicates(
            subset=["fight_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    winner_id = _normalize_text(fights["winner_id"])
    winner_name = _normalize_text(fights["winner"])

    red_id = _normalize_text(fights["r_id"])
    blue_id = _normalize_text(fights["b_id"])

    red_name = _normalize_text(fights["r_name"])
    blue_name = _normalize_text(fights["b_name"])

    red_win = (
        (winner_id != "")
        & (winner_id == red_id)
    ) | (
        (winner_id == "")
        & (winner_name != "")
        & (winner_name == red_name)
    )

    blue_win = (
        (winner_id != "")
        & (winner_id == blue_id)
    ) | (
        (winner_id == "")
        & (winner_name != "")
        & (winner_name == blue_name)
    )

    method = _normalize_text(fights["method"])

    is_submission = method.str.contains(
        "submission",
        regex=False,
        na=False,
    )

    red_sub_attempts = _safe_numeric(
        fights["r_sub_att"]
    )
    blue_sub_attempts = _safe_numeric(
        fights["b_sub_att"]
    )

    red_rows = pd.DataFrame(
        {
            "date": fights["date"],
            "fight_id": fights["fight_id"].astype(str),
            "fighter_id": fights["r_id"].astype(str),
            "fighter_name": fights["r_name"].astype(str),
            "opponent_id": fights["b_id"].astype(str),
            "opponent_name": fights["b_name"].astype(str),
            "fight_won": red_win.astype(int),
            "fight_lost": blue_win.astype(int),
            "submission_win": (
                is_submission & red_win
            ).astype(int),
            "submission_loss": (
                is_submission & blue_win
            ).astype(int),
            "submission_attempts": red_sub_attempts,
            "opponent_submission_attempts": (
                blue_sub_attempts
            ),
        }
    )

    blue_rows = pd.DataFrame(
        {
            "date": fights["date"],
            "fight_id": fights["fight_id"].astype(str),
            "fighter_id": fights["b_id"].astype(str),
            "fighter_name": fights["b_name"].astype(str),
            "opponent_id": fights["r_id"].astype(str),
            "opponent_name": fights["r_name"].astype(str),
            "fight_won": blue_win.astype(int),
            "fight_lost": red_win.astype(int),
            "submission_win": (
                is_submission & blue_win
            ).astype(int),
            "submission_loss": (
                is_submission & red_win
            ).astype(int),
            "submission_attempts": blue_sub_attempts,
            "opponent_submission_attempts": (
                red_sub_attempts
            ),
        }
    )

    history = pd.concat(
        [red_rows, blue_rows],
        ignore_index=True,
    )

    history = history.sort_values(
        ["fighter_id", "date", "fight_id"],
        kind="stable",
    ).reset_index(drop=True)

    grouped = history.groupby(
        "fighter_id",
        sort=False,
        group_keys=False,
    )

    # These are post-fight cumulative snapshots. Selecting a row strictly
    # before the target date therefore includes every completed prior fight.
    history["rfs_sub_career_fight_count"] = (
        grouped.cumcount() + 1
    ).astype(int)

    history["rfs_sub_career_valid_fight_count"] = (
        history["rfs_sub_career_fight_count"]
    )

    cumulative_columns = {
        "submission_win": (
            "rfs_sub_career_submission_wins"
        ),
        "submission_loss": (
            "rfs_sub_career_submission_losses"
        ),
        "submission_attempts": (
            "rfs_sub_career_submission_attempts"
        ),
        "opponent_submission_attempts": (
            "rfs_sub_career_opponent_submission_attempts"
        ),
    }

    for source, destination in cumulative_columns.items():
        history[destination] = (
            grouped[source].cumsum()
        )

    fight_count = history[
        "rfs_sub_career_fight_count"
    ].astype(float)

    submission_wins = history[
        "rfs_sub_career_submission_wins"
    ].astype(float)

    submission_losses = history[
        "rfs_sub_career_submission_losses"
    ].astype(float)

    submission_attempts = history[
        "rfs_sub_career_submission_attempts"
    ].astype(float)

    history[
        "rfs_sub_career_submission_win_rate"
    ] = _safe_divide(
        submission_wins,
        fight_count,
    )

    history[
        "rfs_sub_career_submission_loss_rate"
    ] = _safe_divide(
        submission_losses,
        fight_count,
    )

    # Beta-smoothed attempt conversion. The prior mean is 10%, with the
    # equivalent weight of ten attempts.
    history[
        "rfs_sub_career_smoothed_conversion_rate"
    ] = (
        submission_wins + BETA_PRIOR_ALPHA
    ) / (
        submission_attempts
        + CONVERSION_PRIOR_STRENGTH
    )

    output_columns = [
        "date",
        "fight_id",
        "fighter_id",
        "fighter_name",
        "opponent_id",
        "opponent_name",
        "rfs_sub_career_fight_count",
        "rfs_sub_career_valid_fight_count",
        "rfs_sub_career_submission_wins",
        "rfs_sub_career_submission_losses",
        "rfs_sub_career_submission_attempts",
        "rfs_sub_career_opponent_submission_attempts",
        "rfs_sub_career_submission_win_rate",
        "rfs_sub_career_smoothed_conversion_rate",
        "rfs_sub_career_submission_loss_rate",
    ]

    history = history[output_columns].copy()

    return SubmissionHistoryBuildResult(
        history=history,
    )


def main() -> None:
    """Build and write the submission-history artifact."""

    master_path = Path(MASTER_PATH)

    if not master_path.exists():
        raise SubmissionHistoryBuildError(
            f"Master dataset not found: {master_path}"
        )

    master = pd.read_parquet(master_path)

    result = build_submission_history(master)

    output_path = Path(
        RFS_MC_V1_SUBMISSION_HISTORY_PATH
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.history.to_parquet(
        output_path,
        index=False,
    )

    print(
        "Wrote submission history: "
        f"{output_path}"
    )
    print(
        "Rows: "
        f"{len(result.history):,}"
    )
    print(
        "Fighters: "
        f"{result.history['fighter_id'].nunique():,}"
    )


if __name__ == "__main__":
    main()
