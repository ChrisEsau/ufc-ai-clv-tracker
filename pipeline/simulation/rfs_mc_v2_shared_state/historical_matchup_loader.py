"""Leakage-safe historical matchup loader for RFS Monte Carlo V2.

This module converts one historical fight from the shared Round Fighter State
history into two strictly pre-fight fighter profiles.

Actual fight outcomes are carried separately for later evaluation and are never
included in either fighter's simulator input profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


MIN_PRIOR_FIGHTS = 3

CURRENT_FIGHT_PREFIXES = (
    "rfs_traj_fight_",
    "rfs_open_fight_",
    "rfs_phase_base_fight_",
    "rfs_phase_interact_fight_",
    "rfs_dynamic_response_fight_",
    "rfs_finish_state_fight_",
)

PROFILE_PREFIXES = (
    "rfs_traj_",
    "rfs_open_",
    "rfs_phase_base_",
    "rfs_phase_interact_",
    "rfs_dynamic_response_",
    "rfs_finish_state_",
)


class HistoricalMatchupLoadError(RuntimeError):
    """Raised when a historical matchup cannot be loaded safely."""


@dataclass(frozen=True)
class HistoricalFighterProfile:
    """One fighter's leakage-safe state entering a historical fight."""

    fighter_id: str
    fighter_name: str
    corner: str
    prior_fight_count: int
    features: dict[str, Any]


@dataclass(frozen=True)
class HistoricalFightOutcome:
    """Observed result stored separately from simulator inputs."""

    winner_id: str | None
    method: str | None
    finish_round: int | None


@dataclass(frozen=True)
class HistoricalMatchup:
    """Historical fight with two pre-fight profiles plus evaluation truth."""

    fight_id: str
    date: pd.Timestamp
    event_name: str | None
    division: str | None
    scheduled_rounds: int | None
    red: HistoricalFighterProfile
    blue: HistoricalFighterProfile
    actual: HistoricalFightOutcome


def _profile_feature_columns(history_df: pd.DataFrame) -> list[str]:
    """Return simulator-eligible prior-state columns only."""

    columns = [
        column
        for column in history_df.columns
        if column.startswith(PROFILE_PREFIXES)
        and not column.startswith(CURRENT_FIGHT_PREFIXES)
    ]

    if not columns:
        raise HistoricalMatchupLoadError(
            "shared RFS history contains no prior-state profile columns"
        )

    return columns


def _single_value(
    frame: pd.DataFrame,
    column: str,
) -> Any:
    """Return one consistent metadata value for a fight."""

    if column not in frame.columns:
        return None

    values = frame[column].dropna().drop_duplicates()

    if len(values) == 0:
        return None

    if len(values) != 1:
        raise HistoricalMatchupLoadError(
            f"fight has inconsistent {column}: {values.tolist()}"
        )

    return values.iloc[0]


def _optional_int(value: Any) -> int | None:
    """Convert available numeric metadata to int."""

    if value is None or pd.isna(value):
        return None

    return int(value)


def _build_fighter_profile(
    row: pd.Series,
    feature_columns: list[str],
) -> HistoricalFighterProfile:
    """Build one leakage-safe fighter profile from the historical row."""

    prior_fight_count = int(
        pd.to_numeric(
            row["rfs_traj_prior_fight_count"],
            errors="raise",
        )
    )

    features = {
        column: row[column]
        for column in feature_columns
    }

    forbidden = [
        column
        for column in features
        if column.startswith(CURRENT_FIGHT_PREFIXES)
    ]

    if forbidden:
        raise HistoricalMatchupLoadError(
            "current-fight observations leaked into fighter profile: "
            f"{forbidden[:10]}"
        )

    return HistoricalFighterProfile(
        fighter_id=str(row["fighter_id"]),
        fighter_name=str(row["fighter_name"]),
        corner=str(row["corner"]).lower(),
        prior_fight_count=prior_fight_count,
        features=features,
    )


def load_historical_matchup(
    history_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    fight_id: str,
    *,
    min_prior_fights: int = MIN_PRIOR_FIGHTS,
) -> HistoricalMatchup:
    """Load one historical matchup using only state available pre-fight."""

    if min_prior_fights < 0:
        raise ValueError("min_prior_fights cannot be negative")

    required_history_columns = {
        "fight_id",
        "fighter_id",
        "fighter_name",
        "corner",
        "date",
        "rfs_traj_prior_fight_count",
    }

    missing_history = sorted(
        required_history_columns - set(history_df.columns)
    )

    if missing_history:
        raise HistoricalMatchupLoadError(
            "shared RFS history is missing required columns: "
            f"{missing_history}"
        )

    fight = history_df.loc[
        history_df["fight_id"].astype(str) == str(fight_id)
    ].copy()

    if len(fight) != 2:
        raise HistoricalMatchupLoadError(
            f"fight {fight_id} must have exactly two fighter rows; "
            f"found {len(fight)}"
        )

    if fight["fighter_id"].duplicated().any():
        raise HistoricalMatchupLoadError(
            f"fight {fight_id} contains duplicate fighter rows"
        )

    normalized_corner = (
        fight["corner"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if set(normalized_corner) != {"red", "blue"}:
        raise HistoricalMatchupLoadError(
            f"fight {fight_id} must contain red and blue corners"
        )

    fight["corner"] = normalized_corner

    profile_columns = _profile_feature_columns(history_df)

    red_row = fight.loc[fight["corner"] == "red"].iloc[0]
    blue_row = fight.loc[fight["corner"] == "blue"].iloc[0]

    red = _build_fighter_profile(
        red_row,
        profile_columns,
    )
    blue = _build_fighter_profile(
        blue_row,
        profile_columns,
    )

    if red.prior_fight_count < min_prior_fights:
        raise HistoricalMatchupLoadError(
            f"red fighter {red.fighter_name} has only "
            f"{red.prior_fight_count} prior fights; "
            f"minimum is {min_prior_fights}"
        )

    if blue.prior_fight_count < min_prior_fights:
        raise HistoricalMatchupLoadError(
            f"blue fighter {blue.fighter_name} has only "
            f"{blue.prior_fight_count} prior fights; "
            f"minimum is {min_prior_fights}"
        )

    outcome_rows = outcomes_df.loc[
        outcomes_df["fight_id"].astype(str) == str(fight_id)
    ].copy()

    if len(outcome_rows) != 1:
        raise HistoricalMatchupLoadError(
            f"fight {fight_id} must have exactly one outcome row; "
            f"found {len(outcome_rows)}"
        )

    outcome = outcome_rows.iloc[0]

    date = pd.to_datetime(
        _single_value(fight, "date"),
        errors="raise",
    )

    # Scheduled distance comes from authoritative master metadata.
    scheduled_rounds = outcome.get(
        "total_rounds"
    )

    return HistoricalMatchup(
        fight_id=str(fight_id),
        date=date,
        event_name=_single_value(
            fight,
            "event_name",
        ),
        division=_single_value(
            fight,
            "division",
        ),
        scheduled_rounds=_optional_int(
            scheduled_rounds
        ),
        red=red,
        blue=blue,
        actual=HistoricalFightOutcome(
            winner_id=(
                None
                if pd.isna(outcome.get("winner_id"))
                else str(outcome.get("winner_id"))
            ),
            method=(
                None
                if pd.isna(outcome.get("method"))
                else str(outcome.get("method"))
            ),
            finish_round=_optional_int(
                outcome.get("finish_round")
            ),
        ),
    )
