"""Build leakage-safe RFS Finish State fighter state.

The builder converts reciprocal UFCStats fighter-round rows into:

1. one completed-fight Finish State observation per fighter,
2. leakage-safe prior state for every historical fighter-fight row,
3. one complete latest-state row per fighter.

The builder combines reciprocal UFCStats round rows with authoritative
fight outcomes. It creates observable finish, survival, and adversity evidence.

It does not create the final six simulator Finish State parameters and does
not write parquet artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.round_stats.rfs_finish_state_feature_contracts import (
    FINISH_STATE_AGGREGATE_SPECS,
    FINISH_STATE_EVIDENCE_SPECS,
    FINISH_STATE_FIGHT_OBSERVATION_COLUMNS,
    FinishAggregatePerspective,
    FinishAggregateRule,
    FinishSourceKind,
)


EWM_ALPHA = 0.35

FIGHT_METADATA_COLUMNS = (
    "event_id",
    "event_name",
    "date",
    "fight_id",
    "division",
    "corner",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
)

REQUIRED_ROUND_COLUMNS = (
    "fight_id",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "corner",
    "round",
    "sig_str_landed",
    "sig_str_attempted",
    "td_landed",
    "sub_att",
    "ctrl_sec",
    "kd",
    "head_landed",
    "distance_landed",
    "clinch_landed",
    "clinch_attempted",
    "ground_landed",
    "ground_attempted",
)

REQUIRED_OUTCOME_COLUMNS = (
    "fight_id",
    "winner",
    "winner_id",
    "method",
    "finish_round",
)

COUNT_COLUMNS = (
    "round",
    "sig_str_landed",
    "sig_str_attempted",
    "td_landed",
    "sub_att",
    "ctrl_sec",
    "kd",
    "head_landed",
    "distance_landed",
    "clinch_landed",
    "clinch_attempted",
    "ground_landed",
    "ground_attempted",
)

LANDED_ATTEMPTED_PAIRS = (
    ("sig_str_landed", "sig_str_attempted"),
    ("clinch_landed", "clinch_attempted"),
    ("ground_landed", "ground_attempted"),
)

OPPONENT_RAW_COLUMNS = (
    "sig_str_landed",
    "kd",
    "head_landed",
    "clinch_landed",
    "ground_landed",
    "sub_att",
    "ctrl_sec",
)

VALID_DECISION_METHODS = frozenset(
    {
        "Decision - Unanimous",
        "Decision - Split",
        "Decision - Majority",
    }
)

KO_TKO_METHODS = frozenset(
    {
        "KO/TKO",
        "TKO - Doctor's Stoppage",
    }
)

SUBMISSION_METHODS = frozenset(
    {
        "Submission",
    }
)

VALID_FINISH_METHODS = frozenset(
    {
        *KO_TKO_METHODS,
        *SUBMISSION_METHODS,
        "Could Not Continue",
        "DQ",
    }
)

VALID_OUTCOME_METHODS = frozenset(
    {
        *VALID_DECISION_METHODS,
        *VALID_FINISH_METHODS,
    }
)

INVALID_OUTCOME_METHODS = frozenset(
    {
        "Overturned",
        "Other",
    }
)


@dataclass(frozen=True)
class FinishStateBuildResult:
    """History and latest-state outputs from one Finish State build."""

    history: pd.DataFrame
    latest: pd.DataFrame


class RoundFighterFinishStateBuildError(RuntimeError):
    """Raised when Finish State state cannot be built safely."""


def _require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    """Require all named columns with one readable build error."""

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RoundFighterFinishStateBuildError(
            f"{label} is missing required columns: {missing}"
        )


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Return a finite ratio or NaN when exposure is unavailable."""

    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan

    denominator = float(denominator)

    if denominator <= 0.0:
        return np.nan

    return float(numerator) / denominator


def _safe_difference(
    later: float,
    earlier: float,
) -> float:
    """Return later minus earlier while preserving missing evidence."""

    if pd.isna(later) or pd.isna(earlier):
        return np.nan

    return float(later) - float(earlier)


def _mean_available(values: Iterable[float]) -> float:
    """Average finite available values without manufacturing missing data."""

    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]

    if finite.size == 0:
        return np.nan

    return float(finite.mean())



def expanding_prior(series: pd.Series) -> pd.Series:
    """Expanding mean using only fights before the current row."""

    return (
        pd.to_numeric(series, errors="coerce")
        .shift(1)
        .expanding(min_periods=1)
        .mean()
    )


def last3_prior(series: pd.Series) -> pd.Series:
    """Three-fight mean using only fights before the current row."""

    return (
        pd.to_numeric(series, errors="coerce")
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )


def ewm_prior(
    series: pd.Series,
    alpha: float = EWM_ALPHA,
) -> pd.Series:
    """Exponentially weighted state using only prior fights."""

    return (
        pd.to_numeric(series, errors="coerce")
        .shift(1)
        .ewm(alpha=alpha, adjust=False)
        .mean()
    )


def cumulative_prior_total(series: pd.Series) -> pd.Series:
    """Cumulative total excluding the current fight."""

    values = (
        pd.to_numeric(series, errors="coerce")
        .fillna(0.0)
    )

    return values.shift(1, fill_value=0.0).cumsum()


def _fight_suffix(feature_name: str) -> str:
    """Remove the current-fight namespace from a feature name."""

    marker = "rfs_finish_state_fight_"

    if not feature_name.startswith(marker):
        raise ValueError(
            "Finish State feature does not use the fight prefix: "
            f"{feature_name}"
        )

    return feature_name.removeprefix(marker)


def prior_total_name(feature_name: str) -> str:
    """Return the prior cumulative-total name for an aggregate."""

    return (
        "rfs_finish_state_prior_total_"
        f"{_fight_suffix(feature_name)}"
    )


def evidence_state_name(
    feature_name: str,
    state_kind: str,
) -> str:
    """Return an expanding, last-three, or EWM state name."""

    if state_kind not in {"exp", "last3", "ewm"}:
        raise ValueError(
            "state_kind must be exp, last3, or ewm"
        )

    return (
        f"rfs_finish_state_{state_kind}_"
        f"{_fight_suffix(feature_name)}"
    )


def standardize_round_stats(
    round_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize and validate authoritative reciprocal round rows."""

    df = round_stats_df.copy()

    if "date" not in df.columns:
        if "event_date" not in df.columns:
            raise RoundFighterFinishStateBuildError(
                "round stats require date or event_date"
            )

        df["date"] = df["event_date"]

    _require_columns(
        df,
        REQUIRED_ROUND_COLUMNS,
        "round stats",
    )

    for optional_column in (
        "event_id",
        "event_name",
        "division",
    ):
        if optional_column not in df.columns:
            df[optional_column] = pd.NA

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if df["date"].isna().any():
        raise RoundFighterFinishStateBuildError(
            "round stats contain invalid dates"
        )

    for column in COUNT_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if df[column].isna().any():
            raise RoundFighterFinishStateBuildError(
                f"round stats contain nonnumeric {column}"
            )

        if (df[column] < 0).any():
            raise RoundFighterFinishStateBuildError(
                f"round stats contain negative {column}"
            )

    if (df["round"] < 1).any():
        raise RoundFighterFinishStateBuildError(
            "round numbers must be positive"
        )

    for landed, attempted in LANDED_ATTEMPTED_PAIRS:
        if (df[landed] > df[attempted]).any():
            raise RoundFighterFinishStateBuildError(
                f"{landed} cannot exceed {attempted}"
            )

    normalized_corner = (
        df["corner"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not normalized_corner.isin({"red", "blue"}).all():
        raise RoundFighterFinishStateBuildError(
            "corner must be red or blue"
        )

    df["corner"] = normalized_corner

    duplicate_mask = df.duplicated(
        subset=[
            "fight_id",
            "fighter_id",
            "round",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise RoundFighterFinishStateBuildError(
            "duplicate fighter-round rows detected"
        )

    _validate_reciprocal_round_rows(df)

    return df.sort_values(
        [
            "fighter_id",
            "date",
            "fight_id",
            "round",
        ]
    ).reset_index(drop=True)


def _validate_reciprocal_round_rows(
    df: pd.DataFrame,
) -> None:
    """Require two reciprocal fighter perspectives for every fight."""

    for fight_id, fight in df.groupby(
        "fight_id",
        sort=False,
    ):
        fighter_ids = fight["fighter_id"].drop_duplicates()

        if len(fighter_ids) != 2:
            raise RoundFighterFinishStateBuildError(
                f"fight {fight_id} does not contain exactly two fighters"
            )

        corners = set(fight["corner"])

        if corners != {"red", "blue"}:
            raise RoundFighterFinishStateBuildError(
                f"fight {fight_id} must contain red and blue corners"
            )

        round_sets: dict[object, tuple[int, ...]] = {}

        for fighter_id, fighter_rows in fight.groupby(
            "fighter_id",
            sort=False,
        ):
            opponent_ids = (
                fighter_rows["opponent_id"]
                .drop_duplicates()
                .tolist()
            )

            if len(opponent_ids) != 1:
                raise RoundFighterFinishStateBuildError(
                    f"fight {fight_id} has inconsistent opponent IDs"
                )

            opponent_id = opponent_ids[0]

            if opponent_id not in set(fighter_ids):
                raise RoundFighterFinishStateBuildError(
                    f"fight {fight_id} has nonreciprocal opponent identity"
                )

            if opponent_id == fighter_id:
                raise RoundFighterFinishStateBuildError(
                    f"fight {fight_id} contains self-opposition"
                )

            round_sets[fighter_id] = tuple(
                sorted(
                    pd.to_numeric(
                        fighter_rows["round"],
                        errors="raise",
                    )
                    .astype(int)
                    .tolist()
                )
            )

        if len(set(round_sets.values())) != 1:
            raise RoundFighterFinishStateBuildError(
                f"fight {fight_id} has unequal fighter round sets"
            )


def attach_opponent_round_values(
    standardized_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the exact reciprocal opponent row for each fighter-round."""

    opponent = standardized_df[
        [
            "fight_id",
            "fighter_id",
            "round",
            *OPPONENT_RAW_COLUMNS,
        ]
    ].rename(
        columns={
            "fighter_id": "opponent_id",
            **{
                column: f"opponent_{column}"
                for column in OPPONENT_RAW_COLUMNS
            },
        }
    )

    joined = standardized_df.merge(
        opponent,
        on=[
            "fight_id",
            "opponent_id",
            "round",
        ],
        how="left",
        validate="one_to_one",
    )

    opponent_columns = [
        f"opponent_{column}"
        for column in OPPONENT_RAW_COLUMNS
    ]

    if joined[opponent_columns].isna().any().any():
        raise RoundFighterFinishStateBuildError(
            "missing reciprocal opponent round values"
        )

    return joined



def standardize_outcomes(
    outcomes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize one authoritative outcome row per fight."""

    df = outcomes_df.copy()

    _require_columns(
        df,
        REQUIRED_OUTCOME_COLUMNS,
        "fight outcomes",
    )

    duplicate_mask = df.duplicated(
        subset=["fight_id"],
        keep=False,
    )

    if duplicate_mask.any():
        raise RoundFighterFinishStateBuildError(
            "fight outcomes contain duplicate fight_id values"
        )

    df["method"] = (
        df["method"]
        .astype("string")
        .str.strip()
    )

    if df["method"].isna().any():
        raise RoundFighterFinishStateBuildError(
            "fight outcomes contain missing method values"
        )

    known_methods = (
        VALID_OUTCOME_METHODS
        | INVALID_OUTCOME_METHODS
    )

    unknown_methods = sorted(
        set(df["method"].dropna()) - known_methods
    )

    if unknown_methods:
        raise RoundFighterFinishStateBuildError(
            "fight outcomes contain unknown methods: "
            f"{unknown_methods}"
        )

    df["finish_round"] = pd.to_numeric(
        df["finish_round"],
        errors="coerce",
    )

    if df["finish_round"].isna().any():
        raise RoundFighterFinishStateBuildError(
            "fight outcomes contain nonnumeric finish_round"
        )

    if (df["finish_round"] < 1).any():
        raise RoundFighterFinishStateBuildError(
            "finish_round must be positive"
        )

    return df[
        list(REQUIRED_OUTCOME_COLUMNS)
    ].reset_index(drop=True)


def attach_fight_outcomes(
    reciprocal_rounds_df: pd.DataFrame,
    standardized_outcomes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one authoritative fight outcome to every round row."""

    joined = reciprocal_rounds_df.merge(
        standardized_outcomes_df,
        on="fight_id",
        how="left",
        validate="many_to_one",
    )

    if joined["method"].isna().any():
        missing_fights = sorted(
            joined.loc[
                joined["method"].isna(),
                "fight_id",
            ]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )

        raise RoundFighterFinishStateBuildError(
            "round fights are missing authoritative outcomes: "
            f"{missing_fights[:10]}"
        )

    return joined

def _metadata_value(
    group: pd.DataFrame,
    column: str,
) -> object:
    """Return one consistent fight-level metadata value."""

    values = group[column].drop_duplicates()

    if len(values) != 1:
        raise RoundFighterFinishStateBuildError(
            f"inconsistent {column} within fighter-fight rows"
        )

    return values.iloc[0]



def _aggregate_value(
    group: pd.DataFrame,
    *,
    source_column: str,
    perspective: FinishAggregatePerspective,
    unique_count: bool,
) -> float:
    """Calculate one fighter- or opponent-perspective round aggregate."""

    if perspective is FinishAggregatePerspective.OPPONENT:
        source = f"opponent_{source_column}"
    else:
        source = source_column

    if source not in group.columns:
        raise RoundFighterFinishStateBuildError(
            f"aggregate source is unavailable: {source}"
        )

    values = pd.to_numeric(
        group[source],
        errors="coerce",
    )

    if unique_count:
        return float(values.nunique(dropna=True))

    return float(values.sum())


def _is_valid_outcome(group: pd.DataFrame) -> bool:
    """Return whether the joined outcome is usable for win/loss evidence."""

    method = _metadata_value(group, "method")
    winner_id = _metadata_value(group, "winner_id")

    return bool(
        method in VALID_OUTCOME_METHODS
        and pd.notna(winner_id)
    )


def _adversity_mask(
    group: pd.DataFrame,
) -> pd.Series:
    """Identify observable adversity rounds without outcome leakage."""

    ordered = group.sort_values("round").copy()

    knockdown = ordered["opponent_kd"] > 0

    elevated_signals = []

    for column in (
        "opponent_head_landed",
        "opponent_ground_landed",
        "opponent_ctrl_sec",
    ):
        values = pd.to_numeric(
            ordered[column],
            errors="coerce",
        )

        prior_maximum = (
            values
            .shift(1)
            .cummax()
        )

        elevated_signals.append(
            values > prior_maximum
        )

    elevated = pd.concat(
        elevated_signals,
        axis=1,
    ).any(axis=1)

    return (
        knockdown | elevated.fillna(False)
    ).astype(bool)


def _mean_round_output(
    rows: pd.DataFrame,
) -> float:
    """Return average component-level output for selected rounds."""

    if rows.empty:
        return np.nan

    component_means = [
        pd.to_numeric(
            rows[column],
            errors="coerce",
        ).mean()
        for column in (
            "sig_str_landed",
            "td_landed",
            "ctrl_sec",
        )
    ]

    return _mean_available(component_means)


def _mean_round_efficiency(
    rows: pd.DataFrame,
) -> float:
    """Return aggregate significant-strike accuracy for selected rounds."""

    if rows.empty:
        return np.nan

    landed = pd.to_numeric(
        rows["sig_str_landed"],
        errors="coerce",
    ).sum()

    attempted = pd.to_numeric(
        rows["sig_str_attempted"],
        errors="coerce",
    ).sum()

    return _safe_ratio(
        landed,
        attempted,
    )


def _same_round_output_preservation(
    group: pd.DataFrame,
) -> float:
    """Compare adversity-round output with non-adversity-round output."""

    ordered = group.sort_values("round").copy()
    adversity = _adversity_mask(ordered)

    adversity_output = _mean_round_output(
        ordered.loc[adversity]
    )

    baseline_output = _mean_round_output(
        ordered.loc[~adversity]
    )

    return _safe_ratio(
        adversity_output,
        baseline_output,
    )


def _same_round_efficiency_preservation(
    group: pd.DataFrame,
) -> float:
    """Compare adversity-round accuracy with non-adversity accuracy."""

    ordered = group.sort_values("round").copy()
    adversity = _adversity_mask(ordered)

    adversity_efficiency = _mean_round_efficiency(
        ordered.loc[adversity]
    )

    baseline_efficiency = _mean_round_efficiency(
        ordered.loc[~adversity]
    )

    return _safe_ratio(
        adversity_efficiency,
        baseline_efficiency,
    )


def _damage_exposure_composite(
    aggregates: dict[str, float],
) -> float:
    """Return an uncalibrated component-level damage exposure index."""

    rounds = aggregates[
        "rfs_finish_state_fight_rounds_observed"
    ]

    components = [
        _safe_ratio(
            aggregates[
                "rfs_finish_state_fight_knockdowns_absorbed"
            ],
            rounds,
        ),
        _safe_ratio(
            aggregates[
                "rfs_finish_state_fight_head_strikes_absorbed"
            ],
            rounds,
        ),
        _safe_ratio(
            aggregates[
                "rfs_finish_state_fight_ground_strikes_absorbed"
            ],
            rounds,
        ),
        _safe_ratio(
            aggregates[
                "rfs_finish_state_fight_opponent_control_seconds"
            ],
            rounds * 60.0,
        ),
    ]

    return _mean_available(components)


def _outcome_indicators(
    group: pd.DataFrame,
) -> dict[str, float]:
    """Build leakage-visible current-fight outcome indicators."""

    valid = _is_valid_outcome(group)

    if not valid:
        return {
            "rfs_finish_state_fight_ko_tko_loss_indicator": np.nan,
            "rfs_finish_state_fight_ko_tko_survival_indicator": np.nan,
            "rfs_finish_state_fight_submission_loss_indicator": np.nan,
            "rfs_finish_state_fight_submission_survival_indicator": np.nan,
        }

    fighter_id = _metadata_value(
        group,
        "fighter_id",
    )
    winner_id = _metadata_value(
        group,
        "winner_id",
    )
    method = _metadata_value(
        group,
        "method",
    )

    fighter_lost = fighter_id != winner_id

    ko_tko_loss = bool(
        fighter_lost
        and method in KO_TKO_METHODS
    )

    submission_loss = bool(
        fighter_lost
        and method in SUBMISSION_METHODS
    )

    return {
        "rfs_finish_state_fight_ko_tko_loss_indicator": float(
            ko_tko_loss
        ),
        "rfs_finish_state_fight_ko_tko_survival_indicator": float(
            not ko_tko_loss
        ),
        "rfs_finish_state_fight_submission_loss_indicator": float(
            submission_loss
        ),
        "rfs_finish_state_fight_submission_survival_indicator": float(
            not submission_loss
        ),
    }


def _build_evidence(
    group: pd.DataFrame,
    aggregates: dict[str, float],
) -> dict[str, float]:
    """Build all 14 current-fight Finish State evidence values."""

    def aggregate(suffix: str) -> float:
        return aggregates[
            f"rfs_finish_state_fight_{suffix}"
        ]

    evidence: dict[str, float] = {}

    evidence[
        "rfs_finish_state_fight_knockdowns_per_sig_strike_landed"
    ] = _safe_ratio(
        aggregate("knockdowns_scored"),
        aggregate("sig_strikes_landed"),
    )

    evidence[
        "rfs_finish_state_fight_"
        "knockdowns_per_distance_strike_landed_proxy"
    ] = _safe_ratio(
        aggregate("knockdowns_scored"),
        aggregate("distance_strikes_landed"),
    )

    evidence[
        "rfs_finish_state_fight_clinch_strike_accuracy"
    ] = _safe_ratio(
        aggregate("clinch_strikes_landed"),
        aggregate("clinch_strike_attempts"),
    )

    evidence[
        "rfs_finish_state_fight_clinch_damage_output_per_round"
    ] = _safe_ratio(
        aggregate("clinch_strikes_landed"),
        aggregate("rounds_observed"),
    )

    ground_opportunities = (
        aggregate("takedowns_landed")
        + aggregate("ground_strike_attempts")
    )

    evidence[
        "rfs_finish_state_fight_"
        "submission_attempts_per_ground_opportunity_proxy"
    ] = _safe_ratio(
        aggregate("submission_attempts"),
        ground_opportunities,
    )

    control_minutes = (
        aggregate("control_seconds") / 60.0
    )

    evidence[
        "rfs_finish_state_fight_submission_attempts_per_control_minute"
    ] = _safe_ratio(
        aggregate("submission_attempts"),
        control_minutes,
    )

    evidence[
        "rfs_finish_state_fight_adversity_round_count"
    ] = float(
        _adversity_mask(group).sum()
    )

    evidence[
        "rfs_finish_state_fight_damage_exposure_composite"
    ] = _damage_exposure_composite(
        aggregates
    )

    evidence[
        "rfs_finish_state_fight_same_round_output_preservation"
    ] = _same_round_output_preservation(
        group
    )

    evidence[
        "rfs_finish_state_fight_same_round_efficiency_preservation"
    ] = _same_round_efficiency_preservation(
        group
    )

    evidence.update(
        _outcome_indicators(group)
    )

    return evidence


def build_fight_level_observations(
    standardized_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one completed-fight Finish State row per fighter."""

    _require_columns(
        standardized_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *REQUIRED_ROUND_COLUMNS,
            *REQUIRED_OUTCOME_COLUMNS,
            *[
                f"opponent_{column}"
                for column in OPPONENT_RAW_COLUMNS
            ],
        ),
        "standardized Finish State round rows",
    )

    rows: list[dict[str, object]] = []

    grouped = standardized_df.groupby(
        [
            "fight_id",
            "fighter_id",
        ],
        sort=False,
    )

    for _, group in grouped:
        group = group.sort_values("round").copy()

        row: dict[str, object] = {
            column: _metadata_value(group, column)
            for column in FIGHT_METADATA_COLUMNS
        }

        row.update(
            {
                "winner": _metadata_value(
                    group,
                    "winner",
                ),
                "winner_id": _metadata_value(
                    group,
                    "winner_id",
                ),
                "method": _metadata_value(
                    group,
                    "method",
                ),
                "finish_round": _metadata_value(
                    group,
                    "finish_round",
                ),
            }
        )

        aggregates: dict[str, float] = {}

        for spec in FINISH_STATE_AGGREGATE_SPECS:
            if spec.source_kind is FinishSourceKind.OUTCOME:
                if spec.feature_name != (
                    "rfs_finish_state_fight_valid_outcome"
                ):
                    raise RoundFighterFinishStateBuildError(
                        "unsupported Finish State outcome aggregate: "
                        f"{spec.feature_name}"
                    )

                aggregates[spec.feature_name] = float(
                    _is_valid_outcome(group)
                )
                continue

            aggregates[spec.feature_name] = _aggregate_value(
                group,
                source_column=spec.source_column,
                perspective=spec.perspective,
                unique_count=(
                    spec.rule
                    is FinishAggregateRule.UNIQUE_COUNT
                ),
            )

        row.update(aggregates)
        row.update(
            _build_evidence(
                group,
                aggregates,
            )
        )

        rows.append(row)

    observations = pd.DataFrame(rows)

    _require_columns(
        observations,
        (
            *FIGHT_METADATA_COLUMNS,
            *REQUIRED_OUTCOME_COLUMNS[1:],
            *FINISH_STATE_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Finish State observations",
    )

    if observations.duplicated(
        subset=[
            "fight_id",
            "fighter_id",
        ]
    ).any():
        raise RoundFighterFinishStateBuildError(
            "Finish State observations violate fighter-fight grain"
        )

    numeric_observations = observations[
        list(FINISH_STATE_FIGHT_OBSERVATION_COLUMNS)
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if np.isinf(
        numeric_observations.to_numpy(dtype=float)
    ).any():
        raise RoundFighterFinishStateBuildError(
            "Finish State observations contain infinity"
        )

    unit_interval_columns = [
        spec.feature_name
        for spec in FINISH_STATE_EVIDENCE_SPECS
        if spec.unit_interval
    ]

    unit_values = observations[
        unit_interval_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_unit = (
        unit_values.notna()
        & (
            (unit_values < 0.0)
            | (unit_values > 1.0)
        )
    )

    if invalid_unit.any().any():
        raise RoundFighterFinishStateBuildError(
            "Finish State unit-interval evidence is out of range"
        )

    return observations.sort_values(
        [
            "fighter_id",
            "date",
            "fight_id",
        ]
    ).reset_index(drop=True)

def add_prior_finish_state_state(
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe prior totals and rolling evidence state."""

    _require_columns(
        observations_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *FINISH_STATE_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Finish State observations",
    )

    df = observations_df.sort_values(
        [
            "fighter_id",
            "date",
            "fight_id",
        ]
    ).reset_index(drop=True).copy()

    grouped = df.groupby(
        "fighter_id",
        group_keys=False,
        sort=False,
    )

    state_columns: dict[str, pd.Series] = {}

    for spec in FINISH_STATE_AGGREGATE_SPECS:
        state_columns[
            prior_total_name(spec.feature_name)
        ] = grouped[spec.feature_name].transform(
            cumulative_prior_total
        )

    for spec in FINISH_STATE_EVIDENCE_SPECS:
        fight_column = spec.feature_name

        state_columns[
            evidence_state_name(
                fight_column,
                "exp",
            )
        ] = grouped[fight_column].transform(
            expanding_prior
        )

        state_columns[
            evidence_state_name(
                fight_column,
                "last3",
            )
        ] = grouped[fight_column].transform(
            last3_prior
        )

        state_columns[
            evidence_state_name(
                fight_column,
                "ewm",
            )
        ] = grouped[fight_column].transform(
            ewm_prior
        )

    state_columns[
        "rfs_finish_state_prior_fight_count"
    ] = df.groupby("fighter_id").cumcount()

    evidence_columns = [
        spec.feature_name
        for spec in FINISH_STATE_EVIDENCE_SPECS
    ]

    valid_observation = (
        df[evidence_columns]
        .notna()
        .any(axis=1)
        .astype(int)
    )

    state_columns[
        "rfs_finish_state_prior_valid_observation_count"
    ] = (
        valid_observation
        .groupby(df["fighter_id"])
        .transform(
            lambda series: (
                series.cumsum()
                .shift(1)
                .fillna(0)
            )
        )
        .astype(int)
    )

    state_frame = pd.DataFrame(
        state_columns,
        index=df.index,
    )

    rolling_state_columns = [
        column
        for column in state_frame.columns
        if column.startswith(
            (
                "rfs_finish_state_exp_",
                "rfs_finish_state_last3_",
                "rfs_finish_state_ewm_",
            )
        )
    ]

    state_frame[
        "rfs_finish_state_has_state"
    ] = (
        state_frame[rolling_state_columns]
        .notna()
        .any(axis=1)
        .astype(int)
    )

    return pd.concat(
        [
            df,
            state_frame,
        ],
        axis=1,
    )


def _inclusive_expanding(series: pd.Series) -> float:
    """Return the complete expanding mean through the latest fight."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def _inclusive_last3(series: pd.Series) -> float:
    """Return the complete mean of the latest three valid fights."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.tail(3).mean())


def _inclusive_ewm(
    series: pd.Series,
    alpha: float = EWM_ALPHA,
) -> float:
    """Return complete EWM state through the latest fight."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(
        values.ewm(
            alpha=alpha,
            adjust=False,
        ).mean().iloc[-1]
    )


def build_latest_finish_state_state(
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build complete Finish State state for future matchups."""

    _require_columns(
        history_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *FINISH_STATE_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Finish State history",
    )

    latest_rows: list[dict[str, object]] = []

    ordered_history = history_df.sort_values(
        [
            "fighter_id",
            "date",
            "fight_id",
        ]
    )

    for fighter_id, group in ordered_history.groupby(
        "fighter_id",
        sort=False,
    ):
        group = group.sort_values(
            [
                "date",
                "fight_id",
            ]
        ).reset_index(drop=True)

        last = group.iloc[-1]

        state: dict[str, object] = {
            "fighter_id": fighter_id,
            "fighter_name": last["fighter_name"],
            "division": last["division"],
            "latest_event_name": last["event_name"],
            "latest_date": last["date"],
            "latest_fight_id": last["fight_id"],
        }

        for spec in FINISH_STATE_AGGREGATE_SPECS:
            state[
                prior_total_name(spec.feature_name)
            ] = float(
                pd.to_numeric(
                    group[spec.feature_name],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )

        for spec in FINISH_STATE_EVIDENCE_SPECS:
            fight_column = spec.feature_name

            state[
                evidence_state_name(
                    fight_column,
                    "exp",
                )
            ] = _inclusive_expanding(
                group[fight_column]
            )

            state[
                evidence_state_name(
                    fight_column,
                    "last3",
                )
            ] = _inclusive_last3(
                group[fight_column]
            )

            state[
                evidence_state_name(
                    fight_column,
                    "ewm",
                )
            ] = _inclusive_ewm(
                group[fight_column]
            )

        state[
            "rfs_finish_state_prior_fight_count"
        ] = int(len(group))

        evidence_columns = [
            spec.feature_name
            for spec in FINISH_STATE_EVIDENCE_SPECS
        ]

        state[
            "rfs_finish_state_prior_valid_observation_count"
        ] = int(
            group[evidence_columns]
            .notna()
            .any(axis=1)
            .sum()
        )

        rolling_values = [
            value
            for column, value in state.items()
            if column.startswith(
                (
                    "rfs_finish_state_exp_",
                    "rfs_finish_state_last3_",
                    "rfs_finish_state_ewm_",
                )
            )
        ]

        state[
            "rfs_finish_state_has_state"
        ] = int(
            any(
                pd.notna(value)
                for value in rolling_values
            )
        )

        latest_rows.append(state)

    latest = pd.DataFrame(latest_rows)

    if latest.duplicated(
        subset=["fighter_id"]
    ).any():
        raise RoundFighterFinishStateBuildError(
            "latest Finish State state has duplicate fighters"
        )

    numeric_columns = [
        column
        for column in latest.columns
        if column.startswith("rfs_finish_state_")
    ]

    if numeric_columns and np.isinf(
        latest[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    ).any():
        raise RoundFighterFinishStateBuildError(
            "latest Finish State state contains infinity"
        )

    return latest



def build_round_fighter_finish_state(
    round_stats_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> FinishStateBuildResult:
    """Build history and latest Finish State fighter state."""

    standardized_rounds = standardize_round_stats(
        round_stats_df
    )

    reciprocal_rounds = attach_opponent_round_values(
        standardized_rounds
    )

    standardized_outcomes = standardize_outcomes(
        outcomes_df
    )

    joined_rounds = attach_fight_outcomes(
        reciprocal_rounds,
        standardized_outcomes,
    )

    observations = build_fight_level_observations(
        joined_rounds
    )

    history = add_prior_finish_state_state(
        observations
    )

    latest = build_latest_finish_state_state(
        history
    )

    return FinishStateBuildResult(
        history=history,
        latest=latest,
    )
