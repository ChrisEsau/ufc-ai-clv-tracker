"""Build leakage-safe RFS Dynamic Response fighter state.

The builder converts reciprocal UFCStats fighter-round rows into:

1. one completed-fight Dynamic Response observation per fighter,
2. leakage-safe prior state for every historical fighter-fight row,
3. one complete latest-state row per fighter.

The builder creates observable trajectory and adversity evidence. It does not
create the final four simulator Dynamic Response parameters and does not write
parquet artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.round_stats.rfs_dynamic_response_feature_contracts import (
    DYNAMIC_RESPONSE_AGGREGATE_SPECS,
    DYNAMIC_RESPONSE_EVIDENCE_SPECS,
    DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS,
    DynamicAggregatePerspective,
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
    "total_str_landed",
    "total_str_attempted",
    "td_landed",
    "td_attempted",
    "ctrl_sec",
    "kd",
    "head_landed",
    "ground_landed",
)

COUNT_COLUMNS = (
    "round",
    "sig_str_landed",
    "sig_str_attempted",
    "total_str_landed",
    "total_str_attempted",
    "td_landed",
    "td_attempted",
    "ctrl_sec",
    "kd",
    "head_landed",
    "ground_landed",
)

LANDED_ATTEMPTED_PAIRS = (
    ("sig_str_landed", "sig_str_attempted"),
    ("total_str_landed", "total_str_attempted"),
    ("td_landed", "td_attempted"),
)

OPPONENT_RAW_COLUMNS = (
    "kd",
    "head_landed",
    "ground_landed",
    "ctrl_sec",
)


@dataclass(frozen=True)
class DynamicResponseBuildResult:
    """History and latest-state outputs from one Dynamic Response build."""

    history: pd.DataFrame
    latest: pd.DataFrame


class RoundFighterDynamicResponseBuildError(RuntimeError):
    """Raised when Dynamic Response state cannot be built safely."""


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
        raise RoundFighterDynamicResponseBuildError(
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


def _ols_slope(
    round_numbers: pd.Series,
    values: pd.Series,
) -> float:
    """Calculate an OLS round slope when at least two points exist."""

    x = pd.to_numeric(
        round_numbers,
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2:
        return np.nan

    if np.unique(x).size < 2:
        return np.nan

    return float(np.polyfit(x, y, 1)[0])


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

    marker = "rfs_dynamic_response_fight_"

    if not feature_name.startswith(marker):
        raise ValueError(
            "Dynamic Response feature does not use the fight prefix: "
            f"{feature_name}"
        )

    return feature_name.removeprefix(marker)


def prior_total_name(feature_name: str) -> str:
    """Return the prior cumulative-total name for an aggregate."""

    return (
        "rfs_dynamic_response_prior_total_"
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
        f"rfs_dynamic_response_{state_kind}_"
        f"{_fight_suffix(feature_name)}"
    )


def standardize_round_stats(
    round_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize and validate authoritative reciprocal round rows."""

    df = round_stats_df.copy()

    if "date" not in df.columns:
        if "event_date" not in df.columns:
            raise RoundFighterDynamicResponseBuildError(
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
        raise RoundFighterDynamicResponseBuildError(
            "round stats contain invalid dates"
        )

    for column in COUNT_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if df[column].isna().any():
            raise RoundFighterDynamicResponseBuildError(
                f"round stats contain nonnumeric {column}"
            )

        if (df[column] < 0).any():
            raise RoundFighterDynamicResponseBuildError(
                f"round stats contain negative {column}"
            )

    if (df["round"] < 1).any():
        raise RoundFighterDynamicResponseBuildError(
            "round numbers must be positive"
        )

    for landed, attempted in LANDED_ATTEMPTED_PAIRS:
        if (df[landed] > df[attempted]).any():
            raise RoundFighterDynamicResponseBuildError(
                f"{landed} cannot exceed {attempted}"
            )

    normalized_corner = (
        df["corner"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not normalized_corner.isin({"red", "blue"}).all():
        raise RoundFighterDynamicResponseBuildError(
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
        raise RoundFighterDynamicResponseBuildError(
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
            raise RoundFighterDynamicResponseBuildError(
                f"fight {fight_id} does not contain exactly two fighters"
            )

        corners = set(fight["corner"])

        if corners != {"red", "blue"}:
            raise RoundFighterDynamicResponseBuildError(
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
                raise RoundFighterDynamicResponseBuildError(
                    f"fight {fight_id} has inconsistent opponent IDs"
                )

            opponent_id = opponent_ids[0]

            if opponent_id not in set(fighter_ids):
                raise RoundFighterDynamicResponseBuildError(
                    f"fight {fight_id} has nonreciprocal opponent identity"
                )

            if opponent_id == fighter_id:
                raise RoundFighterDynamicResponseBuildError(
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
            raise RoundFighterDynamicResponseBuildError(
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
        raise RoundFighterDynamicResponseBuildError(
            "missing reciprocal opponent round values"
        )

    return joined


def _metadata_value(
    group: pd.DataFrame,
    column: str,
) -> object:
    """Return one consistent fight-level metadata value."""

    values = group[column].drop_duplicates()

    if len(values) != 1:
        raise RoundFighterDynamicResponseBuildError(
            f"inconsistent {column} within fighter-fight rows"
        )

    return values.iloc[0]


def _aggregate_value(
    group: pd.DataFrame,
    *,
    source_column: str,
    perspective: DynamicAggregatePerspective,
    unique_count: bool,
) -> float:
    """Calculate one fighter- or opponent-perspective aggregate."""

    if perspective is DynamicAggregatePerspective.OPPONENT:
        source = f"opponent_{source_column}"
    else:
        source = source_column

    if source not in group.columns:
        raise RoundFighterDynamicResponseBuildError(
            f"aggregate source is unavailable: {source}"
        )

    values = pd.to_numeric(
        group[source],
        errors="coerce",
    )

    if unique_count:
        return float(values.nunique(dropna=True))

    return float(values.sum())


def _first_last_values(
    group: pd.DataFrame,
    column: str,
) -> tuple[float, float]:
    """Return first- and last-round values for one raw column."""

    ordered = group.sort_values("round")

    return (
        float(ordered.iloc[0][column]),
        float(ordered.iloc[-1][column]),
    )


def _first_last_accuracy(
    group: pd.DataFrame,
    landed: str,
    attempted: str,
) -> tuple[float, float]:
    """Return first- and last-round accuracy with safe denominators."""

    ordered = group.sort_values("round")
    first = ordered.iloc[0]
    last = ordered.iloc[-1]

    return (
        _safe_ratio(first[landed], first[attempted]),
        _safe_ratio(last[landed], last[attempted]),
    )


def _component_first_last_ratios(
    group: pd.DataFrame,
    columns: Iterable[str],
) -> list[float]:
    """Return first-to-last ratios for each available component."""

    ratios: list[float] = []

    for column in columns:
        first, last = _first_last_values(
            group,
            column,
        )
        ratios.append(
            _safe_ratio(last, first)
        )

    return ratios


def _component_first_last_differences(
    group: pd.DataFrame,
    columns: Iterable[str],
) -> list[float]:
    """Return first-to-last differences for each component."""

    differences: list[float] = []

    for column in columns:
        first, last = _first_last_values(
            group,
            column,
        )
        differences.append(
            _safe_difference(last, first)
        )

    return differences


def _adversity_mask(
    group: pd.DataFrame,
) -> pd.Series:
    """Identify observable adversity rounds without outcome leakage.

    A round is marked when:

    - the opponent records a knockdown, or
    - opponent head, ground, or control output exceeds that opponent's
      earlier-round maximum in the same fight.

    The first observed round therefore requires a knockdown to qualify.
    This avoids declaring ordinary first-round activity to be acute adversity.
    """

    ordered = group.sort_values("round").copy()

    knockdown = (
        ordered["opponent_kd"] > 0
    )

    elevated_signals = []

    for column in (
        "opponent_head_landed",
        "opponent_ground_landed",
        "opponent_ctrl_sec",
    ):
        prior_maximum = (
            pd.to_numeric(
                ordered[column],
                errors="coerce",
            )
            .shift(1)
            .cummax()
        )

        elevated_signals.append(
            pd.to_numeric(
                ordered[column],
                errors="coerce",
            )
            > prior_maximum
        )

    elevated = pd.concat(
        elevated_signals,
        axis=1,
    ).any(axis=1)

    return (
        knockdown | elevated.fillna(False)
    ).astype(bool)


def _post_adversity_pairs(
    group: pd.DataFrame,
) -> list[tuple[pd.Series, pd.Series]]:
    """Return adversity-round and immediately following-round pairs."""

    ordered = (
        group.sort_values("round")
        .reset_index(drop=True)
    )

    adversity = (
        _adversity_mask(ordered)
        .reset_index(drop=True)
    )

    pairs: list[tuple[pd.Series, pd.Series]] = []

    for index in range(len(ordered) - 1):
        if adversity.iloc[index]:
            pairs.append(
                (
                    ordered.iloc[index],
                    ordered.iloc[index + 1],
                )
            )

    return pairs


def _post_adversity_attempt_rebound(
    group: pd.DataFrame,
) -> float:
    """Average next-round change in significant-strike attempts."""

    changes = [
        _safe_difference(
            after["sig_str_attempted"],
            adversity["sig_str_attempted"],
        )
        for adversity, after in _post_adversity_pairs(group)
    ]

    return _mean_available(changes)


def _post_adversity_output_rebound(
    group: pd.DataFrame,
) -> float:
    """Average component-level output rebound after adversity."""

    event_changes: list[float] = []

    for adversity, after in _post_adversity_pairs(group):
        component_changes = [
            _safe_difference(
                after[column],
                adversity[column],
            )
            for column in (
                "sig_str_landed",
                "td_landed",
                "ctrl_sec",
            )
        ]

        event_changes.append(
            _mean_available(component_changes)
        )

    return _mean_available(event_changes)


def _post_adversity_efficiency_preservation(
    group: pd.DataFrame,
) -> float:
    """Average next-round change in significant-strike accuracy."""

    changes: list[float] = []

    for adversity, after in _post_adversity_pairs(group):
        adversity_accuracy = _safe_ratio(
            adversity["sig_str_landed"],
            adversity["sig_str_attempted"],
        )
        after_accuracy = _safe_ratio(
            after["sig_str_landed"],
            after["sig_str_attempted"],
        )

        changes.append(
            _safe_difference(
                after_accuracy,
                adversity_accuracy,
            )
        )

    return _mean_available(changes)


def _build_evidence(
    group: pd.DataFrame,
    aggregates: dict[str, float],
) -> dict[str, float]:
    """Build all 24 current-fight Dynamic Response evidence values."""

    n_rounds = len(group)

    def aggregate(suffix: str) -> float:
        return aggregates[
            f"rfs_dynamic_response_fight_{suffix}"
        ]

    evidence: dict[str, float] = {}

    evidence[
        "rfs_dynamic_response_fight_sig_strike_accuracy"
    ] = _safe_ratio(
        aggregate("sig_strikes_landed"),
        aggregate("sig_strike_attempts"),
    )

    evidence[
        "rfs_dynamic_response_fight_total_strike_accuracy"
    ] = _safe_ratio(
        aggregate("total_strikes_landed"),
        aggregate("total_strike_attempts"),
    )

    evidence[
        "rfs_dynamic_response_fight_td_completion_rate"
    ] = _safe_ratio(
        aggregate("td_landed"),
        aggregate("td_attempts"),
    )

    evidence[
        "rfs_dynamic_response_fight_sig_strike_attempts_per_round"
    ] = _safe_ratio(
        aggregate("sig_strike_attempts"),
        aggregate("rounds_observed"),
    )

    evidence[
        "rfs_dynamic_response_fight_total_strike_attempts_per_round"
    ] = _safe_ratio(
        aggregate("total_strike_attempts"),
        aggregate("rounds_observed"),
    )

    evidence[
        "rfs_dynamic_response_fight_td_attempts_per_round"
    ] = _safe_ratio(
        aggregate("td_attempts"),
        aggregate("rounds_observed"),
    )

    evidence[
        "rfs_dynamic_response_fight_control_seconds_per_round"
    ] = _safe_ratio(
        aggregate("control_seconds"),
        aggregate("rounds_observed"),
    )

    slope_columns = {
        "sig_strike_attempt_slope": "sig_str_attempted",
        "total_strike_attempt_slope": "total_str_attempted",
        "td_attempt_slope": "td_attempted",
        "control_seconds_slope": "ctrl_sec",
        "sig_strike_landed_slope": "sig_str_landed",
        "total_strike_landed_slope": "total_str_landed",
    }

    for suffix, column in slope_columns.items():
        evidence[
            f"rfs_dynamic_response_fight_{suffix}"
        ] = _ols_slope(
            group["round"],
            group[column],
        )

    if n_rounds < 2:
        multi_round_names = (
            "sig_strike_attempt_first_last_ratio",
            "total_strike_attempt_first_last_ratio",
            "late_early_workload_ratio",
            "late_early_workload_difference",
            "sig_strike_accuracy_change",
            "total_strike_accuracy_change",
            "late_early_output_ratio",
            "post_adversity_sig_strike_rebound",
            "post_adversity_output_rebound",
            "post_adversity_efficiency_preservation",
        )

        for suffix in multi_round_names:
            evidence[
                f"rfs_dynamic_response_fight_{suffix}"
            ] = np.nan
    else:
        first_sig_attempts, last_sig_attempts = (
            _first_last_values(
                group,
                "sig_str_attempted",
            )
        )

        first_total_attempts, last_total_attempts = (
            _first_last_values(
                group,
                "total_str_attempted",
            )
        )

        evidence[
            "rfs_dynamic_response_fight_"
            "sig_strike_attempt_first_last_ratio"
        ] = _safe_ratio(
            last_sig_attempts,
            first_sig_attempts,
        )

        evidence[
            "rfs_dynamic_response_fight_"
            "total_strike_attempt_first_last_ratio"
        ] = _safe_ratio(
            last_total_attempts,
            first_total_attempts,
        )

        workload_columns = (
            "sig_str_attempted",
            "total_str_attempted",
            "td_attempted",
            "ctrl_sec",
        )

        evidence[
            "rfs_dynamic_response_fight_late_early_workload_ratio"
        ] = _mean_available(
            _component_first_last_ratios(
                group,
                workload_columns,
            )
        )

        evidence[
            "rfs_dynamic_response_fight_"
            "late_early_workload_difference"
        ] = _mean_available(
            _component_first_last_differences(
                group,
                workload_columns,
            )
        )

        first_sig_accuracy, last_sig_accuracy = (
            _first_last_accuracy(
                group,
                "sig_str_landed",
                "sig_str_attempted",
            )
        )

        evidence[
            "rfs_dynamic_response_fight_sig_strike_accuracy_change"
        ] = _safe_difference(
            last_sig_accuracy,
            first_sig_accuracy,
        )

        first_total_accuracy, last_total_accuracy = (
            _first_last_accuracy(
                group,
                "total_str_landed",
                "total_str_attempted",
            )
        )

        evidence[
            "rfs_dynamic_response_fight_total_strike_accuracy_change"
        ] = _safe_difference(
            last_total_accuracy,
            first_total_accuracy,
        )

        output_columns = (
            "sig_str_landed",
            "total_str_landed",
            "td_landed",
            "ctrl_sec",
        )

        evidence[
            "rfs_dynamic_response_fight_late_early_output_ratio"
        ] = _mean_available(
            _component_first_last_ratios(
                group,
                output_columns,
            )
        )

        evidence[
            "rfs_dynamic_response_fight_"
            "post_adversity_sig_strike_rebound"
        ] = _post_adversity_attempt_rebound(group)

        evidence[
            "rfs_dynamic_response_fight_post_adversity_output_rebound"
        ] = _post_adversity_output_rebound(group)

        evidence[
            "rfs_dynamic_response_fight_"
            "post_adversity_efficiency_preservation"
        ] = _post_adversity_efficiency_preservation(group)

    evidence[
        "rfs_dynamic_response_fight_adversity_round_count"
    ] = float(_adversity_mask(group).sum())

    return evidence


def build_fight_level_observations(
    standardized_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one completed-fight Dynamic Response row per fighter."""

    _require_columns(
        standardized_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *REQUIRED_ROUND_COLUMNS,
            *[
                f"opponent_{column}"
                for column in OPPONENT_RAW_COLUMNS
            ],
        ),
        "standardized Dynamic Response round rows",
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

        aggregates: dict[str, float] = {}

        for spec in DYNAMIC_RESPONSE_AGGREGATE_SPECS:
            aggregates[spec.feature_name] = _aggregate_value(
                group,
                source_column=spec.source_column,
                perspective=spec.perspective,
                unique_count=(
                    spec.rule.value == "unique_count"
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
            *DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Dynamic Response observations",
    )

    duplicate_count = int(
        observations.duplicated(
            subset=[
                "fight_id",
                "fighter_id",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RoundFighterDynamicResponseBuildError(
            "Dynamic Response observations violate fighter-fight grain"
        )

    numeric_observations = observations[
        list(DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS)
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if np.isinf(
        numeric_observations.to_numpy(dtype=float)
    ).any():
        raise RoundFighterDynamicResponseBuildError(
            "Dynamic Response observations contain infinity"
        )

    unit_interval_columns = [
        spec.feature_name
        for spec in DYNAMIC_RESPONSE_EVIDENCE_SPECS
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
        raise RoundFighterDynamicResponseBuildError(
            "Dynamic Response unit-interval evidence is out of range"
        )

    return observations.sort_values(
        [
            "fighter_id",
            "date",
            "fight_id",
        ]
    ).reset_index(drop=True)


def add_prior_dynamic_response_state(
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe prior totals and rolling evidence state."""

    _require_columns(
        observations_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Dynamic Response observations",
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

    for spec in DYNAMIC_RESPONSE_AGGREGATE_SPECS:
        state_columns[
            prior_total_name(spec.feature_name)
        ] = grouped[spec.feature_name].transform(
            cumulative_prior_total
        )

    for spec in DYNAMIC_RESPONSE_EVIDENCE_SPECS:
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
        "rfs_dynamic_response_prior_fight_count"
    ] = df.groupby("fighter_id").cumcount()

    evidence_columns = [
        spec.feature_name
        for spec in DYNAMIC_RESPONSE_EVIDENCE_SPECS
    ]

    valid_observation = (
        df[evidence_columns]
        .notna()
        .any(axis=1)
        .astype(int)
    )

    state_columns[
        "rfs_dynamic_response_prior_valid_observation_count"
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
                "rfs_dynamic_response_exp_",
                "rfs_dynamic_response_last3_",
                "rfs_dynamic_response_ewm_",
            )
        )
    ]

    state_frame[
        "rfs_dynamic_response_has_state"
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


def build_latest_dynamic_response_state(
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build complete Dynamic Response state for future matchups."""

    _require_columns(
        history_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *DYNAMIC_RESPONSE_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Dynamic Response history",
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

        for spec in DYNAMIC_RESPONSE_AGGREGATE_SPECS:
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

        for spec in DYNAMIC_RESPONSE_EVIDENCE_SPECS:
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
            "rfs_dynamic_response_prior_fight_count"
        ] = int(len(group))

        evidence_columns = [
            spec.feature_name
            for spec in DYNAMIC_RESPONSE_EVIDENCE_SPECS
        ]

        state[
            "rfs_dynamic_response_prior_valid_observation_count"
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
                    "rfs_dynamic_response_exp_",
                    "rfs_dynamic_response_last3_",
                    "rfs_dynamic_response_ewm_",
                )
            )
        ]

        state[
            "rfs_dynamic_response_has_state"
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
        raise RoundFighterDynamicResponseBuildError(
            "latest Dynamic Response state has duplicate fighters"
        )

    numeric_columns = [
        column
        for column in latest.columns
        if column.startswith("rfs_dynamic_response_")
    ]

    if numeric_columns and np.isinf(
        latest[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    ).any():
        raise RoundFighterDynamicResponseBuildError(
            "latest Dynamic Response state contains infinity"
        )

    return latest


def build_round_fighter_dynamic_response(
    round_stats_df: pd.DataFrame,
) -> DynamicResponseBuildResult:
    """Build history and latest Dynamic Response fighter state."""

    standardized = standardize_round_stats(
        round_stats_df
    )

    reciprocal_rounds = attach_opponent_round_values(
        standardized
    )

    observations = build_fight_level_observations(
        reciprocal_rounds
    )

    history = add_prior_dynamic_response_state(
        observations
    )

    latest = build_latest_dynamic_response_state(
        history
    )

    return DynamicResponseBuildResult(
        history=history,
        latest=latest,
    )
