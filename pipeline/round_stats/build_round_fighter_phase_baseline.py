"""Build simulator-oriented RFS Phase Baseline fighter state.

The builder converts authoritative UFCStats round rows into:

1. Current-fight Phase Baseline observations.
2. Leakage-safe point-in-time fighter history.
3. Latest fighter state for future matchup simulation.

Observable statistics remain distinct from final simulator parameters. Exact
distance, clinch, and ground exposure is not available in the round source, so
this builder does not fabricate phase-conditional 30-second rates.

This module does not write parquet artifacts or modify production feature views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.round_stats.rfs_phase_baseline_feature_contracts import (
    PHASE_BASELINE_AGGREGATE_SPECS,
    PHASE_BASELINE_EVIDENCE_SPECS,
    PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS,
    PHASE_BASELINE_PREFIX,
    FightAggregateRule,
    PhaseBaselineFormula,
)


FIGHT_METADATA_COLUMNS = (
    "event_id",
    "event_name",
    "date",
    "fight_id",
    "corner",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "division",
    "total_rounds",
)

REQUIRED_ROUND_COLUMNS = (
    "event_id",
    "event_name",
    "fight_id",
    "corner",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "division",
    "total_rounds",
    "round",
    "sig_str_attempted",
    "distance_attempted",
    "clinch_attempted",
    "ground_attempted",
    "td_landed",
    "td_attempted",
    "ctrl_sec",
)

NUMERIC_ROUND_COLUMNS = (
    "total_rounds",
    "round",
    "sig_str_attempted",
    "distance_attempted",
    "clinch_attempted",
    "ground_attempted",
    "td_landed",
    "td_attempted",
    "ctrl_sec",
)

COUNT_COLUMNS = (
    "sig_str_attempted",
    "distance_attempted",
    "clinch_attempted",
    "ground_attempted",
    "td_landed",
    "td_attempted",
    "ctrl_sec",
)

LANDED_ATTEMPTED_PAIRS = (
    ("td_landed", "td_attempted"),
)

EWM_ALPHA = 0.35


@dataclass(frozen=True)
class PhaseBaselineBuildResult:
    """History and latest-state outputs from one Phase Baseline build."""

    history: pd.DataFrame
    latest: pd.DataFrame


class RoundFighterPhaseBaselineBuildError(RuntimeError):
    """Raised when Phase Baseline state cannot be built safely."""


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
        raise RoundFighterPhaseBaselineBuildError(
            f"{label} is missing required columns: {missing}"
        )


def safe_scalar_div(
    numerator: float,
    denominator: float,
) -> float:
    """Return a finite scalar ratio or NaN for an invalid denominator."""

    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    value = numerator / denominator

    if np.isfinite(value):
        return float(value)

    return np.nan


def ols_slope(
    x_values: pd.Series,
    y_values: pd.Series,
) -> float:
    """Return the OLS slope for ordered round values."""

    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x_values,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y_values,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(frame) < 2:
        return np.nan

    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)

    if np.allclose(np.var(x), 0.0):
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
    """Cumulative opportunity total excluding the current fight."""

    values = (
        pd.to_numeric(series, errors="coerce")
        .fillna(0.0)
    )

    return values.shift(1, fill_value=0.0).cumsum()


def _fight_suffix(feature_name: str) -> str:
    """Remove the locked current-fight prefix."""

    if not feature_name.startswith(PHASE_BASELINE_PREFIX):
        raise RoundFighterPhaseBaselineBuildError(
            "Unexpected Phase Baseline feature name: "
            f"{feature_name}"
        )

    return feature_name.removeprefix(PHASE_BASELINE_PREFIX)


def prior_total_name(feature_name: str) -> str:
    """Return the leakage-safe cumulative-total state name."""

    return (
        "rfs_phase_base_prior_total_"
        f"{_fight_suffix(feature_name)}"
    )


def evidence_state_name(
    feature_name: str,
    state_kind: str,
) -> str:
    """Return an exp, last3, or ewm state feature name."""

    if state_kind not in {"exp", "last3", "ewm"}:
        raise ValueError(
            "state_kind must be exp, last3, or ewm"
        )

    return (
        f"rfs_phase_base_{state_kind}_"
        f"{_fight_suffix(feature_name)}"
    )


def standardize_round_stats(
    round_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize and validate authoritative round-stat rows."""

    df = round_stats_df.copy()

    if "date" not in df.columns:
        if "event_date" not in df.columns:
            raise RoundFighterPhaseBaselineBuildError(
                "Round stats must include date or event_date."
            )

        df["date"] = pd.to_datetime(
            df["event_date"],
            errors="coerce",
            format="mixed",
        )
    else:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            format="mixed",
        )

    _require_columns(
        df,
        (
            *REQUIRED_ROUND_COLUMNS,
            "date",
        ),
        "round stats",
    )

    if df["date"].isna().any():
        bad_rows = int(df["date"].isna().sum())
        raise RoundFighterPhaseBaselineBuildError(
            "Round stats contain "
            f"{bad_rows} rows with invalid dates."
        )

    df["corner"] = (
        df["corner"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    invalid_corners = sorted(
        set(df["corner"]) - {"red", "blue"}
    )

    if invalid_corners:
        raise RoundFighterPhaseBaselineBuildError(
            "Round stats contain invalid corners: "
            f"{invalid_corners}"
        )

    for column in NUMERIC_ROUND_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0.0)

    negative_columns = [
        column
        for column in COUNT_COLUMNS
        if (df[column] < 0).any()
    ]

    if negative_columns:
        raise RoundFighterPhaseBaselineBuildError(
            "Round stats contain negative values in: "
            f"{negative_columns}"
        )

    for landed, attempted in LANDED_ATTEMPTED_PAIRS:
        if (df[landed] > df[attempted]).any():
            raise RoundFighterPhaseBaselineBuildError(
                f"{landed} cannot exceed {attempted}."
            )

    duplicate_mask = df.duplicated(
        subset=[
            "fight_id",
            "fighter_id",
            "round",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise RoundFighterPhaseBaselineBuildError(
            "Round stats contain duplicate "
            "fight_id + fighter_id + round rows."
        )

    return df.sort_values(
        [
            "fighter_id",
            "date",
            "fight_id",
            "round",
        ]
    ).reset_index(drop=True)


def _metadata_value(
    group: pd.DataFrame,
    column: str,
) -> object:
    """Return one metadata value after consistency validation."""

    unique_values = group[column].drop_duplicates()

    if len(unique_values) != 1:
        raise RoundFighterPhaseBaselineBuildError(
            "Inconsistent fight metadata for "
            f"fight_id={group['fight_id'].iloc[0]}, "
            f"fighter_id={group['fighter_id'].iloc[0]}, "
            f"column={column}."
        )

    return unique_values.iloc[0]


def _build_aggregates(
    group: pd.DataFrame,
) -> dict[str, float]:
    """Build the nine locked fight-level aggregate values."""

    output: dict[str, float] = {}

    for spec in PHASE_BASELINE_AGGREGATE_SPECS:
        if spec.rule is FightAggregateRule.UNIQUE_COUNT:
            source = spec.source_columns[0]
            value = float(group[source].nunique())

        elif spec.rule is FightAggregateRule.SUM:
            source = spec.source_columns[0]
            value = float(
                pd.to_numeric(
                    group[source],
                    errors="coerce",
                ).fillna(0.0).sum()
            )

        elif (
            spec.rule
            is FightAggregateRule
            .SUM_DIFFERENCE_CLIPPED_ZERO
        ):
            minuend, subtrahend = spec.source_columns
            differences = (
                pd.to_numeric(
                    group[minuend],
                    errors="coerce",
                ).fillna(0.0)
                - pd.to_numeric(
                    group[subtrahend],
                    errors="coerce",
                ).fillna(0.0)
            ).clip(lower=0.0)

            value = float(differences.sum())

        else:
            raise RoundFighterPhaseBaselineBuildError(
                f"Unsupported aggregate rule: {spec.rule}"
            )

        output[spec.feature_name] = value

    return output


def _aggregate_input(
    aggregates: dict[str, float],
    feature_name: str,
) -> float:
    """Read one aggregate input with a clear contract error."""

    if feature_name not in aggregates:
        raise RoundFighterPhaseBaselineBuildError(
            "Formula expected unknown aggregate feature: "
            f"{feature_name}"
        )

    return float(aggregates[feature_name])


def _build_evidence_value(
    *,
    group: pd.DataFrame,
    aggregates: dict[str, float],
    formula: PhaseBaselineFormula,
    input_features: tuple[str, ...],
) -> float:
    """Calculate one locked derived evidence value."""

    if formula in {
        PhaseBaselineFormula.PER_OBSERVED_ROUND,
        PhaseBaselineFormula.SAFE_RATIO,
        PhaseBaselineFormula.PHASE_ATTEMPT_SHARE,
    }:
        numerator = _aggregate_input(
            aggregates,
            input_features[0],
        )
        denominator = _aggregate_input(
            aggregates,
            input_features[1],
        )
        return safe_scalar_div(
            numerator,
            denominator,
        )

    if (
        formula
        is PhaseBaselineFormula
        .NON_DISTANCE_PHASE_SHARE
    ):
        numerator = _aggregate_input(
            aggregates,
            input_features[0],
        )
        other_phase = _aggregate_input(
            aggregates,
            input_features[1],
        )
        return safe_scalar_div(
            numerator,
            numerator + other_phase,
        )

    if formula is PhaseBaselineFormula.PER_CONTROL_MINUTE:
        numerator = _aggregate_input(
            aggregates,
            input_features[0],
        )
        control_seconds = _aggregate_input(
            aggregates,
            input_features[1],
        )
        return safe_scalar_div(
            numerator,
            control_seconds / 60.0,
        )

    if formula is PhaseBaselineFormula.OLS_ROUND_SLOPE:
        ordered = group.sort_values("round")

        if input_features == (
            "round",
            "td_attempted",
        ):
            y_values = ordered["td_attempted"]

        elif input_features == (
            "round",
            "td_attempted",
            "td_landed",
        ):
            y_values = (
                ordered["td_attempted"]
                - ordered["td_landed"]
            ).clip(lower=0.0)

        else:
            raise RoundFighterPhaseBaselineBuildError(
                "Unsupported OLS input contract: "
                f"{input_features}"
            )

        return ols_slope(
            ordered["round"],
            y_values,
        )

    if (
        formula
        is PhaseBaselineFormula
        .FIRST_LAST_PERSISTENCE_RATIO
    ):
        ordered = group.sort_values("round")

        if input_features != (
            "round",
            "td_attempted",
        ):
            raise RoundFighterPhaseBaselineBuildError(
                "Unsupported persistence input contract: "
                f"{input_features}"
            )

        first_attempts = float(
            ordered["td_attempted"].iloc[0]
        )
        last_attempts = float(
            ordered["td_attempted"].iloc[-1]
        )

        return safe_scalar_div(
            last_attempts,
            max(first_attempts, 1.0),
        )

    raise RoundFighterPhaseBaselineBuildError(
        f"Unsupported evidence formula: {formula}"
    )


def _build_evidence(
    group: pd.DataFrame,
    aggregates: dict[str, float],
) -> dict[str, float]:
    """Build all 19 locked fight-level evidence features."""

    output: dict[str, float] = {}

    for spec in PHASE_BASELINE_EVIDENCE_SPECS:
        value = _build_evidence_value(
            group=group,
            aggregates=aggregates,
            formula=spec.formula,
            input_features=spec.input_features,
        )

        if (
            spec.unit_interval
            and pd.notna(value)
            and not 0.0 <= value <= 1.0
        ):
            raise RoundFighterPhaseBaselineBuildError(
                "Unit-interval evidence is out of range: "
                f"{spec.feature_name}={value}"
            )

        output[spec.feature_name] = value

    return output


def build_fight_level_observations(
    rounds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one Phase Baseline observation row per fighter-fight."""

    rows: list[dict[str, object]] = []

    grouped = rounds_df.groupby(
        [
            "fight_id",
            "fighter_id",
        ],
        dropna=False,
        sort=False,
    )

    for _, group in grouped:
        group = group.sort_values("round").copy()

        row: dict[str, object] = {
            column: _metadata_value(
                group,
                column,
            )
            for column in FIGHT_METADATA_COLUMNS
        }

        aggregates = _build_aggregates(group)
        evidence = _build_evidence(
            group,
            aggregates,
        )

        row.update(aggregates)
        row.update(evidence)
        rows.append(row)

    output_columns = [
        *FIGHT_METADATA_COLUMNS,
        *PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS,
    ]

    observations = pd.DataFrame(
        rows,
        columns=output_columns,
    )

    if observations.empty:
        return observations

    duplicate_mask = observations.duplicated(
        subset=[
            "fight_id",
            "fighter_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise RoundFighterPhaseBaselineBuildError(
            "Fight observations are not unique at "
            "fight_id + fighter_id grain."
        )

    return observations.sort_values(
        [
            "fighter_id",
            "date",
            "fight_id",
        ]
    ).reset_index(drop=True)


def add_prior_phase_baseline_state(
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe prior totals and rolling evidence state."""

    _require_columns(
        observations_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Phase Baseline observations",
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

    # Opportunity/exposure values are cumulative prior totals.
    for spec in PHASE_BASELINE_AGGREGATE_SPECS:
        fight_column = spec.feature_name
        state_column = prior_total_name(
            fight_column
        )

        df[state_column] = grouped[
            fight_column
        ].transform(cumulative_prior_total)

    # Derived evidence retains expanding, recent, and EWM states.
    for spec in PHASE_BASELINE_EVIDENCE_SPECS:
        fight_column = spec.feature_name

        df[
            evidence_state_name(
                fight_column,
                "exp",
            )
        ] = grouped[fight_column].transform(
            expanding_prior
        )

        df[
            evidence_state_name(
                fight_column,
                "last3",
            )
        ] = grouped[fight_column].transform(
            last3_prior
        )

        df[
            evidence_state_name(
                fight_column,
                "ewm",
            )
        ] = grouped[fight_column].transform(
            ewm_prior
        )

    df["rfs_phase_base_prior_fight_count"] = (
        df.groupby("fighter_id").cumcount()
    )

    valid_observation = (
        df[
            [
                spec.feature_name
                for spec in PHASE_BASELINE_EVIDENCE_SPECS
            ]
        ]
        .notna()
        .any(axis=1)
        .astype(int)
    )

    df[
        "rfs_phase_base_prior_valid_observation_count"
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

    rolling_state_columns = [
        column
        for column in df.columns
        if column.startswith(
            (
                "rfs_phase_base_exp_",
                "rfs_phase_base_last3_",
                "rfs_phase_base_ewm_",
            )
        )
    ]

    df["rfs_phase_base_has_state"] = (
        df[rolling_state_columns]
        .notna()
        .any(axis=1)
        .astype(int)
    )

    return df


def build_latest_phase_baseline_state(
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build complete current fighter state for future matchups."""

    _require_columns(
        history_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *PHASE_BASELINE_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Phase Baseline history",
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
            "rfs_phase_base_prior_fight_count": len(group),
        }

        evidence_columns = [
            spec.feature_name
            for spec in PHASE_BASELINE_EVIDENCE_SPECS
        ]

        state[
            "rfs_phase_base_prior_valid_observation_count"
        ] = int(
            group[evidence_columns]
            .notna()
            .any(axis=1)
            .sum()
        )

        for spec in PHASE_BASELINE_AGGREGATE_SPECS:
            values = pd.to_numeric(
                group[spec.feature_name],
                errors="coerce",
            ).fillna(0.0)

            state[
                prior_total_name(
                    spec.feature_name
                )
            ] = float(values.sum())

        for spec in PHASE_BASELINE_EVIDENCE_SPECS:
            values = pd.to_numeric(
                group[spec.feature_name],
                errors="coerce",
            )

            state[
                evidence_state_name(
                    spec.feature_name,
                    "exp",
                )
            ] = (
                values.expanding(
                    min_periods=1
                )
                .mean()
                .iloc[-1]
            )

            state[
                evidence_state_name(
                    spec.feature_name,
                    "last3",
                )
            ] = values.tail(3).mean()

            state[
                evidence_state_name(
                    spec.feature_name,
                    "ewm",
                )
            ] = (
                values.ewm(
                    alpha=EWM_ALPHA,
                    adjust=False,
                )
                .mean()
                .iloc[-1]
            )

        rolling_state_columns = [
            column
            for column in state
            if column.startswith(
                (
                    "rfs_phase_base_exp_",
                    "rfs_phase_base_last3_",
                    "rfs_phase_base_ewm_",
                )
            )
        ]

        state["rfs_phase_base_has_state"] = int(
            any(
                pd.notna(state[column])
                for column in rolling_state_columns
            )
        )

        latest_rows.append(state)

    return pd.DataFrame(latest_rows)


def build_round_fighter_phase_baseline(
    round_stats_df: pd.DataFrame,
) -> PhaseBaselineBuildResult:
    """Build history and latest Phase Baseline fighter state."""

    standardized = standardize_round_stats(
        round_stats_df
    )
    observations = build_fight_level_observations(
        standardized
    )
    history = add_prior_phase_baseline_state(
        observations
    )
    latest = build_latest_phase_baseline_state(
        history
    )

    return PhaseBaselineBuildResult(
        history=history,
        latest=latest,
    )
