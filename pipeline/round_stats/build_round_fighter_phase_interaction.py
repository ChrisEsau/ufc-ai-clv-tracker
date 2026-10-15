"""Build simulator-oriented RFS Phase Interaction fighter state.

The builder converts authoritative reciprocal UFCStats round rows into:

1. One current-fight Phase Interaction observation per fighter-fight.
2. Leakage-safe point-in-time fighter history.
3. Latest fighter state for future matchup simulation.

Opponent aggregates are copied from the opponent's independently aggregated
fighter row. They are not inferred or recalculated from the current fighter's
statistics. This preserves exact reciprocal matchup symmetry.

This module does not write parquet artifacts or modify production features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.round_stats.rfs_phase_interaction_feature_contracts import (
    PHASE_INTERACTION_AGGREGATE_SPECS,
    PHASE_INTERACTION_EVIDENCE_SPECS,
    PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS,
    PHASE_INTERACTION_PREFIX,
    InteractionAggregateRule,
    PhaseInteractionFormula,
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
    "distance_landed",
    "distance_attempted",
    "clinch_landed",
    "clinch_attempted",
    "ground_landed",
    "ground_attempted",
    "td_landed",
    "td_attempted",
    "sub_att",
    "rev",
    "ctrl_sec",
)

NUMERIC_ROUND_COLUMNS = (
    "total_rounds",
    "round",
    "sig_str_attempted",
    "distance_landed",
    "distance_attempted",
    "clinch_landed",
    "clinch_attempted",
    "ground_landed",
    "ground_attempted",
    "td_landed",
    "td_attempted",
    "sub_att",
    "rev",
    "ctrl_sec",
)

COUNT_COLUMNS = (
    "sig_str_attempted",
    "distance_landed",
    "distance_attempted",
    "clinch_landed",
    "clinch_attempted",
    "ground_landed",
    "ground_attempted",
    "td_landed",
    "td_attempted",
    "sub_att",
    "rev",
    "ctrl_sec",
)

LANDED_ATTEMPTED_PAIRS = (
    ("distance_landed", "distance_attempted"),
    ("clinch_landed", "clinch_attempted"),
    ("ground_landed", "ground_attempted"),
    ("td_landed", "td_attempted"),
)

EWM_ALPHA = 0.35


@dataclass(frozen=True)
class PhaseInteractionBuildResult:
    """History and latest-state outputs from one interaction build."""

    history: pd.DataFrame
    latest: pd.DataFrame


class RoundFighterPhaseInteractionBuildError(RuntimeError):
    """Raised when Phase Interaction state cannot be built safely."""


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
        raise RoundFighterPhaseInteractionBuildError(
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

    if not feature_name.startswith(PHASE_INTERACTION_PREFIX):
        raise RoundFighterPhaseInteractionBuildError(
            "Unexpected Phase Interaction feature name: "
            f"{feature_name}"
        )

    return feature_name.removeprefix(
        PHASE_INTERACTION_PREFIX
    )


def prior_total_name(feature_name: str) -> str:
    """Return the leakage-safe cumulative-total state name."""

    return (
        "rfs_phase_interact_prior_total_"
        f"{_fight_suffix(feature_name)}"
    )


def evidence_state_name(
    feature_name: str,
    state_kind: str,
) -> str:
    """Return an exp, last3, or ewm interaction-state name."""

    if state_kind not in {"exp", "last3", "ewm"}:
        raise ValueError(
            "state_kind must be exp, last3, or ewm"
        )

    return (
        f"rfs_phase_interact_{state_kind}_"
        f"{_fight_suffix(feature_name)}"
    )


def standardize_round_stats(
    round_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize and validate authoritative reciprocal round rows."""

    df = round_stats_df.copy()

    if "date" not in df.columns:
        if "event_date" not in df.columns:
            raise RoundFighterPhaseInteractionBuildError(
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
        raise RoundFighterPhaseInteractionBuildError(
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
        raise RoundFighterPhaseInteractionBuildError(
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
        raise RoundFighterPhaseInteractionBuildError(
            "Round stats contain negative values in: "
            f"{negative_columns}"
        )

    for landed, attempted in LANDED_ATTEMPTED_PAIRS:
        if (df[landed] > df[attempted]).any():
            raise RoundFighterPhaseInteractionBuildError(
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
        raise RoundFighterPhaseInteractionBuildError(
            "Round stats contain duplicate "
            "fight_id + fighter_id + round rows."
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
    """Require exactly two reciprocal fighter perspectives per fight."""

    for fight_id, fight in df.groupby(
        "fight_id",
        dropna=False,
        sort=False,
    ):
        fighter_ids = fight["fighter_id"].drop_duplicates()

        if len(fighter_ids) != 2:
            raise RoundFighterPhaseInteractionBuildError(
                "Each fight must contain exactly two fighters: "
                f"fight_id={fight_id}, "
                f"fighter_count={len(fighter_ids)}."
            )

        round_sets = {
            fighter_id: set(
                fighter_rows["round"].tolist()
            )
            for fighter_id, fighter_rows in fight.groupby(
                "fighter_id",
                sort=False,
            )
        }

        if len({frozenset(value) for value in round_sets.values()}) != 1:
            raise RoundFighterPhaseInteractionBuildError(
                "Reciprocal fighters do not contain identical "
                f"recorded rounds for fight_id={fight_id}."
            )

        fighter_pairs = (
            fight[
                [
                    "fighter_id",
                    "opponent_id",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if len(fighter_pairs) != 2:
            raise RoundFighterPhaseInteractionBuildError(
                "Fighter-opponent identity is inconsistent within "
                f"fight_id={fight_id}."
            )

        first = fighter_pairs.iloc[0]
        second = fighter_pairs.iloc[1]

        reciprocal = (
            first["fighter_id"] == second["opponent_id"]
            and first["opponent_id"] == second["fighter_id"]
        )

        if not reciprocal:
            raise RoundFighterPhaseInteractionBuildError(
                "Fighter-opponent rows are not reciprocal for "
                f"fight_id={fight_id}."
            )


def _metadata_value(
    group: pd.DataFrame,
    column: str,
) -> object:
    """Return one metadata value after consistency validation."""

    unique_values = group[column].drop_duplicates()

    if len(unique_values) != 1:
        raise RoundFighterPhaseInteractionBuildError(
            "Inconsistent fight metadata for "
            f"fight_id={group['fight_id'].iloc[0]}, "
            f"fighter_id={group['fighter_id'].iloc[0]}, "
            f"column={column}."
        )

    return unique_values.iloc[0]


def _build_fighter_aggregates(
    group: pd.DataFrame,
) -> dict[str, float]:
    """Build the fighter-perspective aggregate contract values."""

    output: dict[str, float] = {}

    for spec in PHASE_INTERACTION_AGGREGATE_SPECS:
        if spec.rule is InteractionAggregateRule.UNIQUE_COUNT:
            value = float(
                group[spec.source_column].nunique()
            )

        elif spec.rule is InteractionAggregateRule.SUM:
            value = float(
                pd.to_numeric(
                    group[spec.source_column],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )

        else:
            raise RoundFighterPhaseInteractionBuildError(
                f"Unsupported aggregate rule: {spec.rule}"
            )

        output[spec.feature_name] = value

    return output


def build_fighter_aggregate_rows(
    rounds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one fighter-only aggregate row per fighter-fight."""

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
            column: _metadata_value(group, column)
            for column in FIGHT_METADATA_COLUMNS
        }

        row.update(_build_fighter_aggregates(group))
        rows.append(row)

    fighter_aggregate_columns = [
        spec.feature_name
        for spec in PHASE_INTERACTION_AGGREGATE_SPECS
    ]

    aggregates = pd.DataFrame(
        rows,
        columns=[
            *FIGHT_METADATA_COLUMNS,
            *fighter_aggregate_columns,
        ],
    )

    if aggregates.empty:
        return aggregates

    duplicate_mask = aggregates.duplicated(
        subset=[
            "fight_id",
            "fighter_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise RoundFighterPhaseInteractionBuildError(
            "Fighter aggregates are not unique at "
            "fight_id + fighter_id grain."
        )

    return aggregates


def attach_reciprocal_opponent_aggregates(
    fighter_aggregates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact aggregates from each row's reciprocal opponent."""

    fighter_columns = [
        spec.feature_name
        for spec in PHASE_INTERACTION_AGGREGATE_SPECS
    ]

    _require_columns(
        fighter_aggregates_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *fighter_columns,
        ),
        "fighter aggregates",
    )

    opponent_lookup = fighter_aggregates_df[
        [
            "fight_id",
            "fighter_id",
            *fighter_columns,
        ]
    ].copy()

    opponent_lookup = opponent_lookup.rename(
        columns={
            "fighter_id": "_matched_opponent_id",
            **{
                spec.feature_name: spec.opponent_feature_name
                for spec in PHASE_INTERACTION_AGGREGATE_SPECS
            },
        }
    )

    merged = fighter_aggregates_df.merge(
        opponent_lookup,
        how="left",
        left_on=[
            "fight_id",
            "opponent_id",
        ],
        right_on=[
            "fight_id",
            "_matched_opponent_id",
        ],
        validate="one_to_one",
    )

    if merged["_matched_opponent_id"].isna().any():
        bad_rows = merged.loc[
            merged["_matched_opponent_id"].isna(),
            [
                "fight_id",
                "fighter_id",
                "opponent_id",
            ],
        ]

        raise RoundFighterPhaseInteractionBuildError(
            "Unable to match reciprocal opponent aggregates: "
            f"{bad_rows.to_dict(orient='records')}"
        )

    if not (
        merged["_matched_opponent_id"]
        == merged["opponent_id"]
    ).all():
        raise RoundFighterPhaseInteractionBuildError(
            "Matched opponent identity does not equal opponent_id."
        )

    return merged.drop(
        columns=["_matched_opponent_id"]
    )


def _aggregate_input(
    aggregates: dict[str, float],
    feature_name: str,
) -> float:
    """Read one formula input with a clear contract error."""

    if feature_name not in aggregates:
        raise RoundFighterPhaseInteractionBuildError(
            "Formula expected unknown aggregate feature: "
            f"{feature_name}"
        )

    return float(aggregates[feature_name])


def _build_evidence_value(
    *,
    aggregates: dict[str, float],
    formula: PhaseInteractionFormula,
    input_features: tuple[str, ...],
) -> float:
    """Calculate one locked Phase Interaction evidence value."""

    values = [
        _aggregate_input(aggregates, feature_name)
        for feature_name in input_features
    ]

    if formula in {
        PhaseInteractionFormula.SAFE_RATIO,
        PhaseInteractionFormula.PHASE_ATTEMPT_SHARE,
        PhaseInteractionFormula.PER_OBSERVED_ROUND,
    }:
        return safe_scalar_div(
            values[0],
            values[1],
        )

    if formula is PhaseInteractionFormula.COMPLEMENT_RATIO:
        ratio = safe_scalar_div(
            values[0],
            values[1],
        )

        if pd.isna(ratio):
            return np.nan

        return float(1.0 - ratio)

    if (
        formula
        is PhaseInteractionFormula.NON_DISTANCE_ATTEMPT_SHARE
    ):
        return safe_scalar_div(
            values[0] + values[1],
            values[2],
        )

    if formula is PhaseInteractionFormula.SHARE_OF_COMBINED:
        return safe_scalar_div(
            values[0],
            values[0] + values[1],
        )

    if formula is PhaseInteractionFormula.DIFFERENCE_PER_ROUND:
        return safe_scalar_div(
            values[0] - values[1],
            values[2],
        )

    if formula is PhaseInteractionFormula.PER_CONTROL_MINUTE:
        return safe_scalar_div(
            values[0],
            values[1] / 60.0,
        )

    if (
        formula
        is PhaseInteractionFormula.COMBINED_PER_CONTROL_MINUTE
    ):
        return safe_scalar_div(
            values[0] + values[1],
            (values[2] + values[3]) / 60.0,
        )

    if formula is PhaseInteractionFormula.BALANCE_INDEX:
        total = values[0] + values[1]

        if total == 0:
            return np.nan

        return float(
            1.0
            - abs(values[0] - values[1]) / total
        )

    raise RoundFighterPhaseInteractionBuildError(
        f"Unsupported evidence formula: {formula}"
    )


def _build_evidence(
    aggregates: dict[str, float],
) -> dict[str, float]:
    """Build all locked fight-level interaction evidence."""

    output: dict[str, float] = {}

    for spec in PHASE_INTERACTION_EVIDENCE_SPECS:
        value = _build_evidence_value(
            aggregates=aggregates,
            formula=spec.formula,
            input_features=spec.input_features,
        )

        if pd.notna(value) and not np.isfinite(value):
            raise RoundFighterPhaseInteractionBuildError(
                "Evidence calculation produced a non-finite value: "
                f"{spec.feature_name}={value}"
            )

        if (
            spec.unit_interval
            and pd.notna(value)
            and not 0.0 <= value <= 1.0
        ):
            raise RoundFighterPhaseInteractionBuildError(
                "Unit-interval evidence is out of range: "
                f"{spec.feature_name}={value}"
            )

        output[spec.feature_name] = value

    return output


def _validate_reciprocal_observations(
    observations: pd.DataFrame,
) -> None:
    """Require exact mirrored aggregates across reciprocal rows."""

    indexed = observations.set_index(
        [
            "fight_id",
            "fighter_id",
        ],
        drop=False,
    )

    for row in observations.itertuples(index=False):
        opponent_key = (
            row.fight_id,
            row.opponent_id,
        )

        if opponent_key not in indexed.index:
            raise RoundFighterPhaseInteractionBuildError(
                "Observation is missing reciprocal opponent row: "
                f"fight_id={row.fight_id}, "
                f"fighter_id={row.fighter_id}, "
                f"opponent_id={row.opponent_id}."
            )

        opponent_row = indexed.loc[opponent_key]

        if isinstance(opponent_row, pd.DataFrame):
            raise RoundFighterPhaseInteractionBuildError(
                "Reciprocal opponent lookup returned multiple rows."
            )

        for spec in PHASE_INTERACTION_AGGREGATE_SPECS:
            fighter_value = getattr(
                row,
                spec.feature_name,
            )
            mirrored_value = opponent_row[
                spec.opponent_feature_name
            ]

            opponent_value = getattr(
                row,
                spec.opponent_feature_name,
            )
            reciprocal_fighter_value = opponent_row[
                spec.feature_name
            ]

            if not np.isclose(
                fighter_value,
                mirrored_value,
                equal_nan=True,
            ):
                raise RoundFighterPhaseInteractionBuildError(
                    "Reciprocal aggregate mismatch for "
                    f"{spec.feature_name}, "
                    f"fight_id={row.fight_id}."
                )

            if not np.isclose(
                opponent_value,
                reciprocal_fighter_value,
                equal_nan=True,
            ):
                raise RoundFighterPhaseInteractionBuildError(
                    "Reciprocal opponent aggregate mismatch for "
                    f"{spec.opponent_feature_name}, "
                    f"fight_id={row.fight_id}."
                )


def build_fight_level_observations(
    rounds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one Phase Interaction observation per fighter-fight."""

    fighter_aggregates = build_fighter_aggregate_rows(
        rounds_df
    )

    mirrored = attach_reciprocal_opponent_aggregates(
        fighter_aggregates
    )

    rows: list[dict[str, object]] = []

    for record in mirrored.to_dict(orient="records"):
        aggregate_values = {
            feature_name: float(record[feature_name])
            for feature_name
            in PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS
            if feature_name in record
        }

        evidence = _build_evidence(
            aggregate_values
        )

        row = {
            column: record[column]
            for column in FIGHT_METADATA_COLUMNS
        }

        row.update(aggregate_values)
        row.update(evidence)
        rows.append(row)

    observations = pd.DataFrame(
        rows,
        columns=[
            *FIGHT_METADATA_COLUMNS,
            *PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS,
        ],
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
        raise RoundFighterPhaseInteractionBuildError(
            "Fight observations are not unique at "
            "fight_id + fighter_id grain."
        )

    _validate_reciprocal_observations(observations)

    return observations.sort_values(
        [
            "fighter_id",
            "date",
            "fight_id",
        ]
    ).reset_index(drop=True)


def add_prior_phase_interaction_state(
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe prior totals and rolling evidence state."""

    _require_columns(
        observations_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Phase Interaction observations",
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

    # Aggregate opportunity/exposure counts become cumulative prior totals.
    for spec in PHASE_INTERACTION_AGGREGATE_SPECS:
        for fight_column in (
            spec.feature_name,
            spec.opponent_feature_name,
        ):
            state_columns[
                prior_total_name(fight_column)
            ] = grouped[fight_column].transform(
                cumulative_prior_total
            )

    # Derived evidence receives expanding, last-three, and EWM prior state.
    for spec in PHASE_INTERACTION_EVIDENCE_SPECS:
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
        "rfs_phase_interact_prior_fight_count"
    ] = df.groupby("fighter_id").cumcount()

    evidence_columns = [
        spec.feature_name
        for spec in PHASE_INTERACTION_EVIDENCE_SPECS
    ]

    valid_observation = (
        df[evidence_columns]
        .notna()
        .any(axis=1)
        .astype(int)
    )

    state_columns[
        "rfs_phase_interact_prior_valid_observation_count"
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
                "rfs_phase_interact_exp_",
                "rfs_phase_interact_last3_",
                "rfs_phase_interact_ewm_",
            )
        )
    ]

    state_frame[
        "rfs_phase_interact_has_state"
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


def build_latest_phase_interaction_state(
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build complete interaction state for future matchups."""

    _require_columns(
        history_df,
        (
            *FIGHT_METADATA_COLUMNS,
            *PHASE_INTERACTION_FIGHT_OBSERVATION_COLUMNS,
        ),
        "Phase Interaction history",
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
            "rfs_phase_interact_prior_fight_count": len(group),
        }

        evidence_columns = [
            spec.feature_name
            for spec in PHASE_INTERACTION_EVIDENCE_SPECS
        ]

        state[
            "rfs_phase_interact_prior_valid_observation_count"
        ] = int(
            group[evidence_columns]
            .notna()
            .any(axis=1)
            .sum()
        )

        for spec in PHASE_INTERACTION_AGGREGATE_SPECS:
            for fight_column in (
                spec.feature_name,
                spec.opponent_feature_name,
            ):
                values = pd.to_numeric(
                    group[fight_column],
                    errors="coerce",
                ).fillna(0.0)

                state[
                    prior_total_name(fight_column)
                ] = float(values.sum())

        for spec in PHASE_INTERACTION_EVIDENCE_SPECS:
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
                values.expanding(min_periods=1)
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
                    "rfs_phase_interact_exp_",
                    "rfs_phase_interact_last3_",
                    "rfs_phase_interact_ewm_",
                )
            )
        ]

        state["rfs_phase_interact_has_state"] = int(
            any(
                pd.notna(state[column])
                for column in rolling_state_columns
            )
        )

        latest_rows.append(state)

    return pd.DataFrame(latest_rows)


def build_round_fighter_phase_interaction(
    round_stats_df: pd.DataFrame,
) -> PhaseInteractionBuildResult:
    """Build history and latest Phase Interaction fighter state."""

    standardized = standardize_round_stats(
        round_stats_df
    )

    observations = build_fight_level_observations(
        standardized
    )

    history = add_prior_phase_interaction_state(
        observations
    )

    latest = build_latest_phase_interaction_state(
        history
    )

    return PhaseInteractionBuildResult(
        history=history,
        latest=latest,
    )
