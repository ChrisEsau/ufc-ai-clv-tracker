"""Historical eligibility and source-boundary rules for simulator training data.

The round-level simulation kernel currently supports standard UFC bouts scheduled
for three or five five-minute rounds. Historical source data also contains legacy,
nonstandard, and missing scheduled-round values. Those rows are excluded and
audited rather than silently coerced.

Historical round files may also contain scraper metadata, duplicate fight-level
context, and target-round phase statistics that are not registered simulator
targets. This module removes those columns before they can reach the modeling
table while preserving the canonical observations required to construct targets
and prior-round context inside ``training_dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

import pandas as pd

from pipeline.simulation.training_dataset import (
    SimulationTrainingBuildResult,
    SimulationTrainingDataError,
    build_simulation_training_dataset,
)


SUPPORTED_SCHEDULED_ROUNDS = frozenset({3, 5})

MASTER_AUTHORITATIVE_CONTEXT_COLUMNS = (
    "division",
    "title_fight",
    "method",
    "method_family",
    "finish_round",
    "match_time_sec",
    "total_rounds",
    "winner_id",
)

UNREGISTERED_TARGET_ROUND_OBSERVATION_COLUMNS = (
    "head_landed",
    "head_attempted",
    "body_landed",
    "body_attempted",
    "leg_landed",
    "leg_attempted",
    "distance_landed",
    "distance_attempted",
    "clinch_landed",
    "clinch_attempted",
)

CONTROL_SECONDS_SOURCE_COLUMNS = (
    "control_seconds",
    "ctrl_seconds",
    "ctrl_sec",
    "control_time_seconds",
    "control_time_sec",
    "control_time",
    "ctrl",
    "control",
)

SOURCE_ONLY_ROUND_COLUMNS = (
    "ctrl_sec",
    "ctrl_seconds",
    "control_time",
    "control_time_sec",
    "control_time_seconds",
    "ctrl",
    "control",
    "event_date",
    "event_url",
    "fight_url",
    "fighter_url",
    "opponent_url",
    "fight_order",
)


@dataclass(frozen=True)
class HistoricalEligibilitySummary:
    """Counts describing the historical rows admitted to the simulator build."""

    candidate_fights: int
    eligible_fights: int
    excluded_fights: int
    candidate_round_rows: int
    eligible_round_rows: int
    excluded_round_rows: int
    scheduled_round_distribution: Mapping[str, int]
    control_seconds_source_column: str
    dropped_round_context_columns: tuple[str, ...]
    dropped_target_round_observation_columns: tuple[str, ...]
    dropped_source_only_columns: tuple[str, ...]


def _scheduled_round_distribution(values: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = numeric.map(lambda value: "missing" if pd.isna(value) else str(int(value)))
    counts = labels.value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _present_columns(df: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in candidates if column in df.columns)


def _canonicalize_control_seconds(rounds: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Create the canonical control_seconds observation before aliases are removed."""
    out = rounds.copy()
    source = next(
        (column for column in CONTROL_SECONDS_SOURCE_COLUMNS if column in out.columns),
        None,
    )
    if source is None:
        raise SimulationTrainingDataError(
            "round stats is missing control time; checked: "
            f"{list(CONTROL_SECONDS_SOURCE_COLUMNS)}"
        )
    if "control_seconds" not in out.columns:
        out["control_seconds"] = out[source]
    return out, source


def select_standard_round_history(
    round_stats_df: pd.DataFrame,
    master_df: pd.DataFrame,
    state_sources: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], HistoricalEligibilitySummary]:
    """Select simulator-compatible fights and enforce the historical source boundary."""
    missing_round = [column for column in ["fight_id"] if column not in round_stats_df]
    missing_master = [
        column for column in ["fight_id", "total_rounds"] if column not in master_df
    ]
    if missing_round:
        raise SimulationTrainingDataError(
            f"round stats is missing historical eligibility columns: {missing_round}"
        )
    if missing_master:
        raise SimulationTrainingDataError(
            f"master fights is missing historical eligibility columns: {missing_master}"
        )

    round_fight_ids = pd.Index(round_stats_df["fight_id"].dropna().unique())
    candidate_master = master_df[master_df["fight_id"].isin(round_fight_ids)].copy()

    master_fight_ids = pd.Index(candidate_master["fight_id"].dropna().unique())
    missing_from_master = round_fight_ids.difference(master_fight_ids)
    if len(missing_from_master):
        sample = [str(value) for value in missing_from_master[:10].tolist()]
        raise SimulationTrainingDataError(
            "round stats contains fights missing from master during eligibility selection. "
            f"Count: {len(missing_from_master)}. Sample: {sample}"
        )

    scheduled_rounds = pd.to_numeric(candidate_master["total_rounds"], errors="coerce")
    eligible_mask = scheduled_rounds.isin(SUPPORTED_SCHEDULED_ROUNDS)
    eligible_master = candidate_master.loc[eligible_mask].copy()
    eligible_fight_ids = pd.Index(eligible_master["fight_id"].dropna().unique())

    if eligible_master.empty:
        raise SimulationTrainingDataError(
            "No historical fights with supported scheduled rounds (3 or 5) were found."
        )

    eligible_rounds = round_stats_df[
        round_stats_df["fight_id"].isin(eligible_fight_ids)
    ].copy()
    eligible_rounds, control_seconds_source = _canonicalize_control_seconds(
        eligible_rounds
    )

    dropped_round_context_columns = _present_columns(
        eligible_rounds,
        MASTER_AUTHORITATIVE_CONTEXT_COLUMNS,
    )
    dropped_target_round_observation_columns = _present_columns(
        eligible_rounds,
        UNREGISTERED_TARGET_ROUND_OBSERVATION_COLUMNS,
    )
    dropped_source_only_columns = _present_columns(
        eligible_rounds,
        SOURCE_ONLY_ROUND_COLUMNS,
    )

    columns_to_drop = sorted(
        set(dropped_round_context_columns)
        | set(dropped_target_round_observation_columns)
        | set(dropped_source_only_columns)
    )
    if columns_to_drop:
        eligible_rounds = eligible_rounds.drop(columns=columns_to_drop)

    filtered_states: dict[str, pd.DataFrame] = {}
    for source_name, source_df in (state_sources or {}).items():
        if "fight_id" not in source_df.columns:
            raise SimulationTrainingDataError(
                f"state source {source_name!r} is missing fight_id"
            )
        filtered_states[source_name] = source_df[
            source_df["fight_id"].isin(eligible_fight_ids)
        ].copy()

    summary = HistoricalEligibilitySummary(
        candidate_fights=int(candidate_master["fight_id"].nunique()),
        eligible_fights=int(eligible_master["fight_id"].nunique()),
        excluded_fights=int(candidate_master.loc[~eligible_mask, "fight_id"].nunique()),
        candidate_round_rows=int(len(round_stats_df)),
        eligible_round_rows=int(len(eligible_rounds)),
        excluded_round_rows=int(len(round_stats_df) - len(eligible_rounds)),
        scheduled_round_distribution=_scheduled_round_distribution(
            candidate_master["total_rounds"]
        ),
        control_seconds_source_column=control_seconds_source,
        dropped_round_context_columns=dropped_round_context_columns,
        dropped_target_round_observation_columns=dropped_target_round_observation_columns,
        dropped_source_only_columns=dropped_source_only_columns,
    )

    return eligible_rounds, eligible_master, filtered_states, summary


def _eligibility_audit(summary: HistoricalEligibilitySummary) -> pd.DataFrame:
    rows = [
        (
            "historical_candidate_fights_with_round_data",
            summary.candidate_fights,
            summary.candidate_fights > 0,
            json.dumps(summary.scheduled_round_distribution, sort_keys=True),
        ),
        (
            "historical_eligible_standard_round_fights",
            summary.eligible_fights,
            summary.eligible_fights > 0,
            "Supported scheduled rounds: 3 or 5",
        ),
        (
            "historical_excluded_nonstandard_round_fights",
            summary.excluded_fights,
            True,
            "Intentionally excluded; values are not coerced",
        ),
        (
            "historical_candidate_fighter_round_rows",
            summary.candidate_round_rows,
            summary.candidate_round_rows > 0,
            "",
        ),
        (
            "historical_eligible_fighter_round_rows",
            summary.eligible_round_rows,
            summary.eligible_round_rows > 0,
            "",
        ),
        (
            "historical_excluded_nonstandard_fighter_round_rows",
            summary.excluded_round_rows,
            True,
            "Intentionally excluded with their parent fights",
        ),
        (
            "control_seconds_source_column",
            1,
            True,
            summary.control_seconds_source_column,
        ),
        (
            "round_context_columns_dropped_in_favor_of_master",
            len(summary.dropped_round_context_columns),
            True,
            json.dumps(list(summary.dropped_round_context_columns)),
        ),
        (
            "unregistered_target_round_observations_dropped",
            len(summary.dropped_target_round_observation_columns),
            True,
            json.dumps(list(summary.dropped_target_round_observation_columns)),
        ),
        (
            "source_only_round_columns_dropped",
            len(summary.dropped_source_only_columns),
            True,
            json.dumps(list(summary.dropped_source_only_columns)),
        ),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "passed", "detail"])


def build_historical_simulation_training_dataset(
    round_stats_df: pd.DataFrame,
    master_df: pd.DataFrame,
    state_sources: Mapping[str, pd.DataFrame] | None = None,
) -> SimulationTrainingBuildResult:
    """Build a strict training table after audited historical eligibility filtering."""
    eligible_rounds, eligible_master, eligible_states, eligibility = (
        select_standard_round_history(
            round_stats_df=round_stats_df,
            master_df=master_df,
            state_sources=state_sources,
        )
    )

    result = build_simulation_training_dataset(
        round_stats_df=eligible_rounds,
        master_df=eligible_master,
        state_sources=eligible_states,
    )
    audit = pd.concat(
        [_eligibility_audit(eligibility), result.audit],
        ignore_index=True,
    )
    return SimulationTrainingBuildResult(dataset=result.dataset, audit=audit)
