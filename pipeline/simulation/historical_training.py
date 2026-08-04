"""Historical eligibility layer for simulator parameter-training data.

The round-level simulation kernel currently supports standard UFC bouts scheduled
for three or five five-minute rounds. Historical source data also contains legacy,
nonstandard, and missing scheduled-round values. Those rows must be excluded
explicitly and audited rather than silently coerced or allowed to fail the entire
training build.
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


def _scheduled_round_distribution(values: pd.Series) -> dict[str, int]:
    """Return a JSON-safe distribution including missing values."""
    numeric = pd.to_numeric(values, errors="coerce")
    labels = numeric.map(lambda value: "missing" if pd.isna(value) else str(int(value)))
    counts = labels.value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def select_standard_round_history(
    round_stats_df: pd.DataFrame,
    master_df: pd.DataFrame,
    state_sources: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], HistoricalEligibilitySummary]:
    """Select only historical fights compatible with the current simulator.

    Eligibility is evaluated only among fights that have round-stat rows. Master
    fights without round data are outside this training build and are therefore
    not counted as exclusions.
    """
    required_round_columns = ["fight_id"]
    required_master_columns = ["fight_id", "total_rounds"]

    missing_round = [column for column in required_round_columns if column not in round_stats_df]
    missing_master = [column for column in required_master_columns if column not in master_df]
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
        excluded_fights=int(
            candidate_master.loc[~eligible_mask, "fight_id"].nunique()
        ),
        candidate_round_rows=int(len(round_stats_df)),
        eligible_round_rows=int(len(eligible_rounds)),
        excluded_round_rows=int(len(round_stats_df) - len(eligible_rounds)),
        scheduled_round_distribution=_scheduled_round_distribution(
            candidate_master["total_rounds"]
        ),
    )

    return eligible_rounds, eligible_master, filtered_states, summary


def _eligibility_audit(summary: HistoricalEligibilitySummary) -> pd.DataFrame:
    distribution = json.dumps(
        summary.scheduled_round_distribution,
        sort_keys=True,
    )
    rows = [
        (
            "historical_candidate_fights_with_round_data",
            summary.candidate_fights,
            summary.candidate_fights > 0,
            distribution,
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
