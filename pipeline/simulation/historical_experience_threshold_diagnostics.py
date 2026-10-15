"""Evaluation-only audit for low historical fighter experience.

A fight is flagged when either fighter has fewer than the configured number of
completed prior fights. The audit does not alter probabilities or simulator
mechanics; it reports performance with all fights, flagged fights only, and the
cohort remaining after flagged fights are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from pipeline.simulation.historical_replay_evaluation import (
    _metric_values,
    _prepare_frame,
)
from pipeline.simulation.historical_simulator_replay import (
    HistoricalSimulatorReplayError,
)


@dataclass(frozen=True)
class HistoricalExperienceThresholdResult:
    flagged_predictions: pd.DataFrame
    metrics: pd.DataFrame
    summary: Mapping[str, object]


def _cohort_row(frame: pd.DataFrame, cohort: str) -> dict[str, object]:
    if frame.empty:
        raise HistoricalSimulatorReplayError(
            f"Experience threshold cohort {cohort!r} is empty"
        )
    return {
        "cohort": cohort,
        "fights": int(len(frame)),
        **_metric_values(frame),
    }


def audit_experience_threshold(
    predictions: pd.DataFrame,
    minimum_prior_fights: int = 3,
) -> HistoricalExperienceThresholdResult:
    """Flag low-experience fights and compare included/excluded performance."""
    if minimum_prior_fights <= 0:
        raise HistoricalSimulatorReplayError(
            "minimum_prior_fights must be positive"
        )

    frame = _prepare_frame(predictions)
    threshold = int(minimum_prior_fights)
    frame["red_below_experience_threshold"] = (
        pd.to_numeric(frame["red_prior_fights"], errors="coerce") < threshold
    )
    frame["blue_below_experience_threshold"] = (
        pd.to_numeric(frame["blue_prior_fights"], errors="coerce") < threshold
    )
    if frame[
        ["red_below_experience_threshold", "blue_below_experience_threshold"]
    ].isna().any().any():
        raise HistoricalSimulatorReplayError(
            "Experience threshold audit contains invalid prior-fight values"
        )

    frame["low_experience_flag"] = (
        frame["red_below_experience_threshold"]
        | frame["blue_below_experience_threshold"]
    )
    frame["low_experience_reason"] = "neither"
    frame.loc[
        frame["red_below_experience_threshold"]
        & ~frame["blue_below_experience_threshold"],
        "low_experience_reason",
    ] = "red_only"
    frame.loc[
        ~frame["red_below_experience_threshold"]
        & frame["blue_below_experience_threshold"],
        "low_experience_reason",
    ] = "blue_only"
    frame.loc[
        frame["red_below_experience_threshold"]
        & frame["blue_below_experience_threshold"],
        "low_experience_reason",
    ] = "both"
    frame["experience_filter_group"] = frame["low_experience_flag"].map(
        {
            True: f"flagged_either_under_{threshold}",
            False: f"both_fighters_{threshold}_plus",
        }
    )

    flagged = frame.loc[frame["low_experience_flag"]].copy()
    experienced = frame.loc[~frame["low_experience_flag"]].copy()
    metrics = pd.DataFrame(
        [
            _cohort_row(frame, "all_fights_included"),
            _cohort_row(flagged, f"flagged_either_under_{threshold}"),
            _cohort_row(experienced, f"excluding_flagged_both_{threshold}_plus"),
        ]
    )

    all_row = metrics.loc[metrics["cohort"].eq("all_fights_included")].iloc[0]
    experienced_row = metrics.loc[
        metrics["cohort"].eq(f"excluding_flagged_both_{threshold}_plus")
    ].iloc[0]
    metric_columns = [
        column
        for column in metrics.columns
        if column not in {"cohort", "fights"}
    ]
    deltas = {
        metric: float(experienced_row[metric] - all_row[metric])
        for metric in metric_columns
    }

    summary = {
        "status": "evaluation_only",
        "flag_rule": (
            f"red_prior_fights < {threshold} or blue_prior_fights < {threshold}"
        ),
        "minimum_prior_fights": threshold,
        "total_fights": int(len(frame)),
        "flagged_fights": int(len(flagged)),
        "flagged_rate": float(len(flagged) / len(frame)),
        "experienced_only_fights": int(len(experienced)),
        "flag_reasons": {
            str(reason): int(count)
            for reason, count in frame.loc[
                frame["low_experience_flag"], "low_experience_reason"
            ].value_counts().items()
        },
        "metrics": metrics.to_dict(orient="records"),
        "experienced_only_minus_all_deltas": deltas,
        "probabilities_changed": False,
        "simulator_mechanics_changed": False,
    }
    return HistoricalExperienceThresholdResult(
        flagged_predictions=frame.sort_values(["date", "fight_id"]).reset_index(
            drop=True
        ),
        metrics=metrics,
        summary=summary,
    )
