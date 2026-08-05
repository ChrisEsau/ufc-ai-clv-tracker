"""Experienced-only comparison for historical simulator model candidates.

Only fights where both fighters have at least the configured number of completed
prior fights contribute to model-selection metrics. Low-experience fights remain
in separate audits and are not silently discarded from prediction outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.simulation.historical_replay_evaluation import (
    _metric_values,
    _prepare_frame,
)
from pipeline.simulation.historical_simulator_replay import (
    HistoricalSimulatorReplayError,
)


REFERENCE_VARIANT = "survival_finish_hazard_provider"
METRIC_DIRECTIONS = {
    "winner_brier": "lower",
    "winner_accuracy": "higher",
    "winner_log_loss": "lower",
    "method_log_loss": "lower",
    "method_accuracy": "higher",
    "goes_distance_brier": "lower",
    "goes_distance_accuracy": "higher",
    "fight_time_mae": "lower",
    "fighter_sig_attempt_mae": "lower",
}


@dataclass(frozen=True)
class ExperiencedModelComparisonResult:
    metrics: pd.DataFrame
    paired_deltas: pd.DataFrame
    eligible_fights: pd.DataFrame
    summary: Mapping[str, object]


def _normalize_variants(
    predictions: Mapping[str, pd.DataFrame],
    minimum_prior_fights: int,
) -> dict[str, pd.DataFrame]:
    if minimum_prior_fights <= 0:
        raise HistoricalSimulatorReplayError(
            "minimum_prior_fights must be positive"
        )
    if REFERENCE_VARIANT not in predictions:
        raise HistoricalSimulatorReplayError(
            f"Reference variant {REFERENCE_VARIANT!r} is missing"
        )

    prepared: dict[str, pd.DataFrame] = {}
    fight_sets: dict[str, set[str]] = {}
    for variant, raw in predictions.items():
        frame = _prepare_frame(raw)
        eligible = frame.loc[
            pd.to_numeric(frame["red_prior_fights"], errors="coerce").ge(
                minimum_prior_fights
            )
            & pd.to_numeric(frame["blue_prior_fights"], errors="coerce").ge(
                minimum_prior_fights
            )
        ].copy()
        if eligible.empty:
            raise HistoricalSimulatorReplayError(
                f"Experienced-only cohort is empty for {variant}"
            )
        eligible["fight_id"] = eligible["fight_id"].astype(str)
        eligible = eligible.sort_values("fight_id").reset_index(drop=True)
        prepared[variant] = eligible
        fight_sets[variant] = set(eligible["fight_id"])

    reference_set = fight_sets[REFERENCE_VARIANT]
    mismatched = {
        variant: sorted(reference_set.symmetric_difference(fights))[:10]
        for variant, fights in fight_sets.items()
        if fights != reference_set
    }
    if mismatched:
        raise HistoricalSimulatorReplayError(
            f"Experienced-only variant fight sets do not match: {mismatched}"
        )
    return prepared


def _bootstrap_deltas(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_name: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    if bootstrap_samples <= 0:
        raise HistoricalSimulatorReplayError("bootstrap_samples must be positive")
    candidate = candidate.set_index("fight_id").loc[reference["fight_id"]].reset_index()
    rng = np.random.default_rng(seed)
    indices = np.arange(len(reference))
    sampled_deltas: dict[str, list[float]] = {
        metric: [] for metric in METRIC_DIRECTIONS
    }
    for _ in range(bootstrap_samples):
        sample = rng.choice(indices, size=len(indices), replace=True)
        reference_metrics = _metric_values(reference.iloc[sample])
        candidate_metrics = _metric_values(candidate.iloc[sample])
        for metric in METRIC_DIRECTIONS:
            sampled_deltas[metric].append(
                float(candidate_metrics[metric] - reference_metrics[metric])
            )

    reference_metrics = _metric_values(reference)
    candidate_metrics = _metric_values(candidate)
    rows: list[dict[str, object]] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        delta = float(candidate_metrics[metric] - reference_metrics[metric])
        lower, upper = np.quantile(sampled_deltas[metric], [0.025, 0.975])
        favors_candidate = delta < 0 if direction == "lower" else delta > 0
        ci_supports_candidate = upper < 0 if direction == "lower" else lower > 0
        ci_supports_reference = lower > 0 if direction == "lower" else upper < 0
        rows.append(
            {
                "reference_variant": REFERENCE_VARIANT,
                "candidate_variant": candidate_name,
                "metric": metric,
                "direction": direction,
                "reference_value": float(reference_metrics[metric]),
                "candidate_value": float(candidate_metrics[metric]),
                "candidate_minus_reference": delta,
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "favors_candidate": bool(favors_candidate),
                "confidence_interval_supports_candidate": bool(
                    ci_supports_candidate
                ),
                "confidence_interval_supports_reference": bool(
                    ci_supports_reference
                ),
            }
        )
    return rows


def compare_experienced_model_candidates(
    predictions: Mapping[str, pd.DataFrame],
    minimum_prior_fights: int = 3,
    bootstrap_samples: int = 500,
    seed: int = 211,
    candidate_variants: Sequence[str] | None = None,
) -> ExperiencedModelComparisonResult:
    """Compare model candidates using only fights with two experienced fighters."""
    prepared = _normalize_variants(predictions, minimum_prior_fights)
    if candidate_variants is None:
        candidates = [
            variant for variant in prepared if variant != REFERENCE_VARIANT
        ]
    else:
        candidates = [str(variant) for variant in candidate_variants]
    missing = [variant for variant in candidates if variant not in prepared]
    if missing:
        raise HistoricalSimulatorReplayError(
            f"Candidate variants are missing: {missing}"
        )

    metric_rows: list[dict[str, object]] = []
    for variant, frame in prepared.items():
        metric_rows.append(
            {
                "variant": variant,
                "fights": int(len(frame)),
                **_metric_values(frame),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("variant").reset_index(drop=True)

    reference = prepared[REFERENCE_VARIANT]
    delta_rows: list[dict[str, object]] = []
    for index, candidate_name in enumerate(candidates):
        delta_rows.extend(
            _bootstrap_deltas(
                reference,
                prepared[candidate_name],
                candidate_name,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index * 1009,
            )
        )
    paired = pd.DataFrame(delta_rows)

    evidence_rows: list[dict[str, object]] = []
    for candidate_name, group in paired.groupby("candidate_variant", sort=True):
        evidence_rows.append(
            {
                "candidate_variant": candidate_name,
                "point_metric_wins": int(group["favors_candidate"].sum()),
                "point_metric_losses": int((~group["favors_candidate"]).sum()),
                "decisive_metric_wins": int(
                    group["confidence_interval_supports_candidate"].sum()
                ),
                "decisive_metric_losses": int(
                    group["confidence_interval_supports_reference"].sum()
                ),
            }
        )

    identity = reference[
        [
            column
            for column in (
                "fight_id",
                "date",
                "red_fighter_id",
                "red_fighter_name",
                "blue_fighter_id",
                "blue_fighter_name",
                "red_prior_fights",
                "blue_prior_fights",
                "actual_winner_corner",
                "actual_method",
            )
            if column in reference.columns
        ]
    ].copy()
    summary = {
        "status": "evaluation_only",
        "model_selection_cohort": "both_fighters_meet_prior_fight_threshold",
        "minimum_prior_fights": int(minimum_prior_fights),
        "eligible_fights": int(len(reference)),
        "reference_variant": REFERENCE_VARIANT,
        "candidate_variants": candidates,
        "bootstrap_samples": int(bootstrap_samples),
        "metrics": metrics.to_dict(orient="records"),
        "candidate_evidence": evidence_rows,
        "low_experience_fights_role": "reported_separately_not_used_for_selection",
        "probabilities_changed": False,
        "simulator_mechanics_changed": False,
    }
    return ExperiencedModelComparisonResult(
        metrics=metrics,
        paired_deltas=paired,
        eligible_fights=identity,
        summary=summary,
    )
