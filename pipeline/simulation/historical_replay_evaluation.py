"""Large-cohort diagnostics for historical simulator replay variants.

This module evaluates already-generated fight-level prediction frames. It does
not alter simulator mechanics or probabilities. All diagnostics remain
shadow-only and are intended to guide subsequent component research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from pipeline.simulation.historical_simulator_replay import (
    HistoricalSimulatorReplayError,
)


RECOMMENDED_VARIANT = "survival_finish_hazard_provider"
HISTORICAL_BASELINE = "historical_baseline"
METHODS = ("decision", "ko_tko", "submission")
METHOD_PROBABILITY_COLUMNS = {
    "decision": "sim_decision_probability",
    "ko_tko": "sim_ko_tko_probability",
    "submission": "sim_submission_probability",
}
PRIMARY_METRICS = {
    "winner_brier": "lower",
    "winner_accuracy": "higher",
    "method_log_loss": "lower",
    "goes_distance_brier": "lower",
    "fight_time_mae": "lower",
    "fighter_sig_attempt_mae": "lower",
}
REQUIRED_COLUMNS = (
    "fight_id",
    "date",
    "scheduled_rounds",
    "red_prior_fights",
    "blue_prior_fights",
    "actual_winner_corner",
    "actual_method",
    "actual_fight_time_seconds",
    "actual_red_sig_attempted",
    "actual_blue_sig_attempted",
    "sim_red_win_probability",
    "sim_decision_probability",
    "sim_ko_tko_probability",
    "sim_submission_probability",
    "sim_fight_time_seconds",
    "sim_red_sig_attempted",
    "sim_blue_sig_attempted",
    "baseline_red_win_probability",
    "baseline_decision_probability",
    "baseline_ko_tko_probability",
    "baseline_submission_probability",
    "baseline_fight_time_seconds",
    "baseline_red_sig_attempted",
    "baseline_blue_sig_attempted",
)


@dataclass(frozen=True)
class HistoricalReplayEvaluationResult:
    """Artifacts produced by the large-cohort evaluator."""

    enriched_predictions: Mapping[str, pd.DataFrame]
    subgroup_metrics: pd.DataFrame
    calibration: pd.DataFrame
    bootstrap_metrics: pd.DataFrame
    paired_variant_deltas: pd.DataFrame
    stability_metrics: pd.DataFrame
    summary: Mapping[str, object]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise HistoricalSimulatorReplayError(
            f"{label} is missing required columns: {missing}"
        )


def _normalise_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "title"}


def attach_replay_metadata(
    predictions: Mapping[str, pd.DataFrame],
    training_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Attach optional event/division/title metadata without changing scores."""
    if "fight_id" not in training_df.columns:
        raise HistoricalSimulatorReplayError(
            "Simulator training table is missing fight_id metadata key"
        )

    optional = [
        column
        for column in ("event_id", "event_name", "division", "title_fight")
        if column in training_df.columns
    ]
    if not optional:
        return {name: frame.copy() for name, frame in predictions.items()}

    order = [column for column in ("date", "round", "fighter_id") if column in training_df]
    metadata = training_df[["fight_id", *optional, *order]].copy()
    if order:
        metadata = metadata.sort_values(order)
    metadata = metadata.groupby("fight_id", as_index=False, sort=False)[optional].first()

    enriched: dict[str, pd.DataFrame] = {}
    for variant, frame in predictions.items():
        current = frame.drop(columns=[column for column in optional if column in frame], errors="ignore")
        enriched[variant] = current.merge(
            metadata,
            on="fight_id",
            how="left",
            validate="one_to_one",
        )
    return enriched


def _historical_baseline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame.copy()
    replacements = {
        "sim_red_win_probability": "baseline_red_win_probability",
        "sim_decision_probability": "baseline_decision_probability",
        "sim_ko_tko_probability": "baseline_ko_tko_probability",
        "sim_submission_probability": "baseline_submission_probability",
        "sim_fight_time_seconds": "baseline_fight_time_seconds",
        "sim_red_sig_attempted": "baseline_red_sig_attempted",
        "sim_blue_sig_attempted": "baseline_blue_sig_attempted",
    }
    for target, source in replacements.items():
        baseline[target] = baseline[source]
    baseline["simulator_version"] = HISTORICAL_BASELINE
    return baseline


def _experience_band(minimum_prior_fights: pd.Series) -> pd.Series:
    values = pd.to_numeric(minimum_prior_fights, errors="coerce").fillna(0)
    conditions = [values.eq(0), values.between(1, 2), values.between(3, 5)]
    labels = ["0_cold_start", "1_2_prior", "3_5_prior"]
    return pd.Series(
        np.select(conditions, labels, default="6_plus_prior"),
        index=values.index,
        dtype="string",
    )


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, REQUIRED_COLUMNS, "Historical prediction frame")
    result = frame.copy()
    if result.empty:
        raise HistoricalSimulatorReplayError("Historical prediction frame is empty")
    if result["fight_id"].duplicated().any():
        raise HistoricalSimulatorReplayError(
            "Historical prediction frame contains duplicate fight_id values"
        )

    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise HistoricalSimulatorReplayError(
            "Historical prediction frame contains invalid dates"
        )
    result["sim_red_win_probability"] = pd.to_numeric(
        result["sim_red_win_probability"], errors="coerce"
    )
    if (
        result["sim_red_win_probability"].isna().any()
        or (~result["sim_red_win_probability"].between(0.0, 1.0)).any()
    ):
        raise HistoricalSimulatorReplayError(
            "Winner probabilities must be finite values between zero and one"
        )

    result["actual_red_win"] = result["actual_winner_corner"].eq("red").astype(float)
    result["predicted_red_win"] = result["sim_red_win_probability"].ge(0.5)
    result["winner_correct"] = result["predicted_red_win"].eq(
        result["actual_red_win"].astype(bool)
    )
    result["predicted_winner_probability"] = np.maximum(
        result["sim_red_win_probability"],
        1.0 - result["sim_red_win_probability"],
    )
    result["confidence_band"] = pd.cut(
        result["predicted_winner_probability"],
        bins=[0.50, 0.55, 0.60, 0.70, 0.80, 1.0000001],
        labels=["50_55", "55_60", "60_70", "70_80", "80_100"],
        include_lowest=True,
        right=False,
    ).astype("string")
    result["red_probability_band"] = pd.cut(
        result["sim_red_win_probability"],
        bins=np.linspace(0.0, 1.0, 11),
        labels=[f"{start:02d}_{start + 10:02d}" for start in range(0, 100, 10)],
        include_lowest=True,
    ).astype("string")

    result["minimum_prior_fights"] = result[
        ["red_prior_fights", "blue_prior_fights"]
    ].min(axis=1)
    result["combined_prior_fights"] = result[
        ["red_prior_fights", "blue_prior_fights"]
    ].sum(axis=1)
    result["experience_band"] = _experience_band(result["minimum_prior_fights"])
    result["month"] = result["date"].dt.to_period("M").astype("string")

    division = result.get("division", pd.Series("unknown", index=result.index))
    division = division.astype("string").fillna("unknown").str.strip()
    division = division.mask(division.eq(""), "unknown")
    result["division"] = division
    lower_division = division.str.lower()
    result["sex_segment_inferred"] = np.where(
        division.eq("unknown"),
        "unknown",
        np.where(
            lower_division.str.contains("women|female", regex=True, na=False),
            "women",
            "men",
        ),
    )
    title_values = result.get("title_fight", pd.Series(False, index=result.index))
    result["title_fight"] = title_values.map(_normalise_bool)

    event_name = result.get("event_name", pd.Series(pd.NA, index=result.index)).astype("string")
    event_id = result.get("event_id", pd.Series(pd.NA, index=result.index)).astype("string")
    result["event_group"] = event_name.fillna(event_id).fillna(
        result["date"].dt.strftime("%Y-%m-%d")
    )
    result["scheduled_rounds"] = pd.to_numeric(
        result["scheduled_rounds"], errors="coerce"
    ).astype("Int64")
    return result


def _metric_values(frame: pd.DataFrame) -> dict[str, float]:
    epsilon = 1e-12
    actual_red = frame["actual_red_win"].to_numpy(dtype=float)
    red_probability = frame["sim_red_win_probability"].to_numpy(dtype=float)
    clipped_red = np.clip(red_probability, epsilon, 1.0 - epsilon)

    method_probabilities = frame[
        [METHOD_PROBABILITY_COLUMNS[method] for method in METHODS]
    ].to_numpy(dtype=float)
    method_probabilities = np.clip(method_probabilities, epsilon, None)
    method_probabilities = method_probabilities / method_probabilities.sum(
        axis=1, keepdims=True
    )
    actual_method_index = frame["actual_method"].map(
        {method: index for index, method in enumerate(METHODS)}
    )
    if actual_method_index.isna().any():
        raise HistoricalSimulatorReplayError(
            "Historical prediction frame contains an unsupported actual method"
        )
    method_index = actual_method_index.to_numpy(dtype=int)
    predicted_method_index = np.argmax(method_probabilities, axis=1)

    actual_distance = frame["actual_method"].eq("decision").to_numpy(dtype=float)
    distance_probability = frame["sim_decision_probability"].to_numpy(dtype=float)
    red_strike_error = np.abs(
        frame["sim_red_sig_attempted"].to_numpy(dtype=float)
        - frame["actual_red_sig_attempted"].to_numpy(dtype=float)
    )
    blue_strike_error = np.abs(
        frame["sim_blue_sig_attempted"].to_numpy(dtype=float)
        - frame["actual_blue_sig_attempted"].to_numpy(dtype=float)
    )
    fight_time_error = (
        frame["sim_fight_time_seconds"].to_numpy(dtype=float)
        - frame["actual_fight_time_seconds"].to_numpy(dtype=float)
    )

    return {
        "winner_brier": float(np.mean((red_probability - actual_red) ** 2)),
        "winner_accuracy": float(
            np.mean((red_probability >= 0.5) == actual_red.astype(bool))
        ),
        "winner_log_loss": float(
            -np.mean(
                actual_red * np.log(clipped_red)
                + (1.0 - actual_red) * np.log(1.0 - clipped_red)
            )
        ),
        "method_log_loss": float(
            -np.mean(np.log(method_probabilities[np.arange(len(frame)), method_index]))
        ),
        "method_accuracy": float(np.mean(predicted_method_index == method_index)),
        "goes_distance_brier": float(
            np.mean((distance_probability - actual_distance) ** 2)
        ),
        "goes_distance_accuracy": float(
            np.mean((distance_probability >= 0.5) == actual_distance.astype(bool))
        ),
        "fight_time_mae": float(np.mean(np.abs(fight_time_error))),
        "fight_time_bias": float(np.mean(fight_time_error)),
        "fighter_sig_attempt_mae": float(
            np.mean(np.concatenate([red_strike_error, blue_strike_error]))
        ),
        "mean_pick_confidence": float(
            frame["predicted_winner_probability"].mean()
        ),
    }


def _metric_row(
    frame: pd.DataFrame,
    variant: str,
    segment: str,
    subgroup: object,
    minimum_group_size: int,
) -> dict[str, object]:
    return {
        "variant": variant,
        "segment": segment,
        "subgroup": str(subgroup),
        "fights": int(len(frame)),
        "meets_minimum_group_size": bool(len(frame) >= minimum_group_size),
        **_metric_values(frame),
    }


def _subgroup_metrics(
    frames: Mapping[str, pd.DataFrame],
    minimum_group_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    segment_columns = {
        "experience": "experience_band",
        "confidence": "confidence_band",
        "scheduled_rounds": "scheduled_rounds",
        "division": "division",
        "sex_segment_inferred": "sex_segment_inferred",
        "title_fight": "title_fight",
        "actual_method": "actual_method",
        "month": "month",
        "event": "event_group",
    }
    for variant, frame in frames.items():
        rows.append(
            _metric_row(frame, variant, "overall", "all", minimum_group_size)
        )
        for segment, column in segment_columns.items():
            for subgroup, subset in frame.groupby(column, dropna=False, sort=True):
                rows.append(
                    _metric_row(
                        subset,
                        variant,
                        segment,
                        subgroup,
                        minimum_group_size,
                    )
                )
    return pd.DataFrame(rows).sort_values(
        ["segment", "subgroup", "variant"]
    ).reset_index(drop=True)


def _calibration_tables(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, frame in frames.items():
        for band, subset in frame.groupby("red_probability_band", dropna=False):
            if subset.empty:
                continue
            predicted = float(subset["sim_red_win_probability"].mean())
            actual = float(subset["actual_red_win"].mean())
            rows.append(
                {
                    "variant": variant,
                    "calibration_type": "red_win_probability",
                    "band": str(band),
                    "fights": int(len(subset)),
                    "mean_probability": predicted,
                    "observed_rate": actual,
                    "absolute_gap": abs(predicted - actual),
                }
            )
        for band, subset in frame.groupby("confidence_band", dropna=False):
            if subset.empty:
                continue
            predicted = float(subset["predicted_winner_probability"].mean())
            actual = float(subset["winner_correct"].mean())
            rows.append(
                {
                    "variant": variant,
                    "calibration_type": "predicted_winner_confidence",
                    "band": str(band),
                    "fights": int(len(subset)),
                    "mean_probability": predicted,
                    "observed_rate": actual,
                    "absolute_gap": abs(predicted - actual),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["calibration_type", "band", "variant"]
    ).reset_index(drop=True)


def _bootstrap_metrics(
    frames: Mapping[str, pd.DataFrame],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    if bootstrap_samples <= 0:
        raise HistoricalSimulatorReplayError("bootstrap_samples must be positive")
    rows: list[dict[str, object]] = []
    for variant_index, (variant, frame) in enumerate(frames.items()):
        point = _metric_values(frame)
        samples = {metric: [] for metric in PRIMARY_METRICS}
        rng = np.random.default_rng(seed + variant_index * 1009)
        for _ in range(bootstrap_samples):
            positions = rng.integers(0, len(frame), size=len(frame))
            values = _metric_values(frame.iloc[positions])
            for metric in PRIMARY_METRICS:
                samples[metric].append(values[metric])
        for metric, direction in PRIMARY_METRICS.items():
            distribution = np.asarray(samples[metric], dtype=float)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "direction": direction,
                    "estimate": point[metric],
                    "ci_lower_95": float(np.quantile(distribution, 0.025)),
                    "ci_upper_95": float(np.quantile(distribution, 0.975)),
                    "bootstrap_samples": int(bootstrap_samples),
                    "fights": int(len(frame)),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "variant"]).reset_index(
        drop=True
    )


def _paired_variant_deltas(
    frames: Mapping[str, pd.DataFrame],
    reference_variant: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    if reference_variant not in frames:
        raise HistoricalSimulatorReplayError(
            f"Reference replay variant is missing: {reference_variant}"
        )
    reference = frames[reference_variant].sort_values("fight_id").reset_index(drop=True)
    reference_ids = reference["fight_id"].astype("string").tolist()
    rows: list[dict[str, object]] = []

    for comparator_index, (comparator_name, comparator_frame) in enumerate(frames.items()):
        if comparator_name == reference_variant:
            continue
        comparator = comparator_frame.sort_values("fight_id").reset_index(drop=True)
        if comparator["fight_id"].astype("string").tolist() != reference_ids:
            raise HistoricalSimulatorReplayError(
                f"Variant fight set differs from {reference_variant}: {comparator_name}"
            )
        reference_point = _metric_values(reference)
        comparator_point = _metric_values(comparator)
        distributions = {metric: [] for metric in PRIMARY_METRICS}
        rng = np.random.default_rng(seed + comparator_index * 2027)
        for _ in range(bootstrap_samples):
            positions = rng.integers(0, len(reference), size=len(reference))
            reference_values = _metric_values(reference.iloc[positions])
            comparator_values = _metric_values(comparator.iloc[positions])
            for metric in PRIMARY_METRICS:
                distributions[metric].append(
                    reference_values[metric] - comparator_values[metric]
                )
        for metric, direction in PRIMARY_METRICS.items():
            point_delta = reference_point[metric] - comparator_point[metric]
            distribution = np.asarray(distributions[metric], dtype=float)
            lower = float(np.quantile(distribution, 0.025))
            upper = float(np.quantile(distribution, 0.975))
            favors_reference = point_delta < 0 if direction == "lower" else point_delta > 0
            supported = upper < 0 if direction == "lower" else lower > 0
            rows.append(
                {
                    "reference_variant": reference_variant,
                    "comparator_variant": comparator_name,
                    "metric": metric,
                    "direction": direction,
                    "reference_estimate": reference_point[metric],
                    "comparator_estimate": comparator_point[metric],
                    "reference_minus_comparator": point_delta,
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "favors_reference": bool(favors_reference),
                    "confidence_interval_supports_reference": bool(supported),
                    "bootstrap_samples": int(bootstrap_samples),
                    "fights": int(len(reference)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["metric", "comparator_variant"]
    ).reset_index(drop=True)


def evaluate_historical_replay_cohort(
    predictions: Mapping[str, pd.DataFrame],
    training_df: pd.DataFrame,
    recommended_variant: str = RECOMMENDED_VARIANT,
    minimum_group_size: int = 10,
    bootstrap_samples: int = 500,
    seed: int = 91,
) -> HistoricalReplayEvaluationResult:
    """Evaluate replay variants on identical fights with subgroup diagnostics."""
    if minimum_group_size <= 0:
        raise HistoricalSimulatorReplayError("minimum_group_size must be positive")
    if not predictions:
        raise HistoricalSimulatorReplayError("No replay prediction frames were provided")

    enriched = attach_replay_metadata(predictions, training_df)
    prepared = {name: _prepare_frame(frame) for name, frame in enriched.items()}
    first = next(iter(prepared.values()))
    prepared[HISTORICAL_BASELINE] = _prepare_frame(_historical_baseline_frame(first))

    reference_ids = sorted(first["fight_id"].astype("string").tolist())
    for variant, frame in prepared.items():
        if sorted(frame["fight_id"].astype("string").tolist()) != reference_ids:
            raise HistoricalSimulatorReplayError(
                f"Replay variants do not share the same fight set: {variant}"
            )

    subgroup = _subgroup_metrics(prepared, minimum_group_size)
    calibration = _calibration_tables(prepared)
    bootstrap = _bootstrap_metrics(
        prepared,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    paired = _paired_variant_deltas(
        prepared,
        reference_variant=recommended_variant,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 7919,
    )
    stability = subgroup.loc[
        subgroup["segment"].isin(["month", "event"])
    ].reset_index(drop=True)

    experience_counts = (
        prepared[recommended_variant]["experience_band"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    overall = subgroup.loc[
        subgroup["segment"].eq("overall") & subgroup["subgroup"].eq("all")
    ]
    overall_metrics = {
        row["variant"]: {
            metric: float(row[metric])
            for metric in PRIMARY_METRICS
        }
        for _, row in overall.iterrows()
    }
    supported = paired.loc[
        paired["confidence_interval_supports_reference"]
    ][["comparator_variant", "metric"]].to_dict(orient="records")
    summary = {
        "status": "shadow_only_research_guidance",
        "decision_unit": "large_historical_cohort",
        "recommended_variant": recommended_variant,
        "fights": int(len(first)),
        "variants": list(prepared),
        "minimum_group_size": int(minimum_group_size),
        "bootstrap_samples": int(bootstrap_samples),
        "experience_counts": {
            str(key): int(value) for key, value in experience_counts.items()
        },
        "overall_metrics": overall_metrics,
        "confidence_supported_recommended_improvements": supported,
        "single_card_role": "smoke_test_only",
    }
    return HistoricalReplayEvaluationResult(
        enriched_predictions=prepared,
        subgroup_metrics=subgroup,
        calibration=calibration,
        bootstrap_metrics=bootstrap,
        paired_variant_deltas=paired,
        stability_metrics=stability,
        summary=summary,
    )
