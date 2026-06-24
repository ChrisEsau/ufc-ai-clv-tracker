"""Segmented probability calibration utilities.

This module calibrates raw model probabilities against actual results only.
It intentionally does not use market odds or implied probabilities.

Available segmented calibrators:
- segmented_delta: subtracts learned bucket model-error deltas, optionally smoothed
- segmented_empirical: blends raw probability toward bucket actual win rate
- segmented_isotonic: experimental isotonic curves inside buckets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DEFAULT_SEGMENT_BUCKETS: list[dict[str, Any]] = [
    {"name": "very_low", "min": 0.00, "max": 0.25},
    {"name": "low", "min": 0.25, "max": 0.35},
    {"name": "dog_zone", "min": 0.35, "max": 0.45},
    {"name": "coinflip_low", "min": 0.45, "max": 0.50},
    {"name": "coinflip_high", "min": 0.50, "max": 0.55},
    {"name": "favorite_zone", "min": 0.55, "max": 0.65},
    {"name": "high", "min": 0.65, "max": 0.75},
    {"name": "very_high", "min": 0.75, "max": 1.00},
]


@dataclass(frozen=True)
class SegmentCalibrationMetadata:
    """Serializable metadata for one calibration segment."""

    name: str
    min_probability: float
    max_probability: float
    validation_rows: int
    raw_avg_probability: float | None
    actual_win_rate: float | None
    calibrated_avg_probability: float | None
    calibration_value: float | None
    raw_delta: float | None
    smoothed_delta: float | None
    shrinkage_weight: float
    used_segment_calibrator: bool
    fallback_used: bool


class SegmentedDeltaCalibrator:
    """Wrapper exposing predict_proba for dynamic delta calibration.

    Each bucket learns model error on calibration rows:

        delta = bucket_raw_avg_probability - bucket_actual_win_rate

    Prediction subtracts the bucket delta from each raw probability:

        calibrated = raw_probability - smoothed_delta

    Deltas can be smoothed across neighboring buckets to reduce noisy jumps.
    Buckets with too few rows fall back to the global delta.
    """

    def __init__(
        self,
        *,
        base_model: Any,
        buckets: list[dict[str, Any]],
        bucket_deltas: dict[str, float],
        global_delta: float,
        min_rows_per_segment: int,
        metadata: list[SegmentCalibrationMetadata],
    ) -> None:
        self.base_model = base_model
        self.buckets = buckets
        self.bucket_deltas = bucket_deltas
        self.global_delta = float(global_delta)
        self.min_rows_per_segment = int(min_rows_per_segment)
        self.metadata = metadata
        self.classes_ = getattr(base_model, "classes_", np.array([0, 1]))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_probabilities = _positive_class_probability(self.base_model, X)
        calibrated = self.calibrate_raw_probabilities(raw_probabilities)
        return np.column_stack([1.0 - calibrated, calibrated])

    def calibrate_raw_probabilities(self, raw_probabilities: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_probabilities, dtype=float)
        delta = np.full(shape=raw.shape, fill_value=self.global_delta, dtype=float)

        for bucket in self.buckets:
            name = str(bucket["name"])
            bucket_delta = self.bucket_deltas.get(name)
            if bucket_delta is None:
                continue
            mask = _bucket_mask(raw, bucket)
            if mask.any():
                delta[mask] = float(bucket_delta)

        return np.clip(raw - delta, 0.0, 1.0)

    def calibration_report(self) -> pd.DataFrame:
        return pd.DataFrame([item.__dict__ for item in self.metadata])


class SegmentedEmpiricalCalibrator:
    """Wrapper exposing predict_proba for empirical bucket calibration."""

    def __init__(
        self,
        *,
        base_model: Any,
        buckets: list[dict[str, Any]],
        bucket_values: dict[str, float],
        global_value: float,
        min_rows_per_segment: int,
        shrinkage_weight: float,
        metadata: list[SegmentCalibrationMetadata],
    ) -> None:
        self.base_model = base_model
        self.buckets = buckets
        self.bucket_values = bucket_values
        self.global_value = float(global_value)
        self.min_rows_per_segment = int(min_rows_per_segment)
        self.shrinkage_weight = float(shrinkage_weight)
        self.metadata = metadata
        self.classes_ = getattr(base_model, "classes_", np.array([0, 1]))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_probabilities = _positive_class_probability(self.base_model, X)
        calibrated = self.calibrate_raw_probabilities(raw_probabilities)
        return np.column_stack([1.0 - calibrated, calibrated])

    def calibrate_raw_probabilities(self, raw_probabilities: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_probabilities, dtype=float)
        target = np.full(shape=raw.shape, fill_value=self.global_value, dtype=float)

        for bucket in self.buckets:
            name = str(bucket["name"])
            value = self.bucket_values.get(name)
            if value is None:
                continue
            mask = _bucket_mask(raw, bucket)
            if mask.any():
                target[mask] = float(value)

        calibrated = ((1.0 - self.shrinkage_weight) * raw) + (self.shrinkage_weight * target)
        return np.clip(calibrated, 0.0, 1.0)

    def calibration_report(self) -> pd.DataFrame:
        return pd.DataFrame([item.__dict__ for item in self.metadata])


class SegmentedIsotonicCalibrator:
    """Legacy wrapper exposing predict_proba for segmented isotonic calibration."""

    def __init__(
        self,
        *,
        base_model: Any,
        global_calibrator: IsotonicRegression,
        segment_calibrators: dict[str, IsotonicRegression],
        buckets: list[dict[str, Any]],
        min_rows_per_segment: int,
        metadata: list[SegmentCalibrationMetadata],
    ) -> None:
        self.base_model = base_model
        self.global_calibrator = global_calibrator
        self.segment_calibrators = segment_calibrators
        self.buckets = buckets
        self.min_rows_per_segment = int(min_rows_per_segment)
        self.metadata = metadata
        self.classes_ = getattr(base_model, "classes_", np.array([0, 1]))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_probabilities = _positive_class_probability(self.base_model, X)
        calibrated = self.calibrate_raw_probabilities(raw_probabilities)
        return np.column_stack([1.0 - calibrated, calibrated])

    def calibrate_raw_probabilities(self, raw_probabilities: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_probabilities, dtype=float)
        calibrated = np.asarray(self.global_calibrator.predict(raw), dtype=float)

        for bucket in self.buckets:
            name = str(bucket["name"])
            segment_calibrator = self.segment_calibrators.get(name)
            if segment_calibrator is None:
                continue
            mask = _bucket_mask(raw, bucket)
            if mask.any():
                calibrated[mask] = segment_calibrator.predict(raw[mask])

        return np.clip(calibrated, 0.0, 1.0)

    def calibration_report(self) -> pd.DataFrame:
        return pd.DataFrame([item.__dict__ for item in self.metadata])


def fit_segmented_delta_calibrator(
    *,
    base_model: Any,
    raw_probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
    config: dict[str, Any] | None = None,
) -> SegmentedDeltaCalibrator:
    """Fit dynamic bucket-delta calibration using actual outcomes only."""

    calibration_config = config or {}
    buckets = calibration_config.get("buckets") or DEFAULT_SEGMENT_BUCKETS
    min_rows = int(calibration_config.get("min_rows_per_segment", 150))
    smoothing_config = calibration_config.get("smoothing") or {}
    smoothing_enabled = bool(smoothing_config.get("enabled", True))
    previous_weight = float(smoothing_config.get("previous_weight", 0.25))
    current_weight = float(smoothing_config.get("current_weight", 0.50))
    next_weight = float(smoothing_config.get("next_weight", 0.25))

    raw = np.asarray(raw_probabilities, dtype=float)
    target = np.asarray(y_true, dtype=float)
    if raw.shape[0] != target.shape[0]:
        raise ValueError("raw_probabilities and y_true must have the same length")

    global_delta = float(np.nanmean(raw) - np.nanmean(target))
    raw_deltas: dict[str, float] = {}
    row_counts: dict[str, int] = {}
    bucket_stats: list[dict[str, Any]] = []

    for bucket in buckets:
        name = str(bucket["name"])
        mask = _bucket_mask(raw, bucket)
        rows = int(mask.sum())
        use_segment = rows >= min_rows and len(np.unique(target[mask])) > 1
        raw_avg = _safe_mean(raw[mask])
        actual_rate = _safe_mean(target[mask])
        raw_delta = (float(raw_avg) - float(actual_rate)) if use_segment and raw_avg is not None and actual_rate is not None else global_delta
        raw_deltas[name] = raw_delta
        row_counts[name] = rows
        bucket_stats.append(
            {
                "bucket": bucket,
                "name": name,
                "rows": rows,
                "use_segment": use_segment,
                "raw_avg": raw_avg,
                "actual_rate": actual_rate,
                "raw_delta": raw_delta,
            }
        )

    smoothed_deltas: dict[str, float] = {}
    for idx, item in enumerate(bucket_stats):
        name = item["name"]
        if not item["use_segment"]:
            smoothed_deltas[name] = global_delta
            continue
        if not smoothing_enabled:
            smoothed_deltas[name] = float(item["raw_delta"])
            continue

        weighted_sum = current_weight * float(item["raw_delta"])
        total_weight = current_weight

        if idx > 0 and bucket_stats[idx - 1]["use_segment"]:
            weighted_sum += previous_weight * float(bucket_stats[idx - 1]["raw_delta"])
            total_weight += previous_weight
        if idx < len(bucket_stats) - 1 and bucket_stats[idx + 1]["use_segment"]:
            weighted_sum += next_weight * float(bucket_stats[idx + 1]["raw_delta"])
            total_weight += next_weight

        smoothed_deltas[name] = weighted_sum / total_weight if total_weight else float(item["raw_delta"])

    metadata: list[SegmentCalibrationMetadata] = []
    for item in bucket_stats:
        bucket = item["bucket"]
        name = item["name"]
        mask = _bucket_mask(raw, bucket)
        delta = smoothed_deltas[name]
        calibrated_values = np.clip(raw[mask] - delta, 0.0, 1.0) if item["rows"] > 0 else None
        metadata.append(
            SegmentCalibrationMetadata(
                name=name,
                min_probability=float(bucket["min"]),
                max_probability=float(bucket["max"]),
                validation_rows=int(item["rows"]),
                raw_avg_probability=item["raw_avg"],
                actual_win_rate=item["actual_rate"],
                calibrated_avg_probability=_safe_mean(calibrated_values),
                calibration_value=None,
                raw_delta=float(item["raw_delta"]),
                smoothed_delta=float(delta),
                shrinkage_weight=1.0,
                used_segment_calibrator=bool(item["use_segment"]),
                fallback_used=not bool(item["use_segment"]),
            )
        )

    return SegmentedDeltaCalibrator(
        base_model=base_model,
        buckets=[dict(item) for item in buckets],
        bucket_deltas=smoothed_deltas,
        global_delta=global_delta,
        min_rows_per_segment=min_rows,
        metadata=metadata,
    )


def fit_segmented_empirical_calibrator(
    *,
    base_model: Any,
    raw_probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
    config: dict[str, Any] | None = None,
) -> SegmentedEmpiricalCalibrator:
    """Fit empirical bucket calibration using actual outcomes only."""

    calibration_config = config or {}
    buckets = calibration_config.get("buckets") or DEFAULT_SEGMENT_BUCKETS
    min_rows = int(calibration_config.get("min_rows_per_segment", 150))
    shrinkage_weight = float(calibration_config.get("shrinkage_weight", 0.50))

    raw = np.asarray(raw_probabilities, dtype=float)
    target = np.asarray(y_true, dtype=float)
    if raw.shape[0] != target.shape[0]:
        raise ValueError("raw_probabilities and y_true must have the same length")

    global_value = float(np.nanmean(target))
    bucket_values: dict[str, float] = {}
    metadata: list[SegmentCalibrationMetadata] = []

    for bucket in buckets:
        name = str(bucket["name"])
        min_probability = float(bucket["min"])
        max_probability = float(bucket["max"])
        mask = _bucket_mask(raw, bucket)
        rows = int(mask.sum())
        use_segment = rows >= min_rows and len(np.unique(target[mask])) > 1
        bucket_actual = _safe_mean(target[mask])
        calibration_value = float(bucket_actual) if use_segment and bucket_actual is not None else global_value
        if use_segment:
            bucket_values[name] = calibration_value

        if rows > 0:
            calibrated_values = ((1.0 - shrinkage_weight) * raw[mask]) + (shrinkage_weight * calibration_value)
        else:
            calibrated_values = None

        metadata.append(
            SegmentCalibrationMetadata(
                name=name,
                min_probability=min_probability,
                max_probability=max_probability,
                validation_rows=rows,
                raw_avg_probability=_safe_mean(raw[mask]),
                actual_win_rate=bucket_actual,
                calibrated_avg_probability=_safe_mean(calibrated_values),
                calibration_value=calibration_value,
                raw_delta=None,
                smoothed_delta=None,
                shrinkage_weight=shrinkage_weight,
                used_segment_calibrator=bool(use_segment),
                fallback_used=not bool(use_segment),
            )
        )

    return SegmentedEmpiricalCalibrator(
        base_model=base_model,
        buckets=[dict(item) for item in buckets],
        bucket_values=bucket_values,
        global_value=global_value,
        min_rows_per_segment=min_rows,
        shrinkage_weight=shrinkage_weight,
        metadata=metadata,
    )


def fit_segmented_isotonic_calibrator(
    *,
    base_model: Any,
    raw_probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
    config: dict[str, Any] | None = None,
) -> SegmentedIsotonicCalibrator:
    """Fit segmented isotonic calibrators using actual outcomes only."""

    calibration_config = config or {}
    buckets = calibration_config.get("buckets") or DEFAULT_SEGMENT_BUCKETS
    min_rows = int(calibration_config.get("min_rows_per_segment", 150))

    raw = np.asarray(raw_probabilities, dtype=float)
    target = np.asarray(y_true, dtype=float)
    if raw.shape[0] != target.shape[0]:
        raise ValueError("raw_probabilities and y_true must have the same length")

    global_calibrator = _fit_isotonic(raw, target)
    global_calibrated = np.asarray(global_calibrator.predict(raw), dtype=float)

    segment_calibrators: dict[str, IsotonicRegression] = {}
    metadata: list[SegmentCalibrationMetadata] = []

    for bucket in buckets:
        name = str(bucket["name"])
        min_probability = float(bucket["min"])
        max_probability = float(bucket["max"])
        mask = _bucket_mask(raw, bucket)
        rows = int(mask.sum())
        use_segment = rows >= min_rows and len(np.unique(target[mask])) > 1

        segment_calibrated: np.ndarray | None = None
        if use_segment:
            calibrator = _fit_isotonic(raw[mask], target[mask])
            segment_calibrators[name] = calibrator
            segment_calibrated = np.asarray(calibrator.predict(raw[mask]), dtype=float)

        metadata.append(
            SegmentCalibrationMetadata(
                name=name,
                min_probability=min_probability,
                max_probability=max_probability,
                validation_rows=rows,
                raw_avg_probability=_safe_mean(raw[mask]),
                actual_win_rate=_safe_mean(target[mask]),
                calibrated_avg_probability=_safe_mean(segment_calibrated if segment_calibrated is not None else global_calibrated[mask]),
                calibration_value=None,
                raw_delta=None,
                smoothed_delta=None,
                shrinkage_weight=1.0,
                used_segment_calibrator=bool(use_segment),
                fallback_used=not bool(use_segment),
            )
        )

    return SegmentedIsotonicCalibrator(
        base_model=base_model,
        global_calibrator=global_calibrator,
        segment_calibrators=segment_calibrators,
        buckets=[dict(item) for item in buckets],
        min_rows_per_segment=min_rows,
        metadata=metadata,
    )


def _fit_isotonic(raw: np.ndarray, target: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw, target)
    return calibrator


def _positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("predict_proba must return two class columns")
    return np.asarray(probabilities[:, 1], dtype=float)


def _bucket_mask(raw: np.ndarray, bucket: dict[str, Any]) -> np.ndarray:
    lower = float(bucket["min"])
    upper = float(bucket["max"])
    if upper >= 1.0:
        return (raw >= lower) & (raw <= upper)
    return (raw >= lower) & (raw < upper)


def _safe_mean(values: np.ndarray | None) -> float | None:
    if values is None or len(values) == 0:
        return None
    return float(np.nanmean(values))
