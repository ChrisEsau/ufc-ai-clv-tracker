"""Segmented probability calibration utilities.

This module calibrates raw model probabilities against actual results only.
It intentionally does not use market odds or implied probabilities.
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
    used_segment_calibrator: bool
    fallback_used: bool


class SegmentedIsotonicCalibrator:
    """Wrapper exposing predict_proba for segmented isotonic calibration.

    The wrapper owns the fitted base model plus global and per-segment isotonic
    calibrators. It first gets raw probabilities from the base model, assigns
    each row to a raw-probability bucket, and then applies the matching segment
    calibrator. Segments with too few validation rows fall back to the global
    isotonic calibrator.
    """

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
