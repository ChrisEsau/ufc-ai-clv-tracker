"""Model calibration utilities for UFC model training.

Calibration is model-agnostic. Any classifier that exposes ``predict_proba`` can
be calibrated with this module.

Supported methods:
- isotonic: flexible non-parametric calibration used by the current notebook
- segmented_empirical: actual-results empirical bucket calibration with shrinkage
- segmented_isotonic: experimental actual-results isotonic calibration by bucket
- sigmoid / platt: logistic calibration, more stable with smaller samples
- none: no calibration; raw model probabilities are passed through
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from pipeline.training.segmented_calibration import (
    fit_segmented_empirical_calibrator,
    fit_segmented_isotonic_calibrator,
)

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - compatibility for older sklearn versions
    FrozenEstimator = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CalibrationResult:
    """Container for calibration artifacts and probabilities."""

    calibrator: Any | None
    method: str
    raw_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    n_calibration_rows: int


def calibrate_model(
    model: Any,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    method: str = "isotonic",
    config: dict[str, Any] | None = None,
) -> CalibrationResult:
    """Fit and apply a probability calibration layer.

    Parameters
    ----------
    model:
        Fitted classifier exposing ``predict_proba``.
    X_calibration:
        Feature matrix used to fit the calibration layer.
    y_calibration:
        Binary labels for calibration.
    method:
        ``isotonic``, ``segmented_empirical``, ``segmented_isotonic``,
        ``sigmoid``/``platt``, or ``none``.
    config:
        Optional calibration config. Used by segmented calibration for bucket
        definitions and minimum row thresholds.
    """
    normalized_method = method.strip().lower()
    raw_probabilities = predict_positive_class_probability(model, X_calibration)

    if normalized_method in {"none", "raw", "disabled"}:
        return CalibrationResult(
            calibrator=None,
            method="none",
            raw_probabilities=raw_probabilities,
            calibrated_probabilities=raw_probabilities,
            n_calibration_rows=len(X_calibration),
        )

    if normalized_method in {"segmented_empirical", "bucketed_empirical", "segmented_bin_rate"}:
        calibrator = fit_segmented_empirical_calibrator(
            base_model=model,
            raw_probabilities=raw_probabilities,
            y_true=y_calibration,
            config=config or {},
        )
        calibrated_probabilities = predict_positive_class_probability(calibrator, X_calibration)
        return CalibrationResult(
            calibrator=calibrator,
            method="segmented_empirical",
            raw_probabilities=raw_probabilities,
            calibrated_probabilities=calibrated_probabilities,
            n_calibration_rows=len(X_calibration),
        )

    if normalized_method in {"segmented_isotonic", "bucketed_isotonic"}:
        calibrator = fit_segmented_isotonic_calibrator(
            base_model=model,
            raw_probabilities=raw_probabilities,
            y_true=y_calibration,
            config=config or {},
        )
        calibrated_probabilities = predict_positive_class_probability(calibrator, X_calibration)
        return CalibrationResult(
            calibrator=calibrator,
            method="segmented_isotonic",
            raw_probabilities=raw_probabilities,
            calibrated_probabilities=calibrated_probabilities,
            n_calibration_rows=len(X_calibration),
        )

    sklearn_method = _normalize_sklearn_calibration_method(normalized_method)

    calibrator = _build_prefit_calibrator(model=model, method=sklearn_method)
    calibrator.fit(X_calibration, y_calibration)
    calibrated_probabilities = predict_positive_class_probability(calibrator, X_calibration)

    return CalibrationResult(
        calibrator=calibrator,
        method=sklearn_method,
        raw_probabilities=raw_probabilities,
        calibrated_probabilities=calibrated_probabilities,
        n_calibration_rows=len(X_calibration),
    )


def predict_positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted classifier."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must expose predict_proba for probability calibration")

    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "predict_proba must return a 2D array with at least two class columns"
        )

    return np.asarray(probabilities[:, 1], dtype=float)


def _normalize_sklearn_calibration_method(method: str) -> str:
    """Normalize friendly calibration aliases to scikit-learn method names."""
    if method == "isotonic":
        return "isotonic"
    if method in {"sigmoid", "platt", "platt_scaling"}:
        return "sigmoid"

    raise ValueError(
        f"Unsupported calibration method '{method}'. "
        "Supported methods: isotonic, segmented_empirical, segmented_isotonic, sigmoid/platt, none"
    )


def _build_prefit_calibrator(model: Any, method: str) -> CalibratedClassifierCV:
    """Build a calibrator for an already-fitted base model.

    Newer scikit-learn versions require wrapping already-fitted estimators with
    FrozenEstimator instead of passing ``cv='prefit'``. Older versions do not
    have FrozenEstimator, so we fall back to the legacy API.
    """
    if FrozenEstimator is not None:
        return CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=method,
        )

    try:
        return CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")
