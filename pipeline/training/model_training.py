"""Generic model training dispatcher.

This module keeps the public training API algorithm-agnostic. The caller passes
an algorithm name, prepared train matrices, and parameters. The dispatcher routes
the request to the appropriate algorithm plug-in.

Feature selection, symmetry augmentation, temporal splitting, calibration,
metrics, and artifact saving are handled by other modules.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.training.algorithms.xgboost_trainer import (
    TrainedModelResult,
    train_xgboost_classifier,
)

SUPPORTED_ALGORITHMS = {"xgboost"}


def train_model(
    algorithm: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> TrainedModelResult:
    """Train a model using an algorithm-specific plug-in."""
    normalized_algorithm = algorithm.strip().lower()

    if normalized_algorithm == "xgboost":
        return train_xgboost_classifier(
            X_train=X_train,
            y_train=y_train,
            params=params,
        )

    raise ValueError(
        f"Unsupported algorithm '{algorithm}'. "
        f"Supported algorithms: {sorted(SUPPORTED_ALGORITHMS)}"
    )
