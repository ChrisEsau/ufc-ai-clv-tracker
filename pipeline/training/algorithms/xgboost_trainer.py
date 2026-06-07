"""XGBoost trainer plug-in for the generic UFC training framework.

This module is intentionally narrow:
- receives prepared X/y matrices
- receives XGBoost params
- returns a fitted model plus metadata

It does not handle feature selection, symmetry, temporal splitting, calibration,
metrics, or artifact saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

DEFAULT_XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}


@dataclass(frozen=True)
class TrainedModelResult:
    """Container for a fitted model and training metadata."""

    model: Any
    algorithm: str
    params: dict[str, Any]
    n_train_rows: int
    n_features: int
    feature_columns: list[str]


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> TrainedModelResult:
    """Train an XGBoost binary classifier using provided parameters."""
    final_params = DEFAULT_XGBOOST_PARAMS.copy()
    if params:
        final_params.update(params)

    model = XGBClassifier(**final_params)
    model.fit(X_train, y_train)

    return TrainedModelResult(
        model=model,
        algorithm="xgboost",
        params=final_params,
        n_train_rows=len(X_train),
        n_features=X_train.shape[1],
        feature_columns=list(X_train.columns),
    )
