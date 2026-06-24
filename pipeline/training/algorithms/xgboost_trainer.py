"""XGBoost trainer plug-in for the generic UFC training framework.

This module is intentionally narrow:
- receives prepared X/y matrices
- receives XGBoost params
- returns a fitted model plus metadata

It does not handle feature selection, symmetry augmentation, temporal splitting,
calibration, metrics, or artifact saving.
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
    early_stopping_enabled: bool = False
    early_stopping_rounds: int | None = None
    early_stopping_metric: str | None = None
    best_iteration: int | None = None
    best_score: float | None = None


def _clean_xgboost_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove non-XGBClassifier constructor keys from model params."""

    blocked = {
        "early_stopping_rounds",
        "callbacks",
        "eval_set",
        "verbose",
    }
    return {key: value for key, value in params.items() if key not in blocked}


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration", None)
    if value is None:
        value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None else None


def _best_score(model: Any) -> float | None:
    value = getattr(model, "best_score", None)
    if value is None:
        value = getattr(model, "best_score_", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
    X_validation: pd.DataFrame | None = None,
    y_validation: pd.Series | None = None,
    early_stopping_config: dict[str, Any] | None = None,
) -> TrainedModelResult:
    """Train an XGBoost binary classifier using provided parameters."""
    final_params = DEFAULT_XGBOOST_PARAMS.copy()
    if params:
        final_params.update(params)

    early_config = early_stopping_config or {}
    early_enabled = bool(early_config.get("enabled", False))
    early_rounds = int(early_config.get("rounds", 0) or 0)
    early_metric = str(early_config.get("metric") or final_params.get("eval_metric") or "logloss")
    has_validation = X_validation is not None and y_validation is not None and len(X_validation) > 0

    fit_kwargs: dict[str, Any] = {}
    if early_enabled:
        if not has_validation:
            raise ValueError("Early stopping is enabled but no validation data was provided.")
        if early_rounds <= 0:
            raise ValueError("early_stopping.rounds must be positive when early stopping is enabled.")
        final_params["eval_metric"] = early_metric
        final_params["early_stopping_rounds"] = early_rounds
        fit_kwargs["eval_set"] = [(X_validation, y_validation)]
        fit_kwargs["verbose"] = bool(early_config.get("verbose", False))

    model = XGBClassifier(**_clean_xgboost_params(final_params))
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        # Compatibility path for XGBoost versions that do not accept
        # early_stopping_rounds in the constructor.
        if early_enabled:
            model = XGBClassifier(**_clean_xgboost_params({k: v for k, v in final_params.items() if k != "early_stopping_rounds"}))
            model.fit(X_train, y_train, early_stopping_rounds=early_rounds, **fit_kwargs)
        else:
            raise

    return TrainedModelResult(
        model=model,
        algorithm="xgboost",
        params=final_params,
        n_train_rows=len(X_train),
        n_features=X_train.shape[1],
        feature_columns=list(X_train.columns),
        early_stopping_enabled=early_enabled,
        early_stopping_rounds=early_rounds if early_enabled else None,
        early_stopping_metric=early_metric if early_enabled else None,
        best_iteration=_best_iteration(model),
        best_score=_best_score(model),
    )
