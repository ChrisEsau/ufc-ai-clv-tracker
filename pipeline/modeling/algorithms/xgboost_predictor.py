from __future__ import annotations

import numpy as np


class XGBoostPredictorError(RuntimeError):
    """Raised when XGBoost prediction output is invalid."""



def predict_binary_probability(model, X) -> np.ndarray:
    """Return positive-class probabilities for binary classifiers."""

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2:
        raise XGBoostPredictorError(
            f"Expected 2D probability output, received shape {probabilities.shape}."
        )

    if probabilities.shape[1] < 2:
        raise XGBoostPredictorError(
            "Binary classifier probability output must contain two columns."
        )

    return probabilities[:, 1]



def predict_class_probabilities(model, X) -> np.ndarray:
    """Return class probability matrix for multiclass classifiers."""

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2:
        raise XGBoostPredictorError(
            f"Expected 2D probability output, received shape {probabilities.shape}."
        )

    return probabilities
