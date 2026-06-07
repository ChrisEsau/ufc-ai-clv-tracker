from __future__ import annotations

import numpy as np

from pipeline.modeling.algorithms.xgboost_predictor import (
    predict_binary_probability as xgb_predict_binary_probability,
)
from pipeline.modeling.algorithms.xgboost_predictor import (
    predict_class_probabilities as xgb_predict_class_probabilities,
)


class ProbabilityError(RuntimeError):
    """Raised when prediction probabilities cannot be generated."""


SUPPORTED_ALGORITHMS = {
    "xgboost",
}



def predict_binary_probability(
    model,
    X,
    algorithm: str,
) -> np.ndarray:
    """Dispatch binary probability prediction by algorithm."""

    algorithm = str(algorithm).lower().strip()

    if algorithm == "xgboost":
        probabilities = xgb_predict_binary_probability(model, X)
    else:
        raise ProbabilityError(
            f"Unsupported prediction algorithm: {algorithm}. "
            f"Supported algorithms: {sorted(SUPPORTED_ALGORITHMS)}"
        )

    _validate_probability_vector(probabilities)

    return probabilities



def predict_class_probabilities(
    model,
    X,
    algorithm: str,
) -> np.ndarray:
    """Dispatch multiclass probability prediction by algorithm."""

    algorithm = str(algorithm).lower().strip()

    if algorithm == "xgboost":
        probabilities = xgb_predict_class_probabilities(model, X)
    else:
        raise ProbabilityError(
            f"Unsupported prediction algorithm: {algorithm}. "
            f"Supported algorithms: {sorted(SUPPORTED_ALGORITHMS)}"
        )

    _validate_probability_matrix(probabilities)

    return probabilities



def _validate_probability_vector(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 1:
        raise ProbabilityError(
            f"Binary probability output must be 1D, received shape {probabilities.shape}."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ProbabilityError(
            "Binary probabilities must be between 0 and 1."
        )



def _validate_probability_matrix(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2:
        raise ProbabilityError(
            f"Class probability output must be 2D, received shape {probabilities.shape}."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ProbabilityError(
            "Class probabilities must be between 0 and 1."
        )
