from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.modeling.model_config import get_prediction_config
from pipeline.modeling.model_loader import ModelBundle
from pipeline.modeling.prediction_formatter import format_prediction_outcomes
from pipeline.modeling.probability import predict_binary_probability, predict_class_probabilities


class PredictionAdapterError(RuntimeError):
    """Raised when model prediction adaptation fails."""


@dataclass(frozen=True)
class PredictionAdapterResult:
    """Container for adapter outputs."""

    outcome_df: pd.DataFrame
    feature_matrix: pd.DataFrame
    probabilities: np.ndarray



def run_prediction_adapter(
    *,
    model_bundle: ModelBundle,
    model_config: dict[str, Any],
    live_feature_df: pd.DataFrame,
    prediction_run_id: str,
    prediction_timestamp: str | None = None,
) -> PredictionAdapterResult:
    """Run a loaded model against live features and return outcome rows.

    The adapter is intentionally generic. It does not know whether the model is
    moneyline, goes-distance, method, or another market. Market-specific outcome
    rows are created by the config-driven formatter.
    """

    prediction_config = get_prediction_config(model_config)
    formatter_type = str(prediction_config.get("format", "")).strip().lower()

    if not formatter_type:
        raise PredictionAdapterError(
            "Model config prediction.format is required before running the adapter."
        )

    X = build_feature_matrix(
        live_feature_df=live_feature_df,
        feature_columns=model_bundle.feature_columns,
    )

    probabilities = _predict_probabilities(
        model_bundle=model_bundle,
        feature_matrix=X,
        formatter_type=formatter_type,
    )

    probabilities = _clip_probabilities(
        probabilities=probabilities,
        prediction_config=prediction_config,
    )

    outcome_df = format_prediction_outcomes(
        fight_df=live_feature_df,
        probabilities=probabilities,
        model_config=model_config,
        prediction_run_id=prediction_run_id,
        prediction_timestamp=prediction_timestamp,
    )

    return PredictionAdapterResult(
        outcome_df=outcome_df,
        feature_matrix=X,
        probabilities=probabilities,
    )



def build_feature_matrix(
    *,
    live_feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Align live features to the model feature contract."""

    if not feature_columns:
        raise PredictionAdapterError("Model bundle contains zero feature columns.")

    missing_columns = [column for column in feature_columns if column not in live_feature_df.columns]

    if missing_columns:
        raise PredictionAdapterError(
            "Live feature dataframe is missing model feature columns: "
            f"{missing_columns}"
        )

    return live_feature_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)



def _predict_probabilities(
    *,
    model_bundle: ModelBundle,
    feature_matrix: pd.DataFrame,
    formatter_type: str,
) -> np.ndarray:
    """Call the correct probability dispatcher for the formatter type."""

    if formatter_type in {"binary_matchup", "binary_prop"}:
        return predict_binary_probability(
            model=model_bundle.model,
            X=feature_matrix,
            algorithm=model_bundle.algorithm,
        )

    if formatter_type == "multiclass":
        return predict_class_probabilities(
            model=model_bundle.model,
            X=feature_matrix,
            algorithm=model_bundle.algorithm,
        )

    raise PredictionAdapterError(
        f"Unsupported formatter type for prediction adapter: {formatter_type}"
    )



def _clip_probabilities(
    *,
    probabilities: np.ndarray,
    prediction_config: dict[str, Any],
) -> np.ndarray:
    """Apply optional probability clipping from model config."""

    probability_config = prediction_config.get("probability", {}) or {}

    clip_low = probability_config.get("clip_low")
    clip_high = probability_config.get("clip_high")

    if clip_low is None and clip_high is None:
        return probabilities

    low = 0.0 if clip_low is None else float(clip_low)
    high = 1.0 if clip_high is None else float(clip_high)

    if low < 0 or high > 1 or low >= high:
        raise PredictionAdapterError(
            f"Invalid probability clipping bounds: clip_low={low}, clip_high={high}"
        )

    return np.clip(probabilities, low, high)
