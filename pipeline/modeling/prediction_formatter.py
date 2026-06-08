from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from pipeline.modeling.confidence import score_prediction_confidence
from pipeline.modeling.model_config import get_algorithm, get_model_family, get_model_id, get_prediction_config


class PredictionFormatterError(RuntimeError):
    """Raised when probabilities cannot be formatted into outcome rows."""


REQUIRED_INPUT_COLUMNS = [
    "event_id",
    "event_name",
    "fight_id",
    "red_fighter",
    "blue_fighter",
]

OPTIONAL_INPUT_COLUMNS = [
    "commence_time",
    "red_fighter_id",
    "blue_fighter_id",
    "passes_model_data_quality",
    "passes_feature_validation",
    "nonzero_feature_count",
    "zero_feature_pct",
    "feature_count_expected",
    "feature_count_actual",
    "red_feature_match",
    "blue_feature_match",
    "feature_match_type",
]


def format_prediction_outcomes(
    *,
    fight_df: pd.DataFrame,
    probabilities: np.ndarray,
    model_config: dict[str, Any],
    prediction_run_id: str,
    prediction_timestamp: str | None = None,
) -> pd.DataFrame:
    """Format model probabilities into canonical outcome-level prediction rows.

    Parameters
    ----------
    fight_df:
        Fight-level dataframe containing fighter/event metadata.
    probabilities:
        Binary vector or multiclass matrix returned by the probability layer.
    model_config:
        Parsed model YAML containing a V2 ``prediction`` section.
    prediction_run_id:
        Unique run identifier created by the runner/adapter.
    prediction_timestamp:
        Optional UTC timestamp. If omitted, generated here.
    """

    prediction_config = get_prediction_config(model_config)
    formatter_type = str(prediction_config.get("format", "")).strip().lower()

    if not formatter_type:
        raise PredictionFormatterError(
            "Model config prediction.format is required for outcome formatting."
        )

    timestamp = prediction_timestamp or datetime.now(timezone.utc).isoformat()

    if formatter_type == "binary_matchup":
        return _format_binary_matchup(
            fight_df=fight_df,
            probabilities=np.asarray(probabilities, dtype=float),
            model_config=model_config,
            prediction_config=prediction_config,
            prediction_run_id=prediction_run_id,
            prediction_timestamp=timestamp,
        )

    if formatter_type == "binary_prop":
        return _format_binary_prop(
            fight_df=fight_df,
            probabilities=np.asarray(probabilities, dtype=float),
            model_config=model_config,
            prediction_config=prediction_config,
            prediction_run_id=prediction_run_id,
            prediction_timestamp=timestamp,
        )

    if formatter_type == "multiclass":
        return _format_multiclass(
            fight_df=fight_df,
            probabilities=np.asarray(probabilities, dtype=float),
            model_config=model_config,
            prediction_config=prediction_config,
            prediction_run_id=prediction_run_id,
            prediction_timestamp=timestamp,
        )

    raise PredictionFormatterError(
        f"Unsupported prediction formatter type: {formatter_type}"
    )


def _format_binary_matchup(
    *,
    fight_df: pd.DataFrame,
    probabilities: np.ndarray,
    model_config: dict[str, Any],
    prediction_config: dict[str, Any],
    prediction_run_id: str,
    prediction_timestamp: str,
) -> pd.DataFrame:
    """Format binary matchup probabilities into two outcome rows per fight."""

    _validate_fight_df(fight_df)
    _validate_binary_probabilities(probabilities, len(fight_df))

    outcomes_config = prediction_config.get("outcomes", {})
    positive_config = outcomes_config.get("positive", {})
    negative_config = outcomes_config.get("negative", {})

    positive_label_source = positive_config.get("label_source")
    negative_label_source = negative_config.get("label_source")

    if not positive_label_source or not negative_label_source:
        raise PredictionFormatterError(
            "binary_matchup requires prediction.outcomes positive/negative label_source."
        )

    rows: list[dict[str, Any]] = []

    for i, (_, fight_row) in enumerate(fight_df.reset_index(drop=True).iterrows()):
        positive_probability = float(probabilities[i])
        negative_probability = float(1.0 - positive_probability)

        positive_label = _resolve_label(fight_row, label_source=positive_label_source)
        negative_label = _resolve_label(fight_row, label_source=negative_label_source)

        if positive_probability >= negative_probability:
            model_pick = positive_label
            model_confidence = positive_probability
        else:
            model_pick = negative_label
            model_confidence = negative_probability

        rows.append(
            _build_base_outcome_row(
                fight_row=fight_row,
                model_config=model_config,
                prediction_config=prediction_config,
                prediction_run_id=prediction_run_id,
                prediction_timestamp=prediction_timestamp,
                outcome_label=positive_label,
                outcome_side=str(positive_config.get("outcome_side", "positive")),
                outcome_fighter_id=_resolve_matchup_outcome_fighter_id(
                    fight_row,
                    outcome_side=str(positive_config.get("outcome_side", "positive")),
                ),
                model_probability=positive_probability,
                is_model_pick=positive_label == model_pick,
                model_pick=model_pick,
                model_confidence=model_confidence,
            )
        )
        rows.append(
            _build_base_outcome_row(
                fight_row=fight_row,
                model_config=model_config,
                prediction_config=prediction_config,
                prediction_run_id=prediction_run_id,
                prediction_timestamp=prediction_timestamp,
                outcome_label=negative_label,
                outcome_side=str(negative_config.get("outcome_side", "negative")),
                outcome_fighter_id=_resolve_matchup_outcome_fighter_id(
                    fight_row,
                    outcome_side=str(negative_config.get("outcome_side", "negative")),
                ),
                model_probability=negative_probability,
                is_model_pick=negative_label == model_pick,
                model_pick=model_pick,
                model_confidence=model_confidence,
            )
        )

    return pd.DataFrame(rows)


def _format_binary_prop(
    *,
    fight_df: pd.DataFrame,
    probabilities: np.ndarray,
    model_config: dict[str, Any],
    prediction_config: dict[str, Any],
    prediction_run_id: str,
    prediction_timestamp: str,
) -> pd.DataFrame:
    """Format binary prop probabilities into positive/negative outcome rows."""

    _validate_fight_df(fight_df)
    _validate_binary_probabilities(probabilities, len(fight_df))

    outcomes_config = prediction_config.get("outcomes", {})
    positive_config = outcomes_config.get("positive", {})
    negative_config = outcomes_config.get("negative", {})

    positive_label = positive_config.get("label")
    negative_label = negative_config.get("label")

    if not positive_label or not negative_label:
        raise PredictionFormatterError(
            "binary_prop requires prediction.outcomes positive/negative label."
        )

    rows: list[dict[str, Any]] = []

    for i, (_, fight_row) in enumerate(fight_df.reset_index(drop=True).iterrows()):
        positive_probability = float(probabilities[i])
        negative_probability = float(1.0 - positive_probability)

        if positive_probability >= negative_probability:
            model_pick = str(positive_label)
            model_confidence = positive_probability
        else:
            model_pick = str(negative_label)
            model_confidence = negative_probability

        rows.append(
            _build_base_outcome_row(
                fight_row=fight_row,
                model_config=model_config,
                prediction_config=prediction_config,
                prediction_run_id=prediction_run_id,
                prediction_timestamp=prediction_timestamp,
                outcome_label=str(positive_label),
                outcome_side=str(positive_config.get("outcome_side", "positive")),
                outcome_fighter_id=positive_config.get("outcome_fighter_id"),
                model_probability=positive_probability,
                is_model_pick=str(positive_label) == model_pick,
                model_pick=model_pick,
                model_confidence=model_confidence,
            )
        )
        rows.append(
            _build_base_outcome_row(
                fight_row=fight_row,
                model_config=model_config,
                prediction_config=prediction_config,
                prediction_run_id=prediction_run_id,
                prediction_timestamp=prediction_timestamp,
                outcome_label=str(negative_label),
                outcome_side=str(negative_config.get("outcome_side", "negative")),
                outcome_fighter_id=negative_config.get("outcome_fighter_id"),
                model_probability=negative_probability,
                is_model_pick=str(negative_label) == model_pick,
                model_pick=model_pick,
                model_confidence=model_confidence,
            )
        )

    return pd.DataFrame(rows)


def _format_multiclass(
    *,
    fight_df: pd.DataFrame,
    probabilities: np.ndarray,
    model_config: dict[str, Any],
    prediction_config: dict[str, Any],
    prediction_run_id: str,
    prediction_timestamp: str,
) -> pd.DataFrame:
    """Format multiclass probabilities into one row per configured class."""

    _validate_fight_df(fight_df)

    if probabilities.ndim != 2:
        raise PredictionFormatterError(
            f"multiclass formatter expected 2D probabilities, received shape {probabilities.shape}."
        )

    if probabilities.shape[0] != len(fight_df):
        raise PredictionFormatterError(
            "Probability row count does not match fight dataframe row count: "
            f"{probabilities.shape[0]} != {len(fight_df)}"
        )

    class_labels = prediction_config.get("class_labels") or prediction_config.get("classes")
    if not isinstance(class_labels, list) or not class_labels:
        raise PredictionFormatterError(
            "multiclass formatter requires prediction.class_labels or prediction.classes."
        )

    class_labels = [str(label) for label in class_labels]

    if probabilities.shape[1] != len(class_labels):
        raise PredictionFormatterError(
            "Class probability column count does not match class label count: "
            f"{probabilities.shape[1]} != {len(class_labels)}"
        )

    rows: list[dict[str, Any]] = []

    for i, (_, fight_row) in enumerate(fight_df.reset_index(drop=True).iterrows()):
        row_probabilities = probabilities[i]
        pick_index = int(np.argmax(row_probabilities))
        model_pick = class_labels[pick_index]
        model_confidence = float(row_probabilities[pick_index])

        for class_index, class_label in enumerate(class_labels):
            probability = float(row_probabilities[class_index])
            rows.append(
                _build_base_outcome_row(
                    fight_row=fight_row,
                    model_config=model_config,
                    prediction_config=prediction_config,
                    prediction_run_id=prediction_run_id,
                    prediction_timestamp=prediction_timestamp,
                    outcome_label=class_label,
                    outcome_side=class_label,
                    outcome_fighter_id=None,
                    model_probability=probability,
                    is_model_pick=class_label == model_pick,
                    model_pick=model_pick,
                    model_confidence=model_confidence,
                )
            )

    return pd.DataFrame(rows)


def _build_base_outcome_row(
    *,
    fight_row: pd.Series,
    model_config: dict[str, Any],
    prediction_config: dict[str, Any],
    prediction_run_id: str,
    prediction_timestamp: str,
    outcome_label: str,
    outcome_side: str,
    outcome_fighter_id: Any,
    model_probability: float,
    is_model_pick: bool,
    model_pick: str,
    model_confidence: float,
) -> dict[str, Any]:
    """Build one canonical prediction outcome row."""

    confidence_payload = score_prediction_confidence(
        fight_row,
        model_pick_probability=model_confidence,
    ).to_dict()

    row = {
        "prediction_run_id": prediction_run_id,
        "prediction_timestamp": prediction_timestamp,
        "model_id": get_model_id(model_config),
        "model_family": get_model_family(model_config),
        "algorithm": get_algorithm(model_config),
        "prediction_type": prediction_config.get("prediction_type", prediction_config.get("format")),
        "event_id": fight_row.get("event_id"),
        "event_name": fight_row.get("event_name"),
        "commence_time": fight_row.get("commence_time"),
        "fight_id": fight_row.get("fight_id"),
        "red_fighter": fight_row.get("red_fighter"),
        "blue_fighter": fight_row.get("blue_fighter"),
        "red_fighter_id": fight_row.get("red_fighter_id"),
        "blue_fighter_id": fight_row.get("blue_fighter_id"),
        "market_key": prediction_config.get("market_key"),
        "outcome_label": str(outcome_label),
        "outcome_fighter_id": outcome_fighter_id,
        "outcome_side": str(outcome_side),
        "model_probability": float(model_probability),
        "model_pick_probability": float(model_confidence),
        "is_model_pick": bool(is_model_pick),
        "model_pick": str(model_pick),
        # Backward-compatible alias. True trust confidence is stored in confidence_score/confidence_pct.
        "model_confidence": float(model_confidence),
        "passes_model_data_quality": bool(fight_row.get("passes_model_data_quality", True)),
        "passes_feature_validation": bool(fight_row.get("passes_feature_validation", True)),
        "nonzero_feature_count": fight_row.get("nonzero_feature_count"),
        "zero_feature_pct": fight_row.get("zero_feature_pct"),
        "feature_count_expected": fight_row.get("feature_count_expected"),
        "feature_count_actual": fight_row.get("feature_count_actual"),
        "red_feature_match": fight_row.get("red_feature_match"),
        "blue_feature_match": fight_row.get("blue_feature_match"),
        "feature_match_type": fight_row.get("feature_match_type"),
    }
    row.update(confidence_payload)

    return row


def _resolve_matchup_outcome_fighter_id(fight_row: pd.Series, *, outcome_side: str):
    """Resolve moneyline outcome fighter ID from canonical matchup side."""

    normalized_side = str(outcome_side).strip().lower()

    if normalized_side in {"positive", "red", "r"}:
        return fight_row.get("red_fighter_id")

    if normalized_side in {"negative", "blue", "b"}:
        return fight_row.get("blue_fighter_id")

    return None


def _validate_fight_df(fight_df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in fight_df.columns]
    if missing_columns:
        raise PredictionFormatterError(
            f"Fight dataframe missing required formatter columns: {missing_columns}"
        )


def _validate_binary_probabilities(probabilities: np.ndarray, expected_rows: int) -> None:
    if probabilities.ndim != 1:
        raise PredictionFormatterError(
            f"Binary formatter expected 1D probabilities, received shape {probabilities.shape}."
        )

    if len(probabilities) != expected_rows:
        raise PredictionFormatterError(
            "Probability count does not match fight dataframe row count: "
            f"{len(probabilities)} != {expected_rows}"
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise PredictionFormatterError("Probabilities must be between 0 and 1.")


def _resolve_label(fight_row: pd.Series, *, label_source: str) -> str:
    if label_source not in fight_row.index:
        raise PredictionFormatterError(
            f"Configured label_source '{label_source}' is missing from fight dataframe."
        )

    value = fight_row.get(label_source)
    if pd.isna(value) or str(value).strip() == "":
        raise PredictionFormatterError(
            f"Configured label_source '{label_source}' resolved to an empty value."
        )

    return str(value)
