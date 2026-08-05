"""Build counterfactual scheduled-round finish hazards for a historical holdout.

Observed training data ends when a fight ends. A simulator may counterfactually
continue past the actual terminal round, so a historical finish provider needs a
probability row for every scheduled round, not only the rounds that were observed.

The finish model is deliberately pre-fight-only except for round number. This
module therefore takes one pre-fight feature row per holdout fight, duplicates it
across all scheduled rounds, and predicts every counterfactual round. Model fitting
uses only fights before the holdout year, while probability calibration uses the
sequential factors already estimated from earlier walk-forward holdouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from pipeline.simulation.finish_hazard_model import (
    DEFAULT_TEST_YEARS,
    FINISH_CLASSES,
    PROBABILITY_COLUMNS,
    _feature_columns,
    _fit_matrix_spec,
    _new_model,
    _transform_matrix,
    prepare_finish_hazard_dataset,
)


class FinishHazardHoldoutError(RuntimeError):
    """Raised when counterfactual holdout hazards cannot be constructed."""


@dataclass(frozen=True)
class CounterfactualFinishHazardResult:
    predictions: pd.DataFrame
    model_name: str
    model_seed: int
    calibration_source: str


def _calibration_factors(
    calibration_schedule: pd.DataFrame,
    model_name: str,
    test_year: int,
) -> tuple[np.ndarray, str]:
    required = [
        "model_name",
        "test_year",
        "calibration_source",
        *[f"factor_{name}" for name in FINISH_CLASSES],
    ]
    missing = [column for column in required if column not in calibration_schedule]
    if missing:
        raise FinishHazardHoldoutError(
            f"Calibration schedule is missing columns: {missing}"
        )
    schedule = calibration_schedule.copy()
    schedule["test_year"] = pd.to_numeric(schedule["test_year"], errors="coerce")
    match = schedule.loc[
        schedule["model_name"].astype(str).eq(str(model_name))
        & schedule["test_year"].eq(int(test_year))
    ]
    if len(match) != 1:
        raise FinishHazardHoldoutError(
            f"Expected one calibration row for {model_name}/{test_year}; found {len(match)}"
        )
    row = match.iloc[0]
    factors = np.asarray(
        [float(row[f"factor_{name}"]) for name in FINISH_CLASSES], dtype=float
    )
    if not np.isfinite(factors).all() or np.any(factors <= 0.0):
        raise FinishHazardHoldoutError(
            "Calibration factors must be finite and positive"
        )
    return factors, str(row["calibration_source"])


def _counterfactual_holdout_rows(
    prepared: pd.DataFrame,
    test_year: int,
) -> pd.DataFrame:
    holdout = prepared.loc[
        prepared["date"].dt.year.eq(int(test_year))
    ].copy()
    if holdout.empty:
        raise FinishHazardHoldoutError(
            f"No prepared finish rows exist for holdout year {test_year}"
        )

    # Every eligible fight has an observed first round. All selected predictors
    # except round are pre-fight values and may be reused counterfactually.
    first_round = (
        holdout.sort_values(["fight_id", "round"])
        .groupby("fight_id", as_index=False, sort=False)
        .first()
    )
    rows: list[pd.Series] = []
    for _, row in first_round.iterrows():
        scheduled_rounds = int(row["total_rounds"])
        if scheduled_rounds not in (3, 5):
            raise FinishHazardHoldoutError(
                f"Unsupported scheduled rounds for {row['fight_id']}: {scheduled_rounds}"
            )
        for round_number in range(1, scheduled_rounds + 1):
            copy = row.copy()
            copy["round"] = int(round_number)
            rows.append(copy)
    counterfactual = pd.DataFrame(rows).reset_index(drop=True)
    if counterfactual.duplicated(["fight_id", "round"]).any():
        raise FinishHazardHoldoutError(
            "Counterfactual finish rows contain duplicate fight-round keys"
        )

    expected = first_round["total_rounds"].astype(int).sum()
    if len(counterfactual) != int(expected):
        raise FinishHazardHoldoutError(
            "Counterfactual scheduled-round coverage is incomplete"
        )
    return counterfactual


def build_counterfactual_finish_predictions(
    training_df: pd.DataFrame,
    calibration_schedule: pd.DataFrame,
    test_year: int = 2026,
    model_name: str = "xgb_prefight_context",
    seed: int = 7,
    walk_forward_years: Sequence[int] = DEFAULT_TEST_YEARS,
) -> CounterfactualFinishHazardResult:
    """Fit the pre-holdout model and predict every scheduled holdout round."""
    if model_name not in {"xgb_prefight_context", "xgb_prefight_context_rfs"}:
        raise FinishHazardHoldoutError(
            f"Unsupported finish hazard model: {model_name!r}"
        )

    prepared = prepare_finish_hazard_dataset(training_df)
    train = prepared.loc[
        prepared["date"].dt.year.lt(int(test_year))
    ].copy()
    if train.empty:
        raise FinishHazardHoldoutError(
            f"No pre-{test_year} finish rows are available for training"
        )
    counterfactual = _counterfactual_holdout_rows(prepared, test_year=test_year)

    include_rfs = model_name.endswith("_rfs")
    numeric, categorical = _feature_columns(train, include_rfs=include_rfs)
    spec = _fit_matrix_spec(train, numeric, categorical)
    x_train = _transform_matrix(train, spec)
    x_holdout = _transform_matrix(counterfactual, spec)

    year_list = [int(year) for year in walk_forward_years]
    try:
        fold_index = year_list.index(int(test_year))
    except ValueError:
        fold_index = len([year for year in year_list if year < int(test_year)])
    model_seed = int(seed + fold_index * 101 + (17 if include_rfs else 0))
    model = _new_model(model_seed)
    model.fit(x_train, train["finish_class_index"].to_numpy(dtype=int))
    raw = np.asarray(model.predict_proba(x_holdout), dtype=float)
    if raw.shape != (len(counterfactual), len(FINISH_CLASSES)):
        raise FinishHazardHoldoutError(
            f"Unexpected finish probability shape: {raw.shape}"
        )

    factors, calibration_source = _calibration_factors(
        calibration_schedule,
        model_name=model_name,
        test_year=test_year,
    )
    calibrated = raw * factors.reshape(1, -1)
    calibrated /= calibrated.sum(axis=1, keepdims=True)

    identity_columns = [
        column
        for column in (
            "event_id",
            "event_name",
            "fight_id",
            "date",
            "round",
            "total_rounds",
            "division",
            "title_fight",
            "red_fighter_id",
            "blue_fighter_id",
        )
        if column in counterfactual.columns
    ]
    output = counterfactual[identity_columns].copy()
    output["model_name"] = str(model_name)
    output["model_version"] = "finish_hazard_prefight_v0"
    output["test_year"] = int(test_year)
    output["calibration_source"] = calibration_source
    for index, name in enumerate(FINISH_CLASSES):
        output[f"prob_{name}"] = raw[:, index]
        output[f"calibrated_prob_{name}"] = calibrated[:, index]

    probability_columns = [
        f"calibrated_prob_{name}" for name in FINISH_CLASSES
    ]
    if not np.allclose(
        output[probability_columns].sum(axis=1).to_numpy(dtype=float),
        1.0,
        atol=1e-6,
    ):
        raise FinishHazardHoldoutError(
            "Calibrated counterfactual finish probabilities do not sum to one"
        )
    return CounterfactualFinishHazardResult(
        predictions=output.sort_values(["date", "fight_id", "round"]).reset_index(
            drop=True
        ),
        model_name=str(model_name),
        model_seed=model_seed,
        calibration_source=calibration_source,
    )
