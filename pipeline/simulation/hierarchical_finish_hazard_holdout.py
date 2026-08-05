"""Build counterfactual holdout hazards from the hierarchical finish model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from pipeline.simulation.finish_hazard_holdout import _counterfactual_holdout_rows
from pipeline.simulation.finish_hazard_model import (
    DEFAULT_TEST_YEARS,
    FINISH_CLASSES,
    prepare_finish_hazard_dataset,
)
from pipeline.simulation.hierarchical_finish_hazard_model import (
    CALIBRATED_STAGE_COLUMNS,
    HIERARCHICAL_FINISH_MODEL_VERSION,
    HIERARCHICAL_MODELS,
    PROBABILITY_COLUMNS,
    STAGE_PROBABILITY_COLUMNS,
    add_hierarchical_targets,
    apply_hierarchical_calibration_factors,
    calibration_factors_for_year,
    fit_predict_hierarchical_stages,
)


class HierarchicalFinishHoldoutError(RuntimeError):
    """Raised when hierarchical holdout hazards cannot be built."""


@dataclass(frozen=True)
class HierarchicalCounterfactualFinishResult:
    predictions: pd.DataFrame
    model_name: str
    model_seed: int
    calibration_source: str


def build_hierarchical_counterfactual_finish_predictions(
    training_df: pd.DataFrame,
    calibration_schedule: pd.DataFrame,
    test_year: int = 2026,
    model_name: str = HIERARCHICAL_MODELS[0],
    seed: int = 7,
    walk_forward_years: Sequence[int] = DEFAULT_TEST_YEARS,
) -> HierarchicalCounterfactualFinishResult:
    """Fit on pre-holdout fights and predict every scheduled holdout round."""
    if model_name not in HIERARCHICAL_MODELS:
        raise HierarchicalFinishHoldoutError(
            f"Unsupported hierarchical finish model: {model_name!r}"
        )

    prepared = add_hierarchical_targets(prepare_finish_hazard_dataset(training_df))
    train = prepared.loc[prepared["date"].dt.year.lt(int(test_year))].copy()
    if train.empty:
        raise HierarchicalFinishHoldoutError(
            f"No pre-{test_year} rows are available for hierarchical training"
        )
    counterfactual = _counterfactual_holdout_rows(prepared, test_year=test_year)

    include_rfs = model_name.endswith("_rfs")
    year_list = [int(year) for year in walk_forward_years]
    try:
        fold_index = year_list.index(int(test_year))
    except ValueError:
        fold_index = len([year for year in year_list if year < int(test_year)])
    model_seed = int(seed + fold_index * 101 + (17 if include_rfs else 0))
    stage_probabilities, _, _ = fit_predict_hierarchical_stages(
        train,
        counterfactual,
        include_rfs=include_rfs,
        seed=model_seed,
    )
    factors = calibration_factors_for_year(
        calibration_schedule,
        model_name=model_name,
        test_year=test_year,
    )
    calibrated_stages, calibrated_combined = apply_hierarchical_calibration_factors(
        stage_probabilities,
        factors,
    )

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
    output["model_version"] = HIERARCHICAL_FINISH_MODEL_VERSION
    output["test_year"] = int(test_year)
    output["calibration_source"] = "prior_walk_forward_hierarchy"

    for stage, column in STAGE_PROBABILITY_COLUMNS.items():
        output[column] = np.asarray(stage_probabilities[stage], dtype=float)
        output[CALIBRATED_STAGE_COLUMNS[stage]] = np.asarray(
            calibrated_stages[stage], dtype=float
        )

    raw_finish = np.column_stack(
        [
            1.0 - output["prob_finish"].to_numpy(dtype=float),
            output["prob_finish"].to_numpy(dtype=float)
            * (1.0 - output["prob_submission_given_finish"].to_numpy(dtype=float))
            * output["prob_red_given_ko_tko"].to_numpy(dtype=float),
            output["prob_finish"].to_numpy(dtype=float)
            * output["prob_submission_given_finish"].to_numpy(dtype=float)
            * output["prob_red_given_submission"].to_numpy(dtype=float),
            output["prob_finish"].to_numpy(dtype=float)
            * (1.0 - output["prob_submission_given_finish"].to_numpy(dtype=float))
            * (1.0 - output["prob_red_given_ko_tko"].to_numpy(dtype=float)),
            output["prob_finish"].to_numpy(dtype=float)
            * output["prob_submission_given_finish"].to_numpy(dtype=float)
            * (1.0 - output["prob_red_given_submission"].to_numpy(dtype=float)),
        ]
    )
    for index, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = raw_finish[:, index]
        output[f"calibrated_{column}"] = calibrated_combined[:, index]

    calibrated_columns = [
        f"calibrated_prob_{name}" for name in FINISH_CLASSES
    ]
    probabilities = output[calibrated_columns].to_numpy(dtype=float)
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-8
    ):
        raise HierarchicalFinishHoldoutError(
            "Hierarchical counterfactual probabilities violate the simplex"
        )
    if output.duplicated(["fight_id", "round"]).any():
        raise HierarchicalFinishHoldoutError(
            "Hierarchical counterfactual predictions contain duplicate keys"
        )

    return HierarchicalCounterfactualFinishResult(
        predictions=output.sort_values(["date", "fight_id", "round"]).reset_index(
            drop=True
        ),
        model_name=str(model_name),
        model_seed=model_seed,
        calibration_source="prior_walk_forward_hierarchy",
    )
