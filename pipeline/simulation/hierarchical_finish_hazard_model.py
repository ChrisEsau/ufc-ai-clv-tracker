"""Leakage-safe hierarchical finish-hazard benchmark.

The existing five-class finish model asks one classifier to learn four separate
questions at once: whether the fight ends, whether the finish is a submission,
and which corner wins conditional on KO/TKO or submission. Historical audit
results showed useful conditional submission-side signal but weak submission
method discrimination. This shadow benchmark separates those decisions into:

1. finish versus no finish;
2. submission versus KO/TKO, conditional on a finish;
3. red versus blue, conditional on KO/TKO;
4. red versus blue, conditional on submission.

The four probabilities are recombined into the existing five-class provider
contract. Every feature remains pre-fight except round number. Walk-forward
calibration uses completed prior holdout years only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from pipeline.simulation.finish_hazard_model import (
    DEFAULT_TEST_YEARS,
    FINISH_CLASSES,
    PROBABILITY_COLUMNS,
    _feature_columns,
    _fit_matrix_spec,
    _metrics,
    _transform_matrix,
    prepare_finish_hazard_dataset,
)


HIERARCHICAL_FINISH_MODEL_VERSION = "finish_hazard_hierarchical_prefight_v0"
HIERARCHICAL_MODELS = (
    "hierarchical_xgb_prefight_context",
    "hierarchical_xgb_prefight_context_rfs",
)
STAGES = (
    "finish",
    "submission_given_finish",
    "red_given_ko_tko",
    "red_given_submission",
)
STAGE_PROBABILITY_COLUMNS = {
    "finish": "prob_finish",
    "submission_given_finish": "prob_submission_given_finish",
    "red_given_ko_tko": "prob_red_given_ko_tko",
    "red_given_submission": "prob_red_given_submission",
}
CALIBRATED_STAGE_COLUMNS = {
    stage: f"calibrated_{column}"
    for stage, column in STAGE_PROBABILITY_COLUMNS.items()
}


class HierarchicalFinishHazardModelError(RuntimeError):
    """Raised when hierarchical model data or probabilities are invalid."""


@dataclass(frozen=True)
class HierarchicalFinishBenchmarkResult:
    raw_predictions: pd.DataFrame
    calibrated_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    stage_metrics: pd.DataFrame
    calibration_schedule: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_manifest: pd.DataFrame


def _new_binary_model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=320,
        learning_rate=0.035,
        max_depth=3,
        min_child_weight=20.0,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.10,
        reg_lambda=10.0,
        tree_method="hist",
        random_state=int(seed),
        n_jobs=2,
    )


def add_hierarchical_targets(dataset: pd.DataFrame) -> pd.DataFrame:
    """Attach the four binary stage targets to a prepared finish table."""
    required = {"finish_class", "finish_class_index"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise HierarchicalFinishHazardModelError(
            f"Prepared finish table is missing target columns: {missing}"
        )
    out = dataset.copy()
    finish_class = out["finish_class"].astype(str)
    out["target_finish"] = finish_class.ne("no_finish").astype(int)
    out["target_submission_given_finish"] = finish_class.isin(
        ["red_submission", "blue_submission"]
    ).astype(int)
    out["target_red_given_ko_tko"] = finish_class.eq("red_ko_tko").astype(int)
    out["target_red_given_submission"] = finish_class.eq(
        "red_submission"
    ).astype(int)
    return out


def _stage_mask(frame: pd.DataFrame, stage: str) -> pd.Series:
    finish_class = frame["finish_class"].astype(str)
    if stage == "finish":
        return pd.Series(True, index=frame.index)
    if stage == "submission_given_finish":
        return finish_class.ne("no_finish")
    if stage == "red_given_ko_tko":
        return finish_class.isin(["red_ko_tko", "blue_ko_tko"])
    if stage == "red_given_submission":
        return finish_class.isin(["red_submission", "blue_submission"])
    raise HierarchicalFinishHazardModelError(f"Unknown hierarchy stage: {stage}")


def _stage_target_column(stage: str) -> str:
    return {
        "finish": "target_finish",
        "submission_given_finish": "target_submission_given_finish",
        "red_given_ko_tko": "target_red_given_ko_tko",
        "red_given_submission": "target_red_given_submission",
    }[stage]


def _fit_binary_stage(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Fit one binary stage, with a smoothed constant fallback."""
    target = np.asarray(y_train, dtype=int)
    if len(target) == 0:
        raise HierarchicalFinishHazardModelError(
            "A hierarchical stage received no training rows"
        )
    unique = np.unique(target)
    if len(unique) < 2:
        probability = float((target.sum() + 1.0) / (len(target) + 2.0))
        predictions = np.full(len(x_test), probability, dtype=float)
        importance = np.zeros(x_train.shape[1], dtype=float)
        return predictions, importance, "smoothed_constant_fallback"

    model = _new_binary_model(seed)
    model.fit(x_train, target)
    probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
    if probabilities.shape != (len(x_test), 2):
        raise HierarchicalFinishHazardModelError(
            f"Unexpected binary probability shape: {probabilities.shape}"
        )
    return probabilities[:, 1], np.asarray(model.feature_importances_), "xgboost"


def combine_hierarchical_probabilities(
    finish_probability: np.ndarray,
    submission_given_finish: np.ndarray,
    red_given_ko_tko: np.ndarray,
    red_given_submission: np.ndarray,
) -> np.ndarray:
    """Recombine four conditional probabilities into five competing risks."""
    finish = np.clip(np.asarray(finish_probability, dtype=float), 1e-9, 1.0 - 1e-9)
    submission = np.clip(
        np.asarray(submission_given_finish, dtype=float), 1e-9, 1.0 - 1e-9
    )
    red_ko = np.clip(np.asarray(red_given_ko_tko, dtype=float), 1e-9, 1.0 - 1e-9)
    red_sub = np.clip(
        np.asarray(red_given_submission, dtype=float), 1e-9, 1.0 - 1e-9
    )
    lengths = {len(finish), len(submission), len(red_ko), len(red_sub)}
    if len(lengths) != 1:
        raise HierarchicalFinishHazardModelError(
            "Hierarchical stage probability arrays must have equal length"
        )

    ko_probability = finish * (1.0 - submission)
    submission_probability = finish * submission
    combined = np.column_stack(
        [
            1.0 - finish,
            ko_probability * red_ko,
            submission_probability * red_sub,
            ko_probability * (1.0 - red_ko),
            submission_probability * (1.0 - red_sub),
        ]
    )
    if np.any(combined < 0.0) or not np.allclose(
        combined.sum(axis=1), 1.0, atol=1e-8
    ):
        raise HierarchicalFinishHazardModelError(
            "Recombined hierarchical probabilities violate the simplex"
        )
    return combined


def fit_predict_hierarchical_stages(
    train: pd.DataFrame,
    test: pd.DataFrame,
    include_rfs: bool,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, tuple[str, ...]]:
    """Fit all four stages on their conditional training populations."""
    train_targets = add_hierarchical_targets(train)
    numeric, categorical = _feature_columns(train_targets, include_rfs=include_rfs)
    spec = _fit_matrix_spec(train_targets, numeric, categorical)
    x_train_all = _transform_matrix(train_targets, spec)
    x_test = _transform_matrix(test, spec)

    probabilities: dict[str, np.ndarray] = {}
    importance_rows: list[dict[str, object]] = []
    for stage_index, stage in enumerate(STAGES):
        mask = _stage_mask(train_targets, stage)
        stage_x = x_train_all.loc[mask]
        stage_y = train_targets.loc[mask, _stage_target_column(stage)].to_numpy(
            dtype=int
        )
        predicted, importance, estimator = _fit_binary_stage(
            stage_x,
            stage_y,
            x_test,
            seed=int(seed + stage_index * 37),
        )
        probabilities[stage] = predicted
        for feature, value in zip(spec.feature_columns, importance):
            importance_rows.append(
                {
                    "stage": stage,
                    "feature": feature,
                    "importance": float(value),
                    "training_rows": int(len(stage_y)),
                    "positive_rate": float(stage_y.mean()),
                    "estimator": estimator,
                }
            )
    return probabilities, pd.DataFrame(importance_rows), spec.feature_columns


def _prediction_frame(
    test: pd.DataFrame,
    stage_probabilities: Mapping[str, np.ndarray],
    model_name: str,
    test_year: int,
) -> pd.DataFrame:
    keys = [
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
            "finish_class",
            "finish_class_index",
        )
        if column in test.columns
    ]
    out = test[keys].copy()
    out["model_name"] = str(model_name)
    out["model_version"] = HIERARCHICAL_FINISH_MODEL_VERSION
    out["test_year"] = int(test_year)
    for stage, column in STAGE_PROBABILITY_COLUMNS.items():
        out[column] = np.asarray(stage_probabilities[stage], dtype=float)
    combined = combine_hierarchical_probabilities(
        out["prob_finish"].to_numpy(dtype=float),
        out["prob_submission_given_finish"].to_numpy(dtype=float),
        out["prob_red_given_ko_tko"].to_numpy(dtype=float),
        out["prob_red_given_submission"].to_numpy(dtype=float),
    )
    for index, column in enumerate(PROBABILITY_COLUMNS):
        out[column] = combined[:, index]
    return out


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def _odds_factor(actual_rate: float, predicted_rate: float) -> float:
    actual = float(np.clip(actual_rate, 1e-5, 1.0 - 1e-5))
    predicted = float(np.clip(predicted_rate, 1e-5, 1.0 - 1e-5))
    actual_odds = actual / (1.0 - actual)
    predicted_odds = predicted / (1.0 - predicted)
    return float(np.clip(actual_odds / predicted_odds, 0.25, 4.0))


def apply_odds_factor(probabilities: np.ndarray, factor: float) -> np.ndarray:
    values = _clip_probability(probabilities)
    logits = np.log(values / (1.0 - values)) + log(float(factor))
    return 1.0 / (1.0 + np.exp(-logits))


def _stage_actual(frame: pd.DataFrame, stage: str) -> np.ndarray:
    finish_class = frame["finish_class"].astype(str)
    if stage == "finish":
        return finish_class.ne("no_finish").to_numpy(dtype=float)
    if stage == "submission_given_finish":
        return finish_class.isin(
            ["red_submission", "blue_submission"]
        ).to_numpy(dtype=float)
    if stage == "red_given_ko_tko":
        return finish_class.eq("red_ko_tko").to_numpy(dtype=float)
    if stage == "red_given_submission":
        return finish_class.eq("red_submission").to_numpy(dtype=float)
    raise HierarchicalFinishHazardModelError(f"Unknown hierarchy stage: {stage}")


def sequential_hierarchical_calibration(
    predictions: pd.DataFrame,
    minimum_prior_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate each hierarchy stage from completed prior holdout years only."""
    required = {
        "model_name",
        "test_year",
        "finish_class",
        *STAGE_PROBABILITY_COLUMNS.values(),
        *PROBABILITY_COLUMNS,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise HierarchicalFinishHazardModelError(
            f"Hierarchical predictions are missing calibration columns: {missing}"
        )
    if minimum_prior_rows <= 0:
        raise HierarchicalFinishHazardModelError(
            "minimum_prior_rows must be positive"
        )

    calibrated_frames: list[pd.DataFrame] = []
    schedule_rows: list[dict[str, object]] = []
    for model_name, model_frame in predictions.groupby("model_name", sort=False):
        for year in sorted(model_frame["test_year"].unique()):
            current = model_frame.loc[model_frame["test_year"].eq(year)].copy()
            prior = model_frame.loc[model_frame["test_year"].lt(year)].copy()
            factors: dict[str, float] = {}
            row: dict[str, object] = {
                "model_name": str(model_name),
                "test_year": int(year),
                "calibration_source": "prior_walk_forward_hierarchy",
            }
            for stage in STAGES:
                mask = _stage_mask(prior, stage) if not prior.empty else pd.Series(
                    False, index=prior.index
                )
                stage_prior = prior.loc[mask]
                predicted_column = STAGE_PROBABILITY_COLUMNS[stage]
                if len(stage_prior) >= minimum_prior_rows:
                    actual = _stage_actual(stage_prior, stage)
                    predicted = stage_prior[predicted_column].to_numpy(dtype=float)
                    factor = _odds_factor(float(actual.mean()), float(predicted.mean()))
                    source = "prior_walk_forward_years"
                else:
                    factor = 1.0
                    source = "cold_start_identity"
                factors[stage] = factor
                row[f"factor_{stage}"] = float(factor)
                row[f"prior_rows_{stage}"] = int(len(stage_prior))
                row[f"source_{stage}"] = source
                current[CALIBRATED_STAGE_COLUMNS[stage]] = apply_odds_factor(
                    current[predicted_column].to_numpy(dtype=float),
                    factor,
                )

            combined = combine_hierarchical_probabilities(
                current["calibrated_prob_finish"].to_numpy(dtype=float),
                current[
                    "calibrated_prob_submission_given_finish"
                ].to_numpy(dtype=float),
                current["calibrated_prob_red_given_ko_tko"].to_numpy(dtype=float),
                current[
                    "calibrated_prob_red_given_submission"
                ].to_numpy(dtype=float),
            )
            for index, column in enumerate(PROBABILITY_COLUMNS):
                current[f"calibrated_{column}"] = combined[:, index]
            calibrated_frames.append(current)
            schedule_rows.append(row)

    calibrated = pd.concat(calibrated_frames, ignore_index=True)
    calibrated_columns = [f"calibrated_{column}" for column in PROBABILITY_COLUMNS]
    matrix = calibrated[calibrated_columns].to_numpy(dtype=float)
    if np.any(matrix < 0.0) or not np.allclose(
        matrix.sum(axis=1), 1.0, atol=1e-8
    ):
        raise HierarchicalFinishHazardModelError(
            "Calibrated hierarchical probabilities violate the simplex"
        )
    return calibrated, pd.DataFrame(schedule_rows)


def calibration_factors_for_year(
    schedule: pd.DataFrame,
    model_name: str,
    test_year: int,
) -> dict[str, float]:
    required = {
        "model_name",
        "test_year",
        *[f"factor_{stage}" for stage in STAGES],
    }
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise HierarchicalFinishHazardModelError(
            f"Hierarchical calibration schedule is missing columns: {missing}"
        )
    match = schedule.loc[
        schedule["model_name"].astype(str).eq(str(model_name))
        & pd.to_numeric(schedule["test_year"], errors="coerce").eq(int(test_year))
    ]
    if len(match) != 1:
        raise HierarchicalFinishHazardModelError(
            f"Expected one calibration row for {model_name}/{test_year}; found {len(match)}"
        )
    row = match.iloc[0]
    factors = {stage: float(row[f"factor_{stage}"]) for stage in STAGES}
    if not np.isfinite(list(factors.values())).all() or any(
        factor <= 0.0 for factor in factors.values()
    ):
        raise HierarchicalFinishHazardModelError(
            "Hierarchical calibration factors must be finite and positive"
        )
    return factors


def apply_hierarchical_calibration_factors(
    stage_probabilities: Mapping[str, np.ndarray],
    factors: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    calibrated = {
        stage: apply_odds_factor(
            np.asarray(stage_probabilities[stage], dtype=float),
            float(factors[stage]),
        )
        for stage in STAGES
    }
    combined = combine_hierarchical_probabilities(
        calibrated["finish"],
        calibrated["submission_given_finish"],
        calibrated["red_given_ko_tko"],
        calibrated["red_given_submission"],
    )
    return calibrated, combined


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    targets = np.asarray(actual, dtype=int)
    probabilities = _clip_probability(predicted)
    return {
        "rows": int(len(targets)),
        "actual_rate": float(targets.mean()),
        "predicted_rate": float(probabilities.mean()),
        "accuracy": float(np.mean((probabilities >= 0.5).astype(int) == targets)),
        "brier": float(np.mean(np.square(probabilities - targets))),
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1])),
    }


def hierarchical_stage_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, year), frame in predictions.groupby(
        ["model_name", "test_year"], sort=True
    ):
        for calibration, columns in (
            ("raw", STAGE_PROBABILITY_COLUMNS),
            ("sequential_hierarchical_calibrated", CALIBRATED_STAGE_COLUMNS),
        ):
            for stage in STAGES:
                mask = _stage_mask(frame, stage)
                subset = frame.loc[mask]
                metric = _binary_metrics(
                    _stage_actual(subset, stage),
                    subset[columns[stage]].to_numpy(dtype=float),
                )
                metric.update(
                    {
                        "model_name": model_name,
                        "test_year": int(year),
                        "stage": stage,
                        "calibration": calibration,
                    }
                )
                rows.append(metric)
    return pd.DataFrame(rows)


def walk_forward_hierarchical_finish_benchmark(
    training_df: pd.DataFrame,
    test_years: Sequence[int] = DEFAULT_TEST_YEARS,
    seed: int = 7,
) -> HierarchicalFinishBenchmarkResult:
    """Run expanding-year hierarchical finish benchmarks."""
    dataset = add_hierarchical_targets(prepare_finish_hazard_dataset(training_df))
    raw_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for fold_index, test_year in enumerate(test_years):
        train = dataset.loc[dataset["date"].dt.year.lt(int(test_year))].copy()
        test = dataset.loc[dataset["date"].dt.year.eq(int(test_year))].copy()
        if train.empty or test.empty:
            raise HierarchicalFinishHazardModelError(
                f"Empty train/test split for holdout year {test_year}"
            )

        for model_index, (model_name, include_rfs) in enumerate(
            (
                (HIERARCHICAL_MODELS[0], False),
                (HIERARCHICAL_MODELS[1], True),
            )
        ):
            probabilities, importance, features = fit_predict_hierarchical_stages(
                train,
                test,
                include_rfs=include_rfs,
                seed=int(seed + fold_index * 101 + model_index * 17),
            )
            raw_frames.append(
                _prediction_frame(test, probabilities, model_name, int(test_year))
            )
            importance["model_name"] = model_name
            importance["test_year"] = int(test_year)
            importance_frames.append(importance)
            for stage in STAGES:
                for feature in features:
                    manifest_rows.append(
                        {
                            "model_name": model_name,
                            "test_year": int(test_year),
                            "stage": stage,
                            "feature": feature,
                        }
                    )

    raw_predictions = pd.concat(raw_frames, ignore_index=True)
    calibrated_predictions, schedule = sequential_hierarchical_calibration(
        raw_predictions
    )
    calibrated_columns = [f"calibrated_{column}" for column in PROBABILITY_COLUMNS]

    fold_rows: list[dict[str, object]] = []
    for (model_name, year), frame in calibrated_predictions.groupby(
        ["model_name", "test_year"], sort=True
    ):
        for calibration, columns in (
            ("raw", PROBABILITY_COLUMNS),
            ("sequential_hierarchical_calibrated", calibrated_columns),
        ):
            metric = _metrics(frame, columns)
            metric.update(
                {
                    "model_name": model_name,
                    "test_year": int(year),
                    "calibration": calibration,
                }
            )
            fold_rows.append(metric)
    fold_metrics = pd.DataFrame(fold_rows)

    aggregate_rows: list[dict[str, object]] = []
    for model_name, frame in calibrated_predictions.groupby("model_name", sort=True):
        for calibration, columns in (
            ("raw", PROBABILITY_COLUMNS),
            ("sequential_hierarchical_calibrated", calibrated_columns),
        ):
            metric = _metrics(frame, columns)
            metric.update(
                {
                    "model_name": model_name,
                    "test_year": "all",
                    "calibration": calibration,
                }
            )
            aggregate_rows.append(metric)

    return HierarchicalFinishBenchmarkResult(
        raw_predictions=raw_predictions,
        calibrated_predictions=calibrated_predictions,
        fold_metrics=fold_metrics,
        aggregate_metrics=pd.DataFrame(aggregate_rows),
        stage_metrics=hierarchical_stage_metrics(calibrated_predictions),
        calibration_schedule=schedule,
        feature_importance=pd.concat(importance_frames, ignore_index=True),
        feature_manifest=pd.DataFrame(manifest_rows),
    )
