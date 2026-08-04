"""Stability experiment for compact Round Fighter State strike-pace features.

The first significant-strike benchmark showed a small pooled gain from the full
RFS feature set, but the gain reversed in the newest holdout. This module tests a
smaller ontology-driven RFS bundle and an optional recency-weighted variant.

Feature selection is deterministic and does not use holdout importance. All
training folds remain date-expanding, and sequential calibration uses only prior
walk-forward years.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from xgboost import XGBRegressor

from pipeline.simulation.sig_attempt_calibration import (
    calibrate_walk_forward_predictions,
)
from pipeline.simulation.sig_attempt_model import (
    DEFAULT_TEST_YEARS,
    IDENTITY_COLUMNS,
    prepare_sig_attempt_dataset,
    select_model_columns,
)


class SigAttemptStabilityError(RuntimeError):
    """Raised when the compact-RFS stability experiment is invalid."""


@dataclass(frozen=True)
class SigAttemptStabilityResult:
    raw_predictions: pd.DataFrame
    calibrated_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    subgroup_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_manifest: pd.DataFrame
    gates: pd.DataFrame


COMPACT_MODEL_NAMES = (
    "xgb_context_rfs_compact",
    "xgb_context_rfs_compact_recent",
)
REFERENCE_MODEL_NAMES = ("xgb_context", "xgb_context_rfs")
ALL_MODEL_NAMES = (*REFERENCE_MODEL_NAMES, *COMPACT_MODEL_NAMES)

COMPACT_RFS_TOKENS = (
    "opp_sig_attempt_delta",
    "opp_total_attempt_delta",
    "opp_distance_share_delta",
    "opp_clinch_share_delta",
    "sig_attempt_slope",
    "sig_attempt_late_ratio",
    "total_attempt_slope",
    "total_attempt_late_ratio",
    "control_seconds_slope",
    "control_late_ratio",
    "td_attempt_slope",
    "td_persistence_score",
    "control_per_td_attempt",
    "failed_td_persistence",
    "opp_sig_accuracy_allowed_slope",
    "head_absorbed_slope",
    "defensive_deterioration_score",
)
COMPACT_RFS_HORIZONS = ("_exp_", "_ewm_")


@dataclass(frozen=True)
class _MatrixSpec:
    numeric_columns: tuple[str, ...]
    categorical_levels: Mapping[str, tuple[str, ...]]
    feature_columns: tuple[str, ...]


def _require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SigAttemptStabilityError(f"{label} is missing required columns: {missing}")


def select_compact_rfs_columns(df: pd.DataFrame) -> list[str]:
    """Select a deterministic, strike-relevant subset of RFS columns.

    The bundle retains expanding and EWM state estimates, availability flags,
    and valid-history counts. It intentionally excludes noisier last-three
    variants and unrelated high-dimensional state fields.
    """
    selected: list[str] = []
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if column.endswith("_state_available"):
            selected.append(column)
            continue
        if not column.startswith(("fighter_rfs_", "opponent_rfs_")):
            continue
        if "prior_valid_" in column:
            selected.append(column)
            continue
        if not any(marker in column for marker in COMPACT_RFS_HORIZONS):
            continue
        if any(token in column for token in COMPACT_RFS_TOKENS):
            selected.append(column)
    return sorted(set(selected))


def build_recency_weights(
    train_df: pd.DataFrame,
    test_year: int,
    half_life_years: float = 4.0,
    floor: float = 0.15,
) -> np.ndarray:
    """Combine exposure weights with leakage-safe age decay.

    Weights are normalized back to the original exposure-weight mean so changing
    recency does not unintentionally change XGBoost's total effective weight.
    """
    if half_life_years <= 0:
        raise SigAttemptStabilityError("half_life_years must be positive")
    if not 0 < floor <= 1:
        raise SigAttemptStabilityError("recency floor must be in (0, 1]")
    _require_columns(train_df, ("date", "exposure_weight"), "Training frame")

    dates = pd.to_datetime(train_df["date"], errors="coerce")
    if dates.isna().any():
        raise SigAttemptStabilityError("Training dates must be valid")
    cutoff = pd.Timestamp(year=int(test_year), month=1, day=1)
    age_years = ((cutoff - dates).dt.days.clip(lower=0) / 365.25).to_numpy(dtype=float)
    recency = np.power(0.5, age_years / float(half_life_years))
    recency = np.maximum(recency, float(floor))

    exposure = pd.to_numeric(train_df["exposure_weight"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise SigAttemptStabilityError("Exposure weights must be finite and positive")
    combined = exposure * recency
    combined_mean = float(combined.mean())
    if combined_mean <= 0:
        raise SigAttemptStabilityError("Combined recency weights must be positive")
    return combined * (float(exposure.mean()) / combined_mean)


def _sanitize_feature_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return text[:120] or "missing"


def _fit_matrix_spec(
    train_df: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> _MatrixSpec:
    levels: dict[str, tuple[str, ...]] = {}
    features = list(numeric_columns)
    for column in categorical_columns:
        values = train_df[column].astype("string").fillna("__MISSING__")
        categories = tuple(sorted(str(value) for value in values.unique()))
        levels[column] = categories
        features.extend(
            f"cat__{_sanitize_feature_name(column)}__{_sanitize_feature_name(category)}"
            for category in categories
        )
    return _MatrixSpec(
        numeric_columns=tuple(numeric_columns),
        categorical_levels=levels,
        feature_columns=tuple(features),
    )


def _transform_matrix(df: pd.DataFrame, spec: _MatrixSpec) -> pd.DataFrame:
    matrix = pd.DataFrame(index=df.index)
    for column in spec.numeric_columns:
        matrix[column] = pd.to_numeric(df[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    for column, categories in spec.categorical_levels.items():
        values = df[column].astype("string").fillna("__MISSING__")
        for category in categories:
            feature = (
                f"cat__{_sanitize_feature_name(column)}__"
                f"{_sanitize_feature_name(category)}"
            )
            matrix[feature] = values.eq(category).astype(float)
    return matrix.reindex(columns=spec.feature_columns)


def _new_compact_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=280,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=30.0,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.15,
        reg_lambda=12.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
    )


def _fit_compact_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_year: int,
    seed: int,
    use_recency: bool,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    context_numeric, categorical = select_model_columns(train_df, include_rfs=False)
    compact_rfs = select_compact_rfs_columns(train_df)
    if not compact_rfs:
        raise SigAttemptStabilityError("No compact RFS features were selected")
    numeric = sorted(set(context_numeric) | set(compact_rfs))
    spec = _fit_matrix_spec(train_df, numeric, categorical)
    x_train = _transform_matrix(train_df, spec)
    x_test = _transform_matrix(test_df, spec)
    y_train = np.log1p(
        pd.to_numeric(
            train_df["target_sig_attempt_rate_per_min"], errors="coerce"
        ).to_numpy(dtype=float)
    )
    if not np.isfinite(y_train).all():
        raise SigAttemptStabilityError("Training target contains non-finite values")

    if use_recency:
        weights = build_recency_weights(train_df, test_year=test_year)
    else:
        weights = pd.to_numeric(
            train_df["exposure_weight"], errors="coerce"
        ).to_numpy(dtype=float)

    model = _new_compact_model(seed)
    model.fit(x_train, y_train, sample_weight=weights)
    prediction = np.expm1(model.predict(x_test))
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.001, None)
    importance = pd.DataFrame(
        {"feature": spec.feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return prediction, importance, compact_rfs


def _calendar_period(year: int) -> str:
    if year <= 2020:
        return "2018_2020"
    if year <= 2023:
        return "2021_2023"
    return "2024_2026"


def _availability_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in df.columns
        if column.endswith("_state_available")
        and pd.api.types.is_numeric_dtype(df[column])
    )


def _metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id", "round"]
    _require_columns(df, keys, "Prepared training frame")
    columns = [*keys]
    for column in ("division", "date"):
        if column in df.columns:
            columns.append(column)
    availability = _availability_columns(df)
    columns.extend(availability)
    out = df[columns].copy()
    if out.duplicated(keys).any():
        raise SigAttemptStabilityError("Prepared metadata has duplicate fighter-round keys")
    if availability:
        available = out[availability].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["rfs_availability_fraction"] = available.mean(axis=1)
        out["rfs_coverage"] = np.select(
            [
                out["rfs_availability_fraction"].ge(0.999),
                out["rfs_availability_fraction"].gt(0.0),
            ],
            ["complete", "partial"],
            default="none",
        )
    else:
        out["rfs_availability_fraction"] = np.nan
        out["rfs_coverage"] = "unknown"
    return out[[
        *keys,
        *[column for column in ("division", "date") if column in out.columns],
        "rfs_availability_fraction",
        "rfs_coverage",
    ]]


def _enrich_reference_predictions(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    test_years: Sequence[int],
) -> pd.DataFrame:
    required = (
        "model_name",
        "test_year",
        "fight_id",
        "fighter_id",
        "round",
        "round_exposure_seconds",
        "target_sig_attempted",
        "predicted_rate_per_min",
        "predicted_count_at_actual_exposure",
    )
    _require_columns(predictions, required, "Reference predictions")
    out = predictions.loc[
        predictions["model_name"].isin(REFERENCE_MODEL_NAMES)
        & predictions["test_year"].isin([int(year) for year in test_years])
    ].copy()
    available = set(out["model_name"].astype(str).unique())
    missing = sorted(set(REFERENCE_MODEL_NAMES) - available)
    if missing:
        raise SigAttemptStabilityError(
            f"Reference predictions are missing models: {missing}"
        )
    add_columns = [
        column
        for column in metadata.columns
        if column not in ("fight_id", "fighter_id", "round")
        and column not in out.columns
    ]
    out = out.merge(
        metadata[["fight_id", "fighter_id", "round", *add_columns]],
        on=["fight_id", "fighter_id", "round"],
        how="left",
        validate="many_to_one",
    )
    return out


def _compact_prediction_frame(
    test_df: pd.DataFrame,
    predicted_rate: np.ndarray,
    model_name: str,
    test_year: int,
) -> pd.DataFrame:
    keys = [column for column in IDENTITY_COLUMNS if column in test_df.columns]
    columns = [
        *keys,
        "round",
        "round_exposure_seconds",
        "target_sig_attempted",
        "target_sig_attempt_rate_per_min",
        "exposure_weight",
    ]
    if "division" in test_df.columns and "division" not in columns:
        columns.append("division")
    frame = test_df[columns].copy()
    metadata = _metadata_frame(test_df)
    add_columns = [
        column
        for column in metadata.columns
        if column not in ("fight_id", "fighter_id", "round")
        and column not in frame.columns
    ]
    frame = frame.merge(
        metadata[["fight_id", "fighter_id", "round", *add_columns]],
        on=["fight_id", "fighter_id", "round"],
        how="left",
        validate="one_to_one",
    )
    frame["model_name"] = model_name
    frame["test_year"] = int(test_year)
    frame["calendar_period"] = _calendar_period(int(test_year))
    frame["predicted_rate_per_min"] = np.asarray(predicted_rate, dtype=float)
    frame["predicted_count_at_actual_exposure"] = (
        frame["predicted_rate_per_min"]
        * frame["round_exposure_seconds"].to_numpy(dtype=float)
        / 60.0
    )
    return frame


def _add_calendar_period(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["calendar_period"] = out["test_year"].astype(int).map(_calendar_period)
    return out


def _metric_row(
    model_name: str,
    calibration: str,
    frame: pd.DataFrame,
    predicted_count_column: str,
    test_year: int | str,
) -> dict[str, object]:
    actual = pd.to_numeric(frame["target_sig_attempted"], errors="coerce").to_numpy(
        dtype=float
    )
    predicted = np.clip(
        pd.to_numeric(frame[predicted_count_column], errors="coerce").to_numpy(
            dtype=float
        ),
        0.001,
        None,
    )
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise SigAttemptStabilityError("Metric inputs contain non-finite values")
    return {
        "model_name": model_name,
        "calibration": calibration,
        "test_year": test_year,
        "rows": int(len(frame)),
        "fights": int(frame["fight_id"].nunique()),
        "count_mae": float(mean_absolute_error(actual, predicted)),
        "count_rmse": float(sqrt(mean_squared_error(actual, predicted))),
        "count_poisson_deviance": float(mean_poisson_deviance(actual, predicted)),
        "actual_mean_count": float(actual.mean()),
        "predicted_mean_count": float(predicted.mean()),
        "mean_count_bias": float(predicted.mean() - actual.mean()),
    }


def _metric_tables(
    raw_predictions: pd.DataFrame,
    calibrated_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    sources = (
        (raw_predictions, "raw", "predicted_count_at_actual_exposure"),
        (
            calibrated_predictions,
            "sequential_mean_calibrated",
            "calibrated_count_at_actual_exposure",
        ),
    )
    for frame, calibration, prediction_column in sources:
        for (model_name, year), group in frame.groupby(["model_name", "test_year"]):
            rows.append(
                _metric_row(
                    str(model_name),
                    calibration,
                    group,
                    prediction_column,
                    int(year),
                )
            )
        for model_name, group in frame.groupby("model_name"):
            aggregate_rows.append(
                _metric_row(
                    str(model_name),
                    calibration,
                    group,
                    prediction_column,
                    "all_walk_forward",
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(aggregate_rows)


def _subgroup_metrics(
    raw_predictions: pd.DataFrame,
    calibrated_predictions: pd.DataFrame,
    minimum_rows: int = 100,
) -> pd.DataFrame:
    dimensions = ("round", "division", "calendar_period", "rfs_coverage")
    rows: list[dict[str, object]] = []
    sources = (
        (raw_predictions, "raw", "predicted_count_at_actual_exposure"),
        (
            calibrated_predictions,
            "sequential_mean_calibrated",
            "calibrated_count_at_actual_exposure",
        ),
    )
    for frame, calibration, prediction_column in sources:
        for dimension in dimensions:
            if dimension not in frame.columns:
                continue
            for (model_name, value), group in frame.groupby(
                ["model_name", dimension], dropna=False
            ):
                if len(group) < int(minimum_rows):
                    continue
                row = _metric_row(
                    str(model_name),
                    calibration,
                    group,
                    prediction_column,
                    "subgroup",
                )
                row["group_name"] = dimension
                row["group_value"] = str(value)
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate_stability_gates(
    aggregate_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    candidates: Iterable[str] = COMPACT_MODEL_NAMES,
    baseline: str = "xgb_context",
    min_aggregate_improvement: float = 0.005,
    min_latest_improvement: float = 0.0,
    min_winning_years: int = 3,
    max_abs_bias: float = 1.5,
    max_subgroup_regression: float = -0.05,
) -> pd.DataFrame:
    """Evaluate conservative research gates against context-only XGBoost."""
    calibration = "sequential_mean_calibrated"
    aggregate = aggregate_metrics.loc[
        aggregate_metrics["calibration"].eq(calibration)
    ].set_index("model_name")
    folds = fold_metrics.loc[fold_metrics["calibration"].eq(calibration)].copy()
    subgroups = subgroup_metrics.loc[
        subgroup_metrics["calibration"].eq(calibration)
    ].copy()
    if baseline not in aggregate.index:
        raise SigAttemptStabilityError(f"Baseline model is missing: {baseline}")

    rows: list[dict[str, object]] = []
    baseline_aggregate = float(aggregate.loc[baseline, "count_poisson_deviance"])
    latest_year = int(pd.to_numeric(folds["test_year"], errors="coerce").max())

    for candidate in candidates:
        if candidate not in aggregate.index:
            raise SigAttemptStabilityError(f"Candidate model is missing: {candidate}")
        candidate_aggregate = float(
            aggregate.loc[candidate, "count_poisson_deviance"]
        )
        aggregate_improvement = (
            baseline_aggregate - candidate_aggregate
        ) / baseline_aggregate

        pivot = folds.loc[folds["model_name"].isin([baseline, candidate])].pivot(
            index="test_year",
            columns="model_name",
            values="count_poisson_deviance",
        )
        pivot = pivot.dropna(subset=[baseline, candidate])
        yearly_improvement = (pivot[baseline] - pivot[candidate]) / pivot[baseline]
        winning_years = int(yearly_improvement.gt(0).sum())
        if latest_year not in yearly_improvement.index:
            raise SigAttemptStabilityError(
                f"Latest year {latest_year} is missing for candidate {candidate}"
            )
        latest_improvement = float(yearly_improvement.loc[latest_year])
        bias = float(aggregate.loc[candidate, "mean_count_bias"])

        comparison = subgroups.loc[
            subgroups["model_name"].isin([baseline, candidate])
        ].pivot_table(
            index=["group_name", "group_value"],
            columns="model_name",
            values="count_poisson_deviance",
            aggfunc="first",
        )
        comparison = comparison.dropna(subset=[baseline, candidate])
        if comparison.empty:
            worst_subgroup_improvement = np.nan
            subgroup_pass = False
        else:
            subgroup_improvement = (
                comparison[baseline] - comparison[candidate]
            ) / comparison[baseline]
            worst_subgroup_improvement = float(subgroup_improvement.min())
            subgroup_pass = worst_subgroup_improvement >= max_subgroup_regression

        checks = {
            "aggregate_pass": aggregate_improvement >= min_aggregate_improvement,
            "latest_year_pass": latest_improvement >= min_latest_improvement,
            "winning_years_pass": winning_years >= int(min_winning_years),
            "bias_pass": abs(bias) <= max_abs_bias,
            "subgroup_pass": bool(subgroup_pass),
        }
        reasons = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "candidate_model": candidate,
                "baseline_model": baseline,
                "latest_year": latest_year,
                "aggregate_poisson_improvement": float(aggregate_improvement),
                "latest_year_poisson_improvement": latest_improvement,
                "winning_years": winning_years,
                "evaluated_years": int(len(yearly_improvement)),
                "mean_count_bias": bias,
                "worst_subgroup_poisson_improvement": worst_subgroup_improvement,
                **checks,
                "gate_status": "pass" if not reasons else "blocked",
                "blocking_reasons": ",".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_sig_attempt_stability(
    training_df: pd.DataFrame,
    reference_predictions: pd.DataFrame,
    test_years: Iterable[int] = DEFAULT_TEST_YEARS,
    seed: int = 7,
    minimum_prior_rows: int = 1_000,
    minimum_subgroup_rows: int = 100,
) -> SigAttemptStabilityResult:
    """Run compact-RFS and recency-weighted walk-forward ablations."""
    years_requested = tuple(int(year) for year in test_years)
    if not years_requested:
        raise SigAttemptStabilityError("At least one test year is required")

    df = prepare_sig_attempt_dataset(training_df)
    _require_columns(
        df,
        (
            "fight_id",
            "fighter_id",
            "date",
            "round",
            "target_sig_attempt_rate_per_min",
            "round_exposure_seconds",
            "target_sig_attempted",
            "exposure_weight",
        ),
        "Prepared training frame",
    )
    metadata = _metadata_frame(df)
    references = _enrich_reference_predictions(
        reference_predictions,
        metadata=metadata,
        test_years=years_requested,
    )
    references = _add_calendar_period(references)

    event_year = pd.to_datetime(df["date"], errors="coerce").dt.year
    prediction_frames: list[pd.DataFrame] = [references]
    importance_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for fold_index, test_year in enumerate(years_requested):
        train = df.loc[event_year.lt(test_year)].copy()
        test = df.loc[event_year.eq(test_year)].copy()
        if train.empty or test.empty:
            continue
        for variant_index, (model_name, use_recency) in enumerate(
            (
                ("xgb_context_rfs_compact", False),
                ("xgb_context_rfs_compact_recent", True),
            )
        ):
            prediction, importance, compact_features = _fit_compact_variant(
                train,
                test,
                test_year=test_year,
                seed=seed + fold_index * 10 + variant_index,
                use_recency=use_recency,
            )
            importance.insert(0, "test_year", int(test_year))
            importance.insert(0, "model_name", model_name)
            importance_frames.append(importance)
            prediction_frames.append(
                _compact_prediction_frame(
                    test,
                    prediction,
                    model_name=model_name,
                    test_year=test_year,
                )
            )
            manifest_rows.extend(
                {
                    "model_name": model_name,
                    "test_year": int(test_year),
                    "feature": feature,
                    "feature_family": "compact_rfs",
                }
                for feature in compact_features
            )

    if len(prediction_frames) == 1:
        raise SigAttemptStabilityError("No compact walk-forward folds were available")
    raw_predictions = pd.concat(prediction_frames, ignore_index=True)
    available_models = set(raw_predictions["model_name"].astype(str).unique())
    missing_models = sorted(set(ALL_MODEL_NAMES) - available_models)
    if missing_models:
        raise SigAttemptStabilityError(
            f"Stability predictions are missing models: {missing_models}"
        )

    calibration = calibrate_walk_forward_predictions(
        predictions=raw_predictions,
        model_names=ALL_MODEL_NAMES,
        minimum_prior_rows=minimum_prior_rows,
    )
    calibrated_predictions = calibration.predictions
    fold_metrics, aggregate_metrics = _metric_tables(
        raw_predictions,
        calibrated_predictions,
    )
    subgroup_metrics = _subgroup_metrics(
        raw_predictions,
        calibrated_predictions,
        minimum_rows=minimum_subgroup_rows,
    )
    gates = evaluate_stability_gates(
        aggregate_metrics,
        fold_metrics,
        subgroup_metrics,
    )

    return SigAttemptStabilityResult(
        raw_predictions=raw_predictions,
        calibrated_predictions=calibrated_predictions,
        fold_metrics=fold_metrics.sort_values(
            ["calibration", "test_year", "count_poisson_deviance"]
        ).reset_index(drop=True),
        aggregate_metrics=aggregate_metrics.sort_values(
            ["calibration", "count_poisson_deviance"]
        ).reset_index(drop=True),
        subgroup_metrics=subgroup_metrics.sort_values(
            ["calibration", "group_name", "group_value", "count_poisson_deviance"]
        ).reset_index(drop=True),
        feature_importance=pd.concat(importance_frames, ignore_index=True),
        feature_manifest=pd.DataFrame(manifest_rows).drop_duplicates().reset_index(
            drop=True
        ),
        gates=gates,
    )
