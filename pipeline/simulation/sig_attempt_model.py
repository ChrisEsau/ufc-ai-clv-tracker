"""Walk-forward benchmark for the simulator's significant-strike pace component.

The simulator consumes striking pace, not a terminal prop probability. Historical
terminal rounds may be shorter than five minutes, so this module models an
exposure-adjusted significant-strike attempt rate and converts predictions back to
counts for evaluation.

All fighter-history features are calculated at the completed-fight grain and
shifted before they are joined to the target fight. Current-fight target values
never contribute to their own pre-fight pace state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import re
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from xgboost import XGBRegressor


SIG_ATTEMPT_MODEL_VERSION = "sig_attempt_pace_v0"
DEFAULT_TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
ROUND_SECONDS = 300.0
HISTORY_SHRINKAGE_MINUTES = 30.0


class SigAttemptModelError(RuntimeError):
    """Raised when significant-strike benchmark data or outputs are invalid."""


@dataclass(frozen=True)
class SigAttemptBenchmarkResult:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    final_bundle: Mapping[str, object]


REQUIRED_COLUMNS = (
    "fight_id",
    "fighter_id",
    "opponent_id",
    "date",
    "round",
    "target_sig_attempted",
    "target_finish_time_in_round_seconds",
)

IDENTITY_COLUMNS = (
    "event_id",
    "event_name",
    "fight_id",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "date",
)

BASE_CONTEXT_COLUMNS = (
    "round",
    "total_rounds",
    "title_fight",
    "prior_rounds_completed",
    "rounds_remaining_including_current",
    "elapsed_seconds_before_round",
    "scheduled_fight_seconds",
    "opponent_rounds_remaining_including_current",
    "opponent_elapsed_seconds_before_round",
)

CATEGORICAL_COLUMNS = ("division", "corner")


@dataclass(frozen=True)
class _MatrixSpec:
    numeric_columns: tuple[str, ...]
    categorical_levels: Mapping[str, tuple[str, ...]]
    feature_columns: tuple[str, ...]


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SigAttemptModelError(f"Training table is missing required columns: {missing}")


def _safe_rate(attempts: pd.Series, exposure_seconds: pd.Series) -> pd.Series:
    exposure_minutes = pd.to_numeric(exposure_seconds, errors="coerce") / 60.0
    attempts_numeric = pd.to_numeric(attempts, errors="coerce")
    return attempts_numeric / exposure_minutes.replace(0, np.nan)


def _add_prefight_fighter_pace_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior-fight pace states for fighter and opponent.

    Each fight is first collapsed to one fighter-fight observation. Expanding,
    recent-three, and EWM states are shifted one complete fight before being
    merged back to every round of the target fight.
    """
    fight = (
        df.groupby(["fight_id", "fighter_id", "date"], dropna=False)
        .agg(
            fight_sig_attempts=("target_sig_attempted", "sum"),
            fight_exposure_seconds=("round_exposure_seconds", "sum"),
        )
        .reset_index()
        .sort_values(["fighter_id", "date", "fight_id"])
        .reset_index(drop=True)
    )
    fight["fight_sig_attempt_rate_per_min"] = _safe_rate(
        fight["fight_sig_attempts"],
        fight["fight_exposure_seconds"],
    )

    group = fight.groupby("fighter_id", sort=False, group_keys=False)
    fight["fighter_prior_fight_count"] = group.cumcount()

    cumulative_attempts = group["fight_sig_attempts"].cumsum() - fight["fight_sig_attempts"]
    cumulative_exposure = (
        group["fight_exposure_seconds"].cumsum() - fight["fight_exposure_seconds"]
    )
    fight["fighter_prior_exposure_minutes"] = cumulative_exposure / 60.0
    fight["fighter_prior_sig_attempt_rate_exp"] = _safe_rate(
        cumulative_attempts,
        cumulative_exposure,
    )

    last3_attempts = group["fight_sig_attempts"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).sum()
    )
    last3_exposure = group["fight_exposure_seconds"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).sum()
    )
    fight["fighter_prior_sig_attempt_rate_last3"] = _safe_rate(
        last3_attempts,
        last3_exposure,
    )
    fight["fighter_prior_sig_attempt_rate_ewm"] = group[
        "fight_sig_attempt_rate_per_min"
    ].transform(
        lambda values: values.shift(1).ewm(alpha=0.35, adjust=False).mean()
    )

    pace_columns = [
        "fighter_prior_fight_count",
        "fighter_prior_exposure_minutes",
        "fighter_prior_sig_attempt_rate_exp",
        "fighter_prior_sig_attempt_rate_last3",
        "fighter_prior_sig_attempt_rate_ewm",
    ]
    fighter_state = fight[["fight_id", "fighter_id", *pace_columns]].copy()
    out = df.merge(
        fighter_state,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="many_to_one",
    )

    opponent_state = fighter_state.rename(
        columns={
            "fighter_id": "opponent_id",
            **{
                column: column.replace("fighter_", "opponent_", 1)
                for column in pace_columns
            },
        }
    )
    out = out.merge(
        opponent_state,
        on=["fight_id", "opponent_id"],
        how="left",
        validate="many_to_one",
    )
    return out


def prepare_sig_attempt_dataset(training_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare exposure-adjusted labels and shifted fighter pace states."""
    _require_columns(training_df, REQUIRED_COLUMNS)
    df = training_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise SigAttemptModelError("Significant-strike training rows require valid dates")

    df["target_sig_attempted"] = pd.to_numeric(
        df["target_sig_attempted"], errors="coerce"
    )
    df["round_exposure_seconds"] = pd.to_numeric(
        df["target_finish_time_in_round_seconds"], errors="coerce"
    )
    invalid_exposure = (
        df["round_exposure_seconds"].isna()
        | df["round_exposure_seconds"].le(0)
        | df["round_exposure_seconds"].gt(ROUND_SECONDS)
    )
    if invalid_exposure.any():
        raise SigAttemptModelError(
            f"Invalid round exposure rows: {int(invalid_exposure.sum())}"
        )
    if df["target_sig_attempted"].isna().any() or df["target_sig_attempted"].lt(0).any():
        raise SigAttemptModelError("target_sig_attempted must be finite and nonnegative")

    df["target_sig_attempt_rate_per_min"] = _safe_rate(
        df["target_sig_attempted"],
        df["round_exposure_seconds"],
    )
    df["target_sig_attempts_full_round_equivalent"] = (
        df["target_sig_attempt_rate_per_min"] * 5.0
    )
    df["exposure_weight"] = (df["round_exposure_seconds"] / ROUND_SECONDS).clip(
        lower=0.02,
        upper=1.0,
    )
    df = _add_prefight_fighter_pace_history(df)

    duplicate_count = int(df.duplicated(["fight_id", "fighter_id", "round"]).sum())
    if duplicate_count:
        raise SigAttemptModelError(
            f"Prepared strike table has duplicate fighter-round keys: {duplicate_count}"
        )
    return df.sort_values(["date", "fight_id", "fighter_id", "round"]).reset_index(
        drop=True
    )


def _context_numeric_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in BASE_CONTEXT_COLUMNS if column in df.columns]
    columns.extend(
        column
        for column in df.columns
        if column.startswith(("prior_", "opponent_prior_"))
    )
    columns.extend(
        column
        for column in df.columns
        if column.startswith(
            (
                "fighter_prior_sig_attempt_",
                "opponent_prior_sig_attempt_",
                "fighter_prior_fight_count",
                "opponent_prior_fight_count",
                "fighter_prior_exposure_minutes",
                "opponent_prior_exposure_minutes",
            )
        )
    )
    return sorted(set(columns))


def select_model_columns(df: pd.DataFrame, include_rfs: bool) -> tuple[list[str], list[str]]:
    """Return model feature columns without identities or targets."""
    numeric = _context_numeric_columns(df)
    if include_rfs:
        numeric.extend(
            column
            for column in df.columns
            if column.startswith(("fighter_rfs_", "opponent_rfs_"))
            or column.endswith("_state_available")
        )
    numeric = sorted(
        column
        for column in set(numeric)
        if column not in IDENTITY_COLUMNS
        and not column.startswith("target_")
        and pd.api.types.is_numeric_dtype(df[column])
    )
    categorical = [column for column in CATEGORICAL_COLUMNS if column in df.columns]
    return numeric, categorical


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


def _new_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=320,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=20.0,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.05,
        reg_lambda=8.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
    )


def _round_rate_lookup(train_df: pd.DataFrame) -> tuple[dict[int, float], float]:
    global_rate = float(
        train_df["target_sig_attempted"].sum()
        / (train_df["round_exposure_seconds"].sum() / 60.0)
    )
    grouped = train_df.groupby("round", dropna=False).agg(
        attempts=("target_sig_attempted", "sum"),
        exposure=("round_exposure_seconds", "sum"),
    )
    round_rates = {
        int(round_number): float(row.attempts / (row.exposure / 60.0))
        for round_number, row in grouped.iterrows()
        if row.exposure > 0
    }
    return round_rates, global_rate


def _round_mean_prediction(
    frame: pd.DataFrame,
    round_rates: Mapping[int, float],
    global_rate: float,
) -> np.ndarray:
    return np.asarray(
        [round_rates.get(int(round_number), global_rate) for round_number in frame["round"]],
        dtype=float,
    )


def _fighter_history_prediction(
    frame: pd.DataFrame,
    round_rate: np.ndarray,
) -> np.ndarray:
    # pandas 3 copy-on-write can expose a read-only NumPy view for native
    # float64 parquet columns. The cold-start fallback mutates this temporary
    # array, so request an explicit writable copy rather than relying on dtype
    # conversion to allocate one.
    fighter_rate = pd.to_numeric(
        frame["fighter_prior_sig_attempt_rate_exp"], errors="coerce"
    ).to_numpy(dtype=float, copy=True)
    exposure = pd.to_numeric(
        frame["fighter_prior_exposure_minutes"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    history_weight = exposure / (exposure + HISTORY_SHRINKAGE_MINUTES)
    missing = ~np.isfinite(fighter_rate)
    fighter_rate[missing] = round_rate[missing]
    history_weight[missing] = 0.0
    return history_weight * fighter_rate + (1.0 - history_weight) * round_rate


def _metric_row(
    model_name: str,
    test_year: int,
    frame: pd.DataFrame,
    predicted_rate: np.ndarray,
) -> dict[str, object]:
    actual_count = frame["target_sig_attempted"].to_numpy(dtype=float)
    exposure_minutes = frame["round_exposure_seconds"].to_numpy(dtype=float) / 60.0
    actual_rate = frame["target_sig_attempt_rate_per_min"].to_numpy(dtype=float)
    predicted_rate = np.clip(np.asarray(predicted_rate, dtype=float), 0.001, None)
    predicted_count = np.clip(predicted_rate * exposure_minutes, 0.001, None)
    weights = frame["exposure_weight"].to_numpy(dtype=float)

    weighted_rate_mae = float(np.average(np.abs(actual_rate - predicted_rate), weights=weights))
    weighted_rate_rmse = float(
        sqrt(np.average(np.square(actual_rate - predicted_rate), weights=weights))
    )
    return {
        "model_name": model_name,
        "test_year": int(test_year),
        "rows": int(len(frame)),
        "fights": int(frame["fight_id"].nunique()),
        "count_mae": float(mean_absolute_error(actual_count, predicted_count)),
        "count_rmse": float(sqrt(mean_squared_error(actual_count, predicted_count))),
        "count_poisson_deviance": float(
            mean_poisson_deviance(actual_count, predicted_count)
        ),
        "weighted_rate_mae": weighted_rate_mae,
        "weighted_rate_rmse": weighted_rate_rmse,
        "actual_mean_count": float(actual_count.mean()),
        "predicted_mean_count": float(predicted_count.mean()),
        "mean_count_bias": float(predicted_count.mean() - actual_count.mean()),
    }


def _fit_predict_xgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_rfs: bool,
    seed: int,
) -> tuple[np.ndarray, XGBRegressor, _MatrixSpec, pd.DataFrame]:
    numeric, categorical = select_model_columns(train_df, include_rfs=include_rfs)
    if not numeric:
        raise SigAttemptModelError("No numeric model features were selected")
    spec = _fit_matrix_spec(train_df, numeric, categorical)
    x_train = _transform_matrix(train_df, spec)
    x_test = _transform_matrix(test_df, spec)
    y_train = np.log1p(train_df["target_sig_attempt_rate_per_min"].to_numpy(dtype=float))
    weights = train_df["exposure_weight"].to_numpy(dtype=float)

    model = _new_model(seed)
    model.fit(x_train, y_train, sample_weight=weights)
    prediction = np.expm1(model.predict(x_test))
    prediction = np.clip(prediction, 0.001, None)

    importance = pd.DataFrame(
        {
            "feature": spec.feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return prediction, model, spec, importance


def _aggregate_metrics(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, group in predictions.groupby("model_name"):
        metric = _metric_row(
            model_name=model_name,
            test_year=0,
            frame=group,
            predicted_rate=group["predicted_rate_per_min"].to_numpy(),
        )
        metric["test_year"] = "all_walk_forward"
        rows.append(metric)
    aggregate = pd.DataFrame(rows).sort_values("count_poisson_deviance")

    baseline_match = aggregate.loc[
        aggregate["model_name"].eq("fighter_history_baseline"),
        "count_poisson_deviance",
    ]
    if not baseline_match.empty:
        baseline = float(baseline_match.iloc[0])
        aggregate["poisson_improvement_vs_fighter_history"] = (
            baseline - aggregate["count_poisson_deviance"]
        ) / baseline
    return aggregate.reset_index(drop=True)


def walk_forward_sig_attempt_benchmark(
    training_df: pd.DataFrame,
    test_years: Iterable[int] = DEFAULT_TEST_YEARS,
    seed: int = 7,
) -> SigAttemptBenchmarkResult:
    """Evaluate baselines and XGBoost variants on expanding yearly folds."""
    df = prepare_sig_attempt_dataset(training_df)
    years = pd.to_datetime(df["date"]).dt.year
    fold_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    last_bundle: dict[str, object] | None = None

    for fold_index, test_year in enumerate(test_years):
        train_mask = years.lt(int(test_year))
        test_mask = years.eq(int(test_year))
        train = df.loc[train_mask].copy()
        test = df.loc[test_mask].copy()
        if train.empty or test.empty:
            continue

        round_rates, global_rate = _round_rate_lookup(train)
        round_prediction = _round_mean_prediction(test, round_rates, global_rate)
        fighter_prediction = _fighter_history_prediction(test, round_prediction)

        model_predictions: dict[str, np.ndarray] = {
            "round_mean_baseline": round_prediction,
            "fighter_history_baseline": fighter_prediction,
        }

        for include_rfs, model_name in (
            (False, "xgb_context"),
            (True, "xgb_context_rfs"),
        ):
            prediction, model, spec, importance = _fit_predict_xgb(
                train,
                test,
                include_rfs=include_rfs,
                seed=seed + fold_index,
            )
            model_predictions[model_name] = prediction
            importance.insert(0, "test_year", int(test_year))
            importance.insert(0, "model_name", model_name)
            importance_frames.append(importance)
            if include_rfs:
                last_bundle = {
                    "model": model,
                    "matrix_spec": spec,
                    "model_version": SIG_ATTEMPT_MODEL_VERSION,
                    "target": "target_sig_attempt_rate_per_min",
                    "trained_through_year": int(test_year) - 1,
                    "holdout_year": int(test_year),
                }

        keys = [column for column in IDENTITY_COLUMNS if column in test.columns]
        for model_name, predicted_rate in model_predictions.items():
            fold_metric_rows.append(
                _metric_row(model_name, int(test_year), test, predicted_rate)
            )
            prediction_frame = test[
                [
                    *keys,
                    "round",
                    "round_exposure_seconds",
                    "target_sig_attempted",
                    "target_sig_attempt_rate_per_min",
                    "exposure_weight",
                ]
            ].copy()
            prediction_frame["model_name"] = model_name
            prediction_frame["test_year"] = int(test_year)
            prediction_frame["predicted_rate_per_min"] = np.asarray(
                predicted_rate, dtype=float
            )
            prediction_frame["predicted_count_at_actual_exposure"] = (
                prediction_frame["predicted_rate_per_min"]
                * prediction_frame["round_exposure_seconds"]
                / 60.0
            )
            prediction_frames.append(prediction_frame)

    if not prediction_frames or last_bundle is None:
        raise SigAttemptModelError("No walk-forward folds were available")

    fold_metrics = pd.DataFrame(fold_metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    feature_importance = pd.concat(importance_frames, ignore_index=True)
    aggregate_metrics = _aggregate_metrics(fold_metrics, predictions)

    return SigAttemptBenchmarkResult(
        fold_metrics=fold_metrics,
        predictions=predictions,
        feature_importance=feature_importance,
        aggregate_metrics=aggregate_metrics,
        final_bundle=last_bundle,
    )


def save_model_bundle(bundle: Mapping[str, object], path: str) -> None:
    """Persist a shadow-only model bundle."""
    joblib.dump(dict(bundle), path)