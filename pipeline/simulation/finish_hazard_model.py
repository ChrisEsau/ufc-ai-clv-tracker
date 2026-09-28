"""Leakage-safe competing-risk finish hazard benchmark.

This component predicts one mutually exclusive event for each fight-round:

- no terminal event;
- red KO/TKO;
- red submission;
- blue KO/TKO;
- blue submission.

The first version intentionally uses pre-fight information only. It excludes
within-fight prior-round observations so its historical out-of-fold predictions
can be inserted into a complete pre-fight simulator replay without using actual
rounds from the fight being scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier


FINISH_HAZARD_MODEL_VERSION = "finish_hazard_prefight_v0"
DEFAULT_TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
FINISH_CLASSES = (
    "no_finish",
    "red_ko_tko",
    "red_submission",
    "blue_ko_tko",
    "blue_submission",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(FINISH_CLASSES)}
PROBABILITY_COLUMNS = tuple(f"prob_{name}" for name in FINISH_CLASSES)


class FinishHazardModelError(RuntimeError):
    """Raised when competing-risk data or predictions violate the contract."""


@dataclass(frozen=True)
class FinishHazardBenchmarkResult:
    raw_predictions: pd.DataFrame
    calibrated_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    calibration_schedule: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_manifest: pd.DataFrame


REQUIRED_COLUMNS = (
    "fight_id",
    "fighter_id",
    "opponent_id",
    "corner",
    "date",
    "round",
    "total_rounds",
    "target_finish_time_in_round_seconds",
    "target_sig_attempted",
    "target_sig_landed",
    "target_td_attempted",
    "target_td_landed",
    "target_control_seconds",
    "target_knockdowns",
    "target_submission_attempts",
    "target_fighter_ko_tko_finish",
    "target_opponent_ko_tko_finish",
    "target_fighter_submission_finish",
    "target_opponent_submission_finish",
)

CAREER_OUTCOMES = (
    "ko_win",
    "ko_loss",
    "sub_win",
    "sub_loss",
)
CAREER_STATS = (
    "sig_attempted",
    "sig_landed",
    "td_attempted",
    "td_landed",
    "control_seconds",
    "knockdowns",
    "submission_attempts",
    "exposure_seconds",
)

FINISH_RFS_TOKENS = (
    "head_absorbed",
    "sig_accuracy_allowed",
    "defensive_deterioration",
    "submission_pressure",
    "submission",
    "control",
    "td_",
    "ground",
    "suppression",
)
FINISH_RFS_HORIZONS = ("_exp_", "_ewm_")


@dataclass(frozen=True)
class _MatrixSpec:
    numeric_columns: tuple[str, ...]
    categorical_levels: Mapping[str, tuple[str, ...]]
    feature_columns: tuple[str, ...]


def _require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise FinishHazardModelError(f"{label} is missing required columns: {missing}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(numerator, errors="coerce")
        / pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)


def _add_prefight_career_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add fighter and opponent career states shifted one complete fight."""
    fight = (
        df.groupby(["fight_id", "fighter_id", "date"], dropna=False)
        .agg(
            fight_ko_win=("target_fighter_ko_tko_finish", "max"),
            fight_ko_loss=("target_opponent_ko_tko_finish", "max"),
            fight_sub_win=("target_fighter_submission_finish", "max"),
            fight_sub_loss=("target_opponent_submission_finish", "max"),
            fight_sig_attempted=("target_sig_attempted", "sum"),
            fight_sig_landed=("target_sig_landed", "sum"),
            fight_td_attempted=("target_td_attempted", "sum"),
            fight_td_landed=("target_td_landed", "sum"),
            fight_control_seconds=("target_control_seconds", "sum"),
            fight_knockdowns=("target_knockdowns", "sum"),
            fight_submission_attempts=("target_submission_attempts", "sum"),
            fight_exposure_seconds=("target_finish_time_in_round_seconds", "sum"),
        )
        .reset_index()
        .sort_values(["fighter_id", "date", "fight_id"])
        .reset_index(drop=True)
    )
    group = fight.groupby("fighter_id", sort=False)
    fight["fighter_prior_career_fights"] = group.cumcount().astype(float)

    source_columns = [
        *[f"fight_{name}" for name in CAREER_OUTCOMES],
        *[f"fight_{name}" for name in CAREER_STATS],
    ]
    for column in source_columns:
        values = pd.to_numeric(fight[column], errors="coerce").fillna(0.0)
        fight[column] = values
        prior = group[column].cumsum() - values
        feature = column.replace("fight_", "fighter_prior_career_", 1)
        fight[feature] = prior

    prior_fights = fight["fighter_prior_career_fights"].replace(0, np.nan)
    for outcome in CAREER_OUTCOMES:
        count_column = f"fighter_prior_career_{outcome}"
        fight[f"fighter_prior_career_{outcome}_rate"] = (
            fight[count_column] / prior_fights
        )

    exposure_minutes = fight["fighter_prior_career_exposure_seconds"] / 60.0
    fight["fighter_prior_career_sig_attempt_rate_per_min"] = _safe_divide(
        fight["fighter_prior_career_sig_attempted"], exposure_minutes
    )
    fight["fighter_prior_career_sig_accuracy"] = _safe_divide(
        fight["fighter_prior_career_sig_landed"],
        fight["fighter_prior_career_sig_attempted"],
    )
    fight["fighter_prior_career_td_attempt_rate_per_15"] = _safe_divide(
        fight["fighter_prior_career_td_attempted"], exposure_minutes
    ) * 15.0
    fight["fighter_prior_career_td_accuracy"] = _safe_divide(
        fight["fighter_prior_career_td_landed"],
        fight["fighter_prior_career_td_attempted"],
    )
    fight["fighter_prior_career_control_per_15"] = _safe_divide(
        fight["fighter_prior_career_control_seconds"], exposure_minutes
    ) * 15.0
    fight["fighter_prior_career_kd_per_15"] = _safe_divide(
        fight["fighter_prior_career_knockdowns"], exposure_minutes
    ) * 15.0
    fight["fighter_prior_career_sub_attempts_per_15"] = _safe_divide(
        fight["fighter_prior_career_submission_attempts"], exposure_minutes
    ) * 15.0

    state_columns = [
        column
        for column in fight.columns
        if column.startswith("fighter_prior_career_")
    ]
    state = fight[["fight_id", "fighter_id", *state_columns]].copy()
    out = df.merge(
        state,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="many_to_one",
    )
    opponent_state = state.rename(
        columns={
            "fighter_id": "opponent_id",
            **{
                column: column.replace("fighter_", "opponent_", 1)
                for column in state_columns
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


def select_finish_rfs_columns(df: pd.DataFrame) -> list[str]:
    """Select deterministic finish-relevant pre-fight RFS state features."""
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
        if not any(horizon in column for horizon in FINISH_RFS_HORIZONS):
            continue
        if any(token in column for token in FINISH_RFS_TOKENS):
            selected.append(column)
    return sorted(set(selected))


def prepare_finish_hazard_dataset(training_df: pd.DataFrame) -> pd.DataFrame:
    """Build one pre-fight feature row per fight-round with five risk classes."""
    _require_columns(training_df, REQUIRED_COLUMNS, "Simulator training table")
    df = training_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise FinishHazardModelError("Finish hazard rows require valid dates")
    df["corner"] = df["corner"].astype("string").str.lower()
    if (~df["corner"].isin(["red", "blue"])).any():
        raise FinishHazardModelError("Finish hazard rows contain invalid corners")
    if df.duplicated(["fight_id", "fighter_id", "round"]).any():
        raise FinishHazardModelError("Duplicate fighter-round keys")

    df = _add_prefight_career_history(df)
    red = df.loc[df["corner"].eq("red")].copy()
    blue = df.loc[df["corner"].eq("blue")].copy()
    keys = ["fight_id", "round"]
    if red.duplicated(keys).any() or blue.duplicated(keys).any():
        raise FinishHazardModelError("Duplicate corner rows at fight-round grain")

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
        )
        if column in red.columns
    ]
    target_columns = [
        "target_fighter_ko_tko_finish",
        "target_opponent_ko_tko_finish",
        "target_fighter_submission_finish",
        "target_opponent_submission_finish",
    ]
    career_columns = [
        column
        for column in red.columns
        if column.startswith(("fighter_prior_career_", "opponent_prior_career_"))
    ]
    rfs_columns = select_finish_rfs_columns(red)
    availability_columns = [
        column
        for column in red.columns
        if column.endswith("_state_available")
        and pd.api.types.is_numeric_dtype(red[column])
    ]
    feature_columns = sorted(set(career_columns) | set(rfs_columns) | set(availability_columns))

    out = red[[*identity_columns, "fighter_id", "opponent_id", *target_columns, *feature_columns]].copy()
    out = out.rename(
        columns={
            "fighter_id": "red_fighter_id",
            "opponent_id": "blue_fighter_id",
            **{column: f"red__{column}" for column in feature_columns},
        }
    )

    blue_features = blue[["fight_id", "round", "fighter_id", "opponent_id", *feature_columns]].copy()
    blue_features = blue_features.rename(
        columns={
            "fighter_id": "blue_fighter_id_check",
            "opponent_id": "red_fighter_id_check",
            **{column: f"blue__{column}" for column in feature_columns},
        }
    )
    out = out.merge(blue_features, on=keys, how="inner", validate="one_to_one")
    mismatch = (
        out["red_fighter_id"].astype(str).ne(out["red_fighter_id_check"].astype(str))
        | out["blue_fighter_id"].astype(str).ne(out["blue_fighter_id_check"].astype(str))
    )
    if mismatch.any():
        raise FinishHazardModelError("Red/blue fighter pairing mismatch")
    out = out.drop(columns=["red_fighter_id_check", "blue_fighter_id_check"])

    event_matrix = out[target_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (~event_matrix.isin([0.0, 1.0])).any().any():
        raise FinishHazardModelError("Finish targets must be binary")
    if event_matrix.sum(axis=1).gt(1).any():
        raise FinishHazardModelError("Competing finish targets exceed one event")
    out["finish_class"] = np.select(
        [
            event_matrix["target_fighter_ko_tko_finish"].eq(1.0),
            event_matrix["target_fighter_submission_finish"].eq(1.0),
            event_matrix["target_opponent_ko_tko_finish"].eq(1.0),
            event_matrix["target_opponent_submission_finish"].eq(1.0),
        ],
        [
            "red_ko_tko",
            "red_submission",
            "blue_ko_tko",
            "blue_submission",
        ],
        default="no_finish",
    )
    out["finish_class_index"] = out["finish_class"].map(CLASS_TO_INDEX).astype(int)
    out = out.drop(columns=target_columns)
    return out.sort_values(["date", "fight_id", "round"]).reset_index(drop=True)


def _sanitize(value: object) -> str:
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
            f"cat__{_sanitize(column)}__{_sanitize(category)}"
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
            feature = f"cat__{_sanitize(column)}__{_sanitize(category)}"
            matrix[feature] = values.eq(category).astype(float)
    return matrix.reindex(columns=spec.feature_columns)


def _feature_columns(df: pd.DataFrame, include_rfs: bool) -> tuple[list[str], list[str]]:
    numeric = [
        column
        for column in ("round", "total_rounds", "title_fight")
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]
    numeric.extend(
        column
        for column in df.columns
        if column.startswith(("red__fighter_prior_career_", "red__opponent_prior_career_", "blue__fighter_prior_career_", "blue__opponent_prior_career_"))
        and pd.api.types.is_numeric_dtype(df[column])
    )
    if include_rfs:
        numeric.extend(
            column
            for column in df.columns
            if (
                column.startswith(("red__fighter_rfs_", "red__opponent_rfs_", "blue__fighter_rfs_", "blue__opponent_rfs_"))
                or column.endswith("_state_available")
            )
            and pd.api.types.is_numeric_dtype(df[column])
        )
    categorical = [column for column in ("division",) if column in df.columns]
    return sorted(set(numeric)), categorical


def _new_model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(FINISH_CLASSES),
        eval_metric="mlogloss",
        n_estimators=360,
        learning_rate=0.035,
        max_depth=3,
        min_child_weight=25.0,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.10,
        reg_lambda=10.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
    )


def _baseline_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    global_counts = train["finish_class"].value_counts().reindex(FINISH_CLASSES, fill_value=0)
    global_probs = (global_counts.to_numpy(dtype=float) + 1.0)
    global_probs /= global_probs.sum()
    round_lookup: dict[int, np.ndarray] = {}
    for round_number, group in train.groupby("round"):
        counts = group["finish_class"].value_counts().reindex(FINISH_CLASSES, fill_value=0)
        probs = counts.to_numpy(dtype=float) + 2.0 * global_probs
        probs /= probs.sum()
        round_lookup[int(round_number)] = probs
    return np.vstack(
        [round_lookup.get(int(round_number), global_probs) for round_number in test["round"]]
    )


def _prediction_frame(
    test: pd.DataFrame,
    probabilities: np.ndarray,
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
    out["model_name"] = model_name
    out["test_year"] = int(test_year)
    for index, column in enumerate(PROBABILITY_COLUMNS):
        out[column] = probabilities[:, index]
    return out


def _metrics(frame: pd.DataFrame, probability_columns: Sequence[str]) -> dict[str, float]:
    actual = frame["finish_class_index"].to_numpy(dtype=int)
    probabilities = np.clip(frame[list(probability_columns)].to_numpy(dtype=float), 1e-9, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    one_hot = np.eye(len(FINISH_CLASSES), dtype=float)[actual]
    result = {
        "rows": int(len(frame)),
        "fights": int(frame["fight_id"].nunique()),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == actual)),
        "log_loss": float(log_loss(actual, probabilities, labels=list(range(len(FINISH_CLASSES))))),
        "multiclass_brier": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
    }
    for index, name in enumerate(FINISH_CLASSES):
        result[f"actual_rate_{name}"] = float(one_hot[:, index].mean())
        result[f"predicted_rate_{name}"] = float(probabilities[:, index].mean())
        result[f"brier_{name}"] = float(np.mean(np.square(probabilities[:, index] - one_hot[:, index])))
    return result


def _sequential_calibration(
    predictions: pd.DataFrame,
    minimum_prior_rows: int = 1_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibrated_frames: list[pd.DataFrame] = []
    schedule_rows: list[dict[str, object]] = []
    for model_name, model_frame in predictions.groupby("model_name", sort=False):
        for year in sorted(model_frame["test_year"].unique()):
            current = model_frame.loc[model_frame["test_year"].eq(year)].copy()
            prior = model_frame.loc[model_frame["test_year"].lt(year)].copy()
            if len(prior) >= minimum_prior_rows:
                actual_counts = (
                    prior["finish_class"]
                    .value_counts()
                    .reindex(FINISH_CLASSES, fill_value=0)
                    .to_numpy(dtype=float)
                )
                predicted_counts = prior[list(PROBABILITY_COLUMNS)].sum().to_numpy(dtype=float)
                factors = np.clip(
                    (actual_counts + 2.0) / (predicted_counts + 2.0),
                    0.25,
                    4.0,
                )
                source = "prior_walk_forward_years"
            else:
                factors = np.ones(len(FINISH_CLASSES), dtype=float)
                source = "cold_start_identity"
            raw = current[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
            adjusted = raw * factors.reshape(1, -1)
            adjusted /= adjusted.sum(axis=1, keepdims=True)
            for index, column in enumerate(PROBABILITY_COLUMNS):
                current[f"calibrated_{column}"] = adjusted[:, index]
            calibrated_frames.append(current)
            row: dict[str, object] = {
                "model_name": model_name,
                "test_year": int(year),
                "prior_rows": int(len(prior)),
                "calibration_source": source,
            }
            for index, name in enumerate(FINISH_CLASSES):
                row[f"factor_{name}"] = float(factors[index])
            schedule_rows.append(row)
    return pd.concat(calibrated_frames, ignore_index=True), pd.DataFrame(schedule_rows)


def walk_forward_finish_hazard_benchmark(
    training_df: pd.DataFrame,
    test_years: Sequence[int] = DEFAULT_TEST_YEARS,
    seed: int = 7,
) -> FinishHazardBenchmarkResult:
    """Run expanding-year competing-risk benchmarks and sequential calibration."""
    dataset = prepare_finish_hazard_dataset(training_df)
    raw_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for fold_index, test_year in enumerate(test_years):
        train = dataset.loc[dataset["date"].dt.year.lt(int(test_year))].copy()
        test = dataset.loc[dataset["date"].dt.year.eq(int(test_year))].copy()
        if train.empty or test.empty:
            raise FinishHazardModelError(
                f"Empty train/test split for holdout year {test_year}"
            )

        baseline_probs = _baseline_probabilities(train, test)
        raw_frames.append(
            _prediction_frame(
                test,
                baseline_probs,
                "round_frequency_baseline",
                int(test_year),
            )
        )

        for model_name, include_rfs in (
            ("xgb_prefight_context", False),
            ("xgb_prefight_context_rfs", True),
        ):
            numeric, categorical = _feature_columns(train, include_rfs=include_rfs)
            spec = _fit_matrix_spec(train, numeric, categorical)
            x_train = _transform_matrix(train, spec)
            x_test = _transform_matrix(test, spec)
            model = _new_model(seed + fold_index * 101 + (17 if include_rfs else 0))
            model.fit(x_train, train["finish_class_index"].to_numpy(dtype=int))
            probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
            raw_frames.append(
                _prediction_frame(test, probabilities, model_name, int(test_year))
            )
            importance = pd.DataFrame(
                {
                    "feature": spec.feature_columns,
                    "importance": model.feature_importances_,
                }
            )
            importance["model_name"] = model_name
            importance["test_year"] = int(test_year)
            importance_frames.append(importance)
            for feature in spec.feature_columns:
                manifest_rows.append(
                    {
                        "model_name": model_name,
                        "test_year": int(test_year),
                        "feature": feature,
                    }
                )

    raw_predictions = pd.concat(raw_frames, ignore_index=True)
    calibrated_predictions, schedule = _sequential_calibration(raw_predictions)

    metric_rows: list[dict[str, object]] = []
    calibrated_columns = [f"calibrated_{column}" for column in PROBABILITY_COLUMNS]
    for (model_name, year), frame in calibrated_predictions.groupby(
        ["model_name", "test_year"], sort=True
    ):
        raw_metric = _metrics(frame, PROBABILITY_COLUMNS)
        raw_metric.update(
            {
                "model_name": model_name,
                "test_year": int(year),
                "calibration": "raw",
            }
        )
        metric_rows.append(raw_metric)
        calibrated_metric = _metrics(frame, calibrated_columns)
        calibrated_metric.update(
            {
                "model_name": model_name,
                "test_year": int(year),
                "calibration": "sequential_class_calibrated",
            }
        )
        metric_rows.append(calibrated_metric)
    fold_metrics = pd.DataFrame(metric_rows)

    aggregate_rows: list[dict[str, object]] = []
    for model_name, frame in calibrated_predictions.groupby("model_name", sort=True):
        for calibration, columns in (
            ("raw", PROBABILITY_COLUMNS),
            ("sequential_class_calibrated", calibrated_columns),
        ):
            metric = _metrics(frame, columns)
            metric.update(
                {
                    "model_name": model_name,
                    "calibration": calibration,
                    "test_year": "all",
                }
            )
            aggregate_rows.append(metric)
    aggregate_metrics = pd.DataFrame(aggregate_rows)

    return FinishHazardBenchmarkResult(
        raw_predictions=raw_predictions,
        calibrated_predictions=calibrated_predictions,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate_metrics,
        calibration_schedule=schedule,
        feature_importance=pd.concat(importance_frames, ignore_index=True),
        feature_manifest=pd.DataFrame(manifest_rows),
    )
