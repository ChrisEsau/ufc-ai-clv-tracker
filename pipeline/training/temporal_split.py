"""Temporal split utilities for model training.

Notebook source section:
- SECTION 8 — TEMPORAL TRAIN / TEST SPLIT

This module keeps model inputs aligned with resolved feature contracts. It does
not decide which features to use; it receives the already-resolved feature list
from ``feature_selection.py``.

The legacy notebook used one train/test cutoff and then fit calibration on the
same test rows used for evaluation. This module keeps that two-way split for
parity, but also supports a production-safe train/calibration/test split.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Container for temporal train/test matrices and metadata."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_end_date: str
    feature_columns: list[str]
    target_col: str


@dataclass(frozen=True)
class TemporalCalibrationSplit:
    """Container for production-safe train/calibration/test matrices."""

    train_df: pd.DataFrame
    calibration_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    X_calibration: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_calibration: pd.Series
    y_test: pd.Series
    train_end_date: str
    calibration_end_date: str
    feature_columns: list[str]
    target_col: str


def build_temporal_train_test_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_end_date: str = "2022-12-31",
    target_col: str = "target",
    date_col: str = "date",
) -> TemporalSplit:
    """Build chronological train/test datasets using a fixed cutoff date.

    Rows with dates less than or equal to ``train_end_date`` become training rows.
    Rows after the cutoff become test rows. Feature values are coerced to numeric
    and missing values are filled with zero to mirror the existing notebook.
    """
    model_df = _prepare_temporal_model_df(
        df=df,
        feature_columns=feature_columns,
        target_col=target_col,
        date_col=date_col,
    )
    cutoff = pd.to_datetime(train_end_date)

    train_df = model_df[model_df[date_col] <= cutoff].copy()
    test_df = model_df[model_df[date_col] > cutoff].copy()

    _require_non_empty_split(train_df, "train", train_end_date)
    _require_non_empty_split(test_df, "test", train_end_date)

    X_train = _build_numeric_feature_matrix(train_df, feature_columns)
    X_test = _build_numeric_feature_matrix(test_df, feature_columns)
    y_train = train_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)

    return TemporalSplit(
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_end_date=train_end_date,
        feature_columns=feature_columns,
        target_col=target_col,
    )


def build_temporal_train_calibration_test_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_end_date: str,
    calibration_end_date: str,
    target_col: str = "target",
    date_col: str = "date",
) -> TemporalCalibrationSplit:
    """Build chronological train/calibration/test datasets.

    Rows are assigned as:
    - train: date <= train_end_date
    - calibration: train_end_date < date <= calibration_end_date
    - test: date > calibration_end_date

    This avoids fitting the calibrator on the same rows used for final model
    evaluation.
    """
    model_df = _prepare_temporal_model_df(
        df=df,
        feature_columns=feature_columns,
        target_col=target_col,
        date_col=date_col,
    )
    train_cutoff = pd.to_datetime(train_end_date)
    calibration_cutoff = pd.to_datetime(calibration_end_date)

    if calibration_cutoff <= train_cutoff:
        raise ValueError(
            "calibration_end_date must be after train_end_date: "
            f"train_end_date={train_end_date}, calibration_end_date={calibration_end_date}"
        )

    train_df = model_df[model_df[date_col] <= train_cutoff].copy()
    calibration_df = model_df[
        (model_df[date_col] > train_cutoff) & (model_df[date_col] <= calibration_cutoff)
    ].copy()
    test_df = model_df[model_df[date_col] > calibration_cutoff].copy()

    _require_non_empty_split(train_df, "train", train_end_date)
    _require_non_empty_split(calibration_df, "calibration", calibration_end_date)
    _require_non_empty_split(test_df, "test", calibration_end_date)

    X_train = _build_numeric_feature_matrix(train_df, feature_columns)
    X_calibration = _build_numeric_feature_matrix(calibration_df, feature_columns)
    X_test = _build_numeric_feature_matrix(test_df, feature_columns)
    y_train = train_df[target_col].astype(int)
    y_calibration = calibration_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)

    return TemporalCalibrationSplit(
        train_df=train_df,
        calibration_df=calibration_df,
        test_df=test_df,
        X_train=X_train,
        X_calibration=X_calibration,
        X_test=X_test,
        y_train=y_train,
        y_calibration=y_calibration,
        y_test=y_test,
        train_end_date=train_end_date,
        calibration_end_date=calibration_end_date,
        feature_columns=feature_columns,
        target_col=target_col,
    )


def _prepare_temporal_model_df(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    date_col: str,
) -> pd.DataFrame:
    """Validate and normalize dataframe for temporal splitting."""
    required_cols = [date_col, target_col, *feature_columns]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Temporal split input is missing required columns: {missing_cols}")

    model_df = df.copy()
    model_df[date_col] = pd.to_datetime(model_df[date_col], errors="coerce")

    if model_df[date_col].isna().any():
        bad_count = int(model_df[date_col].isna().sum())
        raise ValueError(f"Temporal split found {bad_count} rows with invalid dates")

    model_df = model_df.dropna(subset=[target_col]).copy()
    model_df[target_col] = model_df[target_col].astype(int)
    return model_df.sort_values(date_col).reset_index(drop=True)


def _require_non_empty_split(split_df: pd.DataFrame, split_name: str, cutoff_date: str) -> None:
    """Raise a readable error when a split has no rows."""
    if split_df.empty:
        raise ValueError(f"Temporal split produced empty {split_name} set for cutoff {cutoff_date}")


def _build_numeric_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return numeric model matrix for a resolved feature contract."""
    return df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
