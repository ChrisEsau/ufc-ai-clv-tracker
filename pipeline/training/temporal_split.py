"""Temporal train/test split utilities for model training.

Notebook source section:
- SECTION 8 — TEMPORAL TRAIN / TEST SPLIT

This module keeps model inputs aligned with resolved feature contracts. It does
not decide which features to use; it receives the already-resolved feature list
from ``feature_selection.py``.
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
    required_cols = [date_col, target_col, *feature_columns]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Temporal split input is missing required columns: {missing_cols}")

    model_df = df.copy()
    model_df[date_col] = pd.to_datetime(model_df[date_col], errors="coerce")
    cutoff = pd.to_datetime(train_end_date)

    if model_df[date_col].isna().any():
        bad_count = int(model_df[date_col].isna().sum())
        raise ValueError(f"Temporal split found {bad_count} rows with invalid dates")

    model_df = model_df.dropna(subset=[target_col]).copy()
    model_df[target_col] = model_df[target_col].astype(int)
    model_df = model_df.sort_values(date_col).reset_index(drop=True)

    train_df = model_df[model_df[date_col] <= cutoff].copy()
    test_df = model_df[model_df[date_col] > cutoff].copy()

    if train_df.empty:
        raise ValueError(f"Temporal split produced empty train set for cutoff {train_end_date}")
    if test_df.empty:
        raise ValueError(f"Temporal split produced empty test set for cutoff {train_end_date}")

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


def _build_numeric_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return numeric model matrix for a resolved feature contract."""
    return df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
