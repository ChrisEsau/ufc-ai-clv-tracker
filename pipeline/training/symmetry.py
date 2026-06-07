"""Symmetry augmentation utilities for matchup models.

Notebook source section:
- SECTION 7 — SYMMETRY AUGMENTATION

The moneyline model uses red-minus-blue matchup features. Symmetry augmentation
adds a flipped blue-minus-red copy of every row and inverts the target, reducing
red-corner bias and forcing the model to learn matchup edge instead of corner.
"""

from __future__ import annotations

import pandas as pd


def apply_symmetry_augmentation(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "target",
    date_col: str = "date",
) -> pd.DataFrame:
    """Return original rows plus flipped matchup rows.

    For flipped rows, all selected matchup feature values are multiplied by -1
    and the binary target is inverted.
    """
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns for symmetry augmentation: {missing_features}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    base_df = df.copy()
    flipped_df = base_df.copy()

    for col in feature_columns:
        flipped_df[col] = -pd.to_numeric(flipped_df[col], errors="coerce").fillna(0)

    flipped_df[target_col] = 1 - flipped_df[target_col].astype(int)

    model_df = pd.concat([base_df, flipped_df], axis=0, ignore_index=True)

    if date_col in model_df.columns:
        model_df[date_col] = pd.to_datetime(model_df[date_col], errors="coerce")
        model_df = model_df.sort_values(date_col).reset_index(drop=True)

    return model_df
