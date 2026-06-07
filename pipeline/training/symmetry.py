"""Symmetry augmentation utilities for matchup models.

Notebook source section:
- SECTION 7 — SYMMETRY AUGMENTATION

The base moneyline model uses red-minus-blue matchup features. Symmetry
augmentation adds a flipped blue-minus-red copy of every row and inverts the
target, reducing red-corner bias and forcing the model to learn matchup edge
instead of corner.

Future model contracts may include both directional and non-directional features.
Directional features should be negated when a row is flipped. Preserved features
should stay unchanged.
"""

from __future__ import annotations

import pandas as pd


def apply_symmetry_augmentation(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "target",
    date_col: str = "date",
    flip_feature_columns: list[str] | None = None,
    preserve_feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return original rows plus flipped matchup rows.

    Parameters
    ----------
    df:
        Input dataframe containing model features and target.
    feature_columns:
        Full resolved feature contract for the model.
    target_col:
        Binary target where 1 means the original red-side perspective won.
    date_col:
        Date column used to preserve chronological ordering.
    flip_feature_columns:
        Directional features to multiply by -1 in flipped rows. If omitted,
        defaults to all ``feature_columns`` for backward compatibility with the
        original 124-feature moneyline model.
    preserve_feature_columns:
        Non-directional features that should remain unchanged in flipped rows.
    """
    _validate_symmetry_inputs(
        df=df,
        feature_columns=feature_columns,
        target_col=target_col,
        flip_feature_columns=flip_feature_columns,
        preserve_feature_columns=preserve_feature_columns,
    )

    base_df = df.copy()
    flipped_df = base_df.copy()

    columns_to_flip = flip_feature_columns if flip_feature_columns is not None else feature_columns

    for col in columns_to_flip:
        flipped_df[col] = -pd.to_numeric(flipped_df[col], errors="coerce").fillna(0)

    # Preserved feature columns intentionally remain unchanged.
    flipped_df[target_col] = 1 - flipped_df[target_col].astype(int)

    model_df = pd.concat([base_df, flipped_df], axis=0, ignore_index=True)

    if date_col in model_df.columns:
        model_df[date_col] = pd.to_datetime(model_df[date_col], errors="coerce")
        model_df = model_df.sort_values(date_col).reset_index(drop=True)

    return model_df


def infer_directional_features(feature_columns: list[str]) -> list[str]:
    """Infer directional features from common matchup naming conventions.

    This helper is useful for future explicit contracts. It is intentionally
    conservative: it flips columns ending in ``_diff`` or known directional edge
    names. Contracts can still override this with explicit flip/preserve lists.
    """
    directional_keywords = (
        "edge",
        "adv",
        "mismatch",
        "combo",
        "volatility",
    )

    return [
        col
        for col in feature_columns
        if col.endswith("_diff") or any(keyword in col for keyword in directional_keywords)
    ]


def infer_preserved_features(feature_columns: list[str], flip_feature_columns: list[str]) -> list[str]:
    """Return feature columns not included in the flip list."""
    flip_set = set(flip_feature_columns)
    return [col for col in feature_columns if col not in flip_set]


def _validate_symmetry_inputs(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    flip_feature_columns: list[str] | None,
    preserve_feature_columns: list[str] | None,
) -> None:
    """Validate feature and target columns before augmentation."""
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns for symmetry augmentation: {missing_features}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    flip_set = set(flip_feature_columns or feature_columns)
    preserve_set = set(preserve_feature_columns or [])

    unknown_flip = sorted(col for col in flip_set if col not in feature_columns)
    if unknown_flip:
        raise ValueError(f"flip_feature_columns contains columns outside feature contract: {unknown_flip}")

    unknown_preserve = sorted(col for col in preserve_set if col not in feature_columns)
    if unknown_preserve:
        raise ValueError(
            f"preserve_feature_columns contains columns outside feature contract: {unknown_preserve}"
        )

    overlap = sorted(flip_set.intersection(preserve_set))
    if overlap:
        raise ValueError(f"Columns cannot be both flipped and preserved: {overlap}")
