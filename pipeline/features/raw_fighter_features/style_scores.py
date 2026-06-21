"""Continuous style-score fighter-state enrichment.

This module applies the locked k=5 style-score weights produced by the style
matchup research workflow. It is intentionally implemented as a history-level
enrichment because the score formula standardizes input columns across the
fighter-state history before applying the signed feature weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_WEIGHTS_PATH = Path("data/research/style_matchups/style_score_weights.yaml")

OUTPUT_COLUMNS = [
    "style_control_wrestler_score",
    "style_ko_finisher_score",
    "style_submission_grappler_score",
    "style_decision_technician_score",
    "style_all_round_finisher_score",
    "style_primary_score",
    "style_score_spread",
]


def load_style_score_weights(weights_path: str | Path = DEFAULT_WEIGHTS_PATH) -> dict[str, dict[str, float]]:
    """Load style score weights keyed by style name."""

    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Style score weights not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_weights = payload.get("style_score_weights") or {}
    if not raw_weights:
        raise ValueError(f"No style_score_weights block found in: {path}")

    weights: dict[str, dict[str, float]] = {}
    for style_name, spec in raw_weights.items():
        feature_weights = spec.get("weights") or {}
        weights[str(style_name)] = {str(feature): float(weight) for feature, weight in feature_weights.items()}
    return weights


def zscore(series: pd.Series) -> pd.Series:
    """Return population z-score with safe zero-variance handling."""

    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return ((values - values.mean()) / std).fillna(0.0)


def enrich_history(history_df: pd.DataFrame, *, weights_path: str | Path = DEFAULT_WEIGHTS_PATH) -> pd.DataFrame:
    """Add continuous style scores to fighter-state history."""

    if history_df.empty:
        return history_df.copy()

    weights = load_style_score_weights(weights_path)
    required_features = sorted({feature for feature_weights in weights.values() for feature in feature_weights})
    missing = [feature for feature in required_features if feature not in history_df.columns]
    if missing:
        raise ValueError(f"Cannot build style scores; missing fighter-state columns: {missing}")

    out = history_df.copy()
    zscores = {feature: zscore(out[feature]) for feature in required_features}
    score_columns: list[str] = []

    for style_name, feature_weights in weights.items():
        score_column = f"style_{style_name}_score"
        score = pd.Series(0.0, index=out.index)
        for feature, weight in feature_weights.items():
            score = score + float(weight) * zscores[feature]
        out[score_column] = score
        score_columns.append(score_column)

    out["style_primary_score"] = out[score_columns].max(axis=1)
    out["style_score_spread"] = out[score_columns].max(axis=1) - out[score_columns].min(axis=1)
    return out


# Plugin contract stubs. The history builder does not call this module as a
# normal state updater, but these make the module shape consistent with the raw
# fighter feature plugin directory.
def initial_state() -> dict[str, Any]:
    return {}


def calculate(fighter_history: pd.DataFrame, fight_row: pd.Series, context: dict | None = None) -> dict[str, float]:
    del fighter_history, fight_row, context
    return {column: 0.0 for column in OUTPUT_COLUMNS}
