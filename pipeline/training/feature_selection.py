"""Moneyline feature selection utilities.

Notebook source sections:
- SECTION 5 — LOAD FEATURE REGISTRY
- SECTION 6 — SYMMETRY-SAFE FEATURE FILTER

Responsibilities:
- Select all differential matchup features.
- Add approved engineered features from the registry.
- Block unsafe raw red/blue fighter-specific columns.
- Return the model feature contract used by moneyline training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

UNSAFE_PREFIXES = ("r_pre_", "b_pre_", "R_", "B_", "r_", "b_")


def load_registered_engineered_features(registry_path: str | Path) -> list[str]:
    """Load engineered feature names from CSV or YAML registry files."""
    registry_path = Path(registry_path)

    if registry_path.suffix.lower() in {".yaml", ".yml"}:
        with registry_path.open("r", encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}

        if isinstance(registry, dict):
            for key in ("engineered_features", "features", "moneyline_features"):
                values = registry.get(key)
                if isinstance(values, list):
                    return [str(v) for v in values]

            # Fallback for richer feature registries that store records.
            values = []
            for item in registry.values():
                if isinstance(item, list):
                    for entry in item:
                        if isinstance(entry, dict) and "feature" in entry:
                            values.append(str(entry["feature"]))
            if values:
                return values

        return []

    registry_df = pd.read_csv(registry_path)
    if "feature" not in registry_df.columns:
        raise ValueError(f"Feature registry is missing required 'feature' column: {registry_path}")

    return registry_df["feature"].dropna().astype(str).tolist()


def select_moneyline_features(
    df: pd.DataFrame,
    registered_features: list[str],
) -> list[str]:
    """Return symmetry-safe moneyline feature columns.

    The rule mirrors the training notebook:
    - include all columns ending in ``_diff``
    - include explicitly registered engineered features
    - reject raw red/blue fighter-specific columns
    """
    safe_cols: list[str] = []

    for col in df.columns:
        if col.endswith("_diff"):
            safe_cols.append(col)
        elif col in registered_features:
            safe_cols.append(col)

    safe_cols = list(dict.fromkeys(safe_cols))

    unsafe_cols = [col for col in safe_cols if col.startswith(UNSAFE_PREFIXES)]
    if unsafe_cols:
        raise ValueError(f"Unsafe raw red/blue fighter columns detected: {unsafe_cols}")

    return safe_cols
