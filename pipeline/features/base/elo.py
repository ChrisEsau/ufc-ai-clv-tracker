"""Elo settings and helper functions for rolling UFC base features.

Notebook source section:
- SETTINGS + HELPERS

Migration status:
- Migrated constants and helper functions from UFC_rolling_dataset_V4_refactored.ipynb.
"""

START_ELO = 1500
K_FACTOR = 32
RECENT_N = 3


def safe_div(a: float, b: float) -> float:
    """Return a / b, using 0 when the denominator is 0.

    This preserves the notebook helper behavior exactly.
    """
    return a / b if b != 0 else 0


def expected_score(elo_a: float, elo_b: float) -> float:
    """Calculate the expected Elo score for fighter A against fighter B."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
