"""V2-native terminal submission conversion from validated Event2 replay."""

from __future__ import annotations

from math import exp, log
import numpy as np


def submission_conversion_probability(baseline: float, conversion_offset: float) -> float:
    """Final integrated Event2 semantic: logistic(logit(baseline) + offset)."""
    p = float(np.clip(baseline, 1e-8, 1.0 - 1e-8))
    z = log(p / (1.0 - p)) + float(conversion_offset)
    return 1.0 / (1.0 + exp(-float(np.clip(z, -40.0, 40.0))))


def stage11c_matchup_probability(baseline: float, attacker_offense: float, defender_defense: float) -> float:
    """Auditable Stage-11C research equation; superseded by integrated replay."""
    p = float(np.clip(baseline, 1e-8, 1.0 - 1e-8))
    z = log(p / (1.0 - p)) + float(attacker_offense) - float(defender_defense)
    return 1.0 / (1.0 + exp(-float(np.clip(z, -40.0, 40.0))))


def resolve_submission(baseline: float, conversion_offset: float, rng: np.random.Generator) -> tuple[float, bool]:
    probability = submission_conversion_probability(baseline, conversion_offset)
    return probability, bool(rng.random() < probability)
