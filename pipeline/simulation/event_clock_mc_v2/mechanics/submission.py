"""V2-native terminal submission conversion with validated matchup signal."""

from __future__ import annotations

from math import exp, log
import numpy as np

# Chronological historical fit on attempt-positive fighter-fights.  The raw
# coefficient is applied to inherited prefight submission_offense minus the
# opponent's inherited prefight submission_defense.  It was estimated after
# controlling for attempt count and the existing conversion baseline.
SUBMISSION_OFFENSE_DEFENSE_BETA = 4.9075089640448715


def submission_conversion_probability(
    baseline: float,
    conversion_offset: float,
    attacker_offense: float = 0.0,
    defender_defense: float = 0.0,
) -> float:
    """Return per-attempt conversion with the validated offense-defense edge."""
    p = float(np.clip(baseline, 1e-8, 1.0 - 1e-8))
    edge = float(attacker_offense) - float(defender_defense)
    z = (
        log(p / (1.0 - p))
        + float(conversion_offset)
        + SUBMISSION_OFFENSE_DEFENSE_BETA * edge
    )
    return 1.0 / (1.0 + exp(-float(np.clip(z, -40.0, 40.0))))


def stage11c_matchup_probability(
    baseline: float, attacker_offense: float, defender_defense: float
) -> float:
    """Auditable original Stage-11C equation retained for comparison."""
    p = float(np.clip(baseline, 1e-8, 1.0 - 1e-8))
    z = log(p / (1.0 - p)) + float(attacker_offense) - float(defender_defense)
    return 1.0 / (1.0 + exp(-float(np.clip(z, -40.0, 40.0))))


def resolve_submission(
    baseline: float,
    conversion_offset: float,
    attacker_offense: float,
    defender_defense: float,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    probability = submission_conversion_probability(
        baseline,
        conversion_offset,
        attacker_offense,
        defender_defense,
    )
    return probability, bool(rng.random() < probability)
