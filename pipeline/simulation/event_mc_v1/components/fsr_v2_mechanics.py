"""Literal, uncalibrated structural transforms from validated FSR V2 traits."""

from math import exp, log


# Leakage-safe TD-completion validation:
# exact-scale attacker-age adjustment, centered at age 30.
TAKEDOWN_ATTACKER_AGE_CENTER_YEARS = 30.0
TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR = -0.018388


def effective_rate(tendency: float, suppression: float) -> float:
    return max(0.0, tendency - suppression)


def matchup_probability(
    baseline: float,
    offense: float,
    defense: float,
    logit_offset: float = 0.0,
) -> float:
    baseline = min(max(baseline, 1e-6), 1.0 - 1e-6)
    value = log(baseline / (1.0 - baseline)) + offense - defense + logit_offset
    return 1.0 / (1.0 + exp(-max(-40.0, min(40.0, value))))


def escape_rate(escape_offense: float, escape_defense: float, baseline_mean: float) -> float:
    # Ratings are matchup effects; offense shortens and defense lengthens wait.
    mean = baseline_mean * exp(-escape_offense + escape_defense)
    return 1.0 / max(mean, 1e-9)
