"""Run the 2026 baseline+age cohort with one uniform physical age rule.

Research-only age contract for this experiment:
- stored prefight FSR remains immutable
- no trajectory adjustment
- age <= 30: no physical-trait adjustment
- age > 30: -2 FSR points/year after 30
- apply that exact rule to:
    striking_power
    knockdown_resistance
    damage_durability
- clamp effective ratings to the standard 10-90 FSR range

This experiment intentionally does NOT use config/fsr_age_modifiers.yaml. The
three physical traits are controlled here so there is one transparent age rule
and no possibility of mixed or double-applied age functions.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scripts.experimental import run_2026_baseline_age_full_cohort as runner


POWER_AGE_CENTER = 30.0
POWER_AGE_POINTS_PER_YEAR = -2.0
PHYSICAL_AGE_TRAITS = (
    "striking_power",
    "knockdown_resistance",
    "damage_durability",
)


def _physical_age_modifier(age: float | None) -> float:
    if age is None or pd.isna(age):
        return 0.0
    years_over_30 = max(0.0, float(age) - POWER_AGE_CENTER)
    return float(POWER_AGE_POINTS_PER_YEAR * years_over_30)


def _apply_same_physical_age_decay(
    profile: pd.Series,
    age: float | None,
    *,
    enabled: bool,
) -> tuple[pd.Series, float]:
    """Apply the experiment's one intended striking-power age adjustment."""
    effective = profile.copy(deep=True)
    if not enabled:
        return effective, 0.0
    if "striking_power" not in effective.index or pd.isna(effective["striking_power"]):
        raise ValueError("profile missing striking_power required by --age-power")

    modifier = _physical_age_modifier(age)
    effective["striking_power"] = float(
        np.clip(
            float(effective["striking_power"]) + modifier,
            runner.FSR_MIN,
            runner.FSR_MAX,
        )
    )
    return effective, modifier


def _install_uniform_physical_age_layer() -> None:
    """Replace YAML evaluation with the same post-30 rule for KD/durability.

    striking_power is already adjusted by _apply_same_physical_age_decay before
    the simulator is constructed. This evaluator therefore applies the same
    modifier exactly once to knockdown_resistance and damage_durability.
    """

    def apply_uniform_physical_age(
        profile: pd.Series,
        age: float | None,
        *,
        config_path=None,
    ) -> tuple[pd.Series, dict[str, float]]:
        effective = profile.copy(deep=True)
        modifier = _physical_age_modifier(age)
        applied: dict[str, float] = {}
        for trait in ("knockdown_resistance", "damage_durability"):
            if trait not in effective.index or pd.isna(effective[trait]):
                raise ValueError(f"profile missing required physical age trait: {trait}")
            effective[trait] = float(
                np.clip(
                    float(effective[trait]) + modifier,
                    runner.FSR_MIN,
                    runner.FSR_MAX,
                )
            )
            applied[trait] = modifier
        return effective, applied

    def enabled_uniform_traits(*, config_path=None) -> tuple[str, ...]:
        return PHYSICAL_AGE_TRAITS

    # The simulator imports the same fsr_age_modifiers module object, so these
    # in-process replacements affect the simulator call path as well.
    runner.age_modifiers.apply_age_modifiers = apply_uniform_physical_age
    runner.age_modifiers.enabled_calibrated_traits = enabled_uniform_traits


# Backward-compatible alias for the single-fight diagnostic created earlier.
def _install_yaml_power_suppression() -> None:
    _install_uniform_physical_age_layer()


def main() -> None:
    runner._apply_optional_power_age = _apply_same_physical_age_decay
    runner.POWER_AGE_INTERCEPT = 0.0
    runner.POWER_AGE_LINEAR = POWER_AGE_POINTS_PER_YEAR
    runner.POWER_AGE_CENTER = POWER_AGE_CENTER
    _install_uniform_physical_age_layer()

    # Force the parent runner into its isolated power_on output mode while
    # preserving user-supplied arguments such as --paths/--fresh.
    if "--age-power" not in sys.argv:
        sys.argv.append("--age-power")

    runner.main()


if __name__ == "__main__":
    main()
