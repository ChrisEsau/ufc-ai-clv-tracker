"""Run the 2026 baseline+age cohort with striking power using the exact same
post-30 age reduction as knockdown resistance and damage durability.

This is a thin research wrapper around run_2026_baseline_age_full_cohort.py.
It leaves stored FSR immutable and changes only the temporary fight-night
striking_power profile used when --age-power is active.

Rule:
- age <= 30: 0 FSR adjustment
- age > 30: -2 FSR points per year after 30
- clamp effective striking_power to the standard 10-90 FSR range
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scripts.experimental import run_2026_baseline_age_full_cohort as runner


POWER_AGE_CENTER = 30.0
POWER_AGE_POINTS_PER_YEAR = -2.0


def _apply_same_physical_age_decay(
    profile: pd.Series,
    age: float | None,
    *,
    enabled: bool,
) -> tuple[pd.Series, float]:
    effective = profile.copy(deep=True)
    if not enabled or age is None or pd.isna(age):
        return effective, 0.0
    if "striking_power" not in effective.index or pd.isna(effective["striking_power"]):
        raise ValueError("profile missing striking_power required by --age-power")

    years_over_30 = max(0.0, float(age) - POWER_AGE_CENTER)
    modifier = POWER_AGE_POINTS_PER_YEAR * years_over_30
    effective["striking_power"] = float(
        np.clip(
            float(effective["striking_power"]) + modifier,
            runner.FSR_MIN,
            runner.FSR_MAX,
        )
    )
    return effective, float(modifier)


def main() -> None:
    runner._apply_optional_power_age = _apply_same_physical_age_decay
    runner.POWER_AGE_INTERCEPT = 0.0
    runner.POWER_AGE_LINEAR = POWER_AGE_POINTS_PER_YEAR
    runner.POWER_AGE_CENTER = POWER_AGE_CENTER

    # Force the parent runner into its isolated power_on output mode while
    # preserving all user-supplied arguments such as --paths/--fresh.
    if "--age-power" not in sys.argv:
        sys.argv.append("--age-power")

    runner.main()


if __name__ == "__main__":
    main()
