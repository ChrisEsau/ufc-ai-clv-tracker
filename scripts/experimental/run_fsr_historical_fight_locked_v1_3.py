"""Run corrected-activity FSR V1.2 with a provisional finish-hazard scale.

Shadow/research only.

This checkpoint does NOT change the Monte Carlo engine, FSR equations, rating
centering, activity-rate conversion, transition mapping, cardio bridge, or
scoring. It keeps the corrected V1.2 activity adapter and changes only the
existing finish-calibration values to a provisional scale located by the
Topuria/Gaethje hazard grid.

These values are NOT frozen and are NOT a one-fight fit. They are only a
reasonable scale point for detailed-path inspection before broader historical
validation.
"""

from __future__ import annotations

from dataclasses import replace

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1_2 as v1_2


PROVISIONAL_LANDED_KO_HAZARD = 0.0025
PROVISIONAL_KNOCKDOWN_BONUS_HAZARD = 0.080
PROVISIONAL_SUBMISSION_HAZARD = 0.36

# Capture the original calibration function before installing any overrides.
_original_finish_calibration = base.finish_calibration


def finish_calibration(_candidate):
    """Return the provisional corrected-activity finish calibration."""

    selected_candidate = base.Candidate(
        landed_ko_hazard=PROVISIONAL_LANDED_KO_HAZARD,
        knockdown_bonus_hazard=PROVISIONAL_KNOCKDOWN_BONUS_HAZARD,
    )

    calibration = _original_finish_calibration(selected_candidate)

    return replace(
        calibration,
        submission=replace(
            calibration.submission,
            base_probability_per_attempt=PROVISIONAL_SUBMISSION_HAZARD,
        ),
    )


def install_overrides() -> None:
    """Install V1.2 adapter plus provisional finish calibration only."""

    v1_2.install_overrides()
    base.finish_calibration = finish_calibration


if __name__ == "__main__":
    install_overrides()
    base.main()
