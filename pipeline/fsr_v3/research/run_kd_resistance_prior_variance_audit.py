"""Run KD-resistance audit only where V3 POWER has validated non-neutral state."""
from __future__ import annotations

import pandas as pd

from pipeline.fsr_v3.research import kd_resistance_prior_variance_audit as audit

POWER_VALIDATED_START = pd.Timestamp("2020-01-01")
_ORIGINAL_BUILD = audit.build_observations


def _validated_power_observations():
    frame = _ORIGINAL_BUILD()
    return frame[frame["event_date"] >= POWER_VALIDATED_START].reset_index(drop=True)


def main():
    audit.build_observations = _validated_power_observations
    audit.main()


if __name__ == "__main__":
    main()
