"""Damage Reservoir V1 shadow variant with provisional ground-exit calibration.

This module intentionally leaves ``fsr_static_mc_damage_v1.py`` and the frozen
V0 simulator unchanged.  It applies only the selected ground-persistence shadow
candidate from the 300-bout sensitivity sweep:

    GROUND_EXIT_BASE_30S = 0.17

The candidate improved ground residence/control and several matchup-ranking
metrics without the broader degradation seen at 0.14/0.11.  It remains a shadow
research setting, not a production lock.
"""
from __future__ import annotations

from math import exp

import numpy as np

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_v0 as base


GROUND_EXIT_BASE_30S_SHADOW = 0.17
GROUND_EXIT_BASE_SHADOW = base._rescale_interval_prob(
    GROUND_EXIT_BASE_30S_SHADOW,
    base.CALIBRATION_INTERVAL_SECONDS,
    base.SEGMENT_SECONDS,
)


class StaticFSRMCDamageV1Ground017(damage.StaticFSRMCDamageV1):
    """Damage V1 with only the provisional 0.17 ground-exit candidate changed."""

    def _ground_exit_hazard(self, controller: int) -> float:
        bottom = self._other(controller)
        escape_edge = (
            base._value(self.fighters[bottom], "control_resistance")
            - base._value(self.fighters[controller], "control_imposition")
        ) / base.RATING_SCALE
        reversal_edge = (
            base._value(self.fighters[bottom], "reversal_ability")
            - base._value(self.fighters[controller], "control_imposition")
        ) / base.RATING_SCALE
        modifier = exp(
            float(np.clip(0.60 * escape_edge + 0.40 * reversal_edge, -1.5, 1.5))
        )
        return base._prob(GROUND_EXIT_BASE_SHADOW * modifier, high=0.90)
