"""V3 activity-dependent fatigue calibration for RFS Monte Carlo V2.

This checkpoint extends the V2 dynamic-adversity calibration without changing:

- FSR fighter cards
- finishing-power / chin behavior
- damage and acute-stress adversity costs
- finish calibration
- transition-effect calibration

V3 changes only fatigue workload construction.  V1/V2 assigned every fighter
the same 0.05 workload in every 30-second segment and set all realized-activity
workload costs to zero.  That made fatigue behave primarily as a round clock.

The V3 values below are deliberately conservative starting values.  Passive
phase workload is reduced so activity costs do not simply add more fatigue on
top of the old clock.  Fighter-specific fatigue resistance and recovery remain
neutral in the current FSR adapter; this checkpoint isolates workload first.

Shadow/research only.  These are not production-calibrated constants.
"""

from __future__ import annotations

from dataclasses import replace

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    ActivityWorkloadCalibration,
    DynamicStateCalibration,
    PhaseWorkloadCalibration,
)

# Preserve/re-export the complete V2 adversity checkpoint and frozen V1
# finish/transition contracts used by the historical benchmark runner.
from scripts.calibrate_rfs_mc_v2_dynamic_adversity_v2 import (
    Candidate,
    MIN_PRIOR_FIGHTS,
    V1_KNOCKDOWN_BONUS_HAZARD,
    V1_LANDED_KO_HAZARD,
    build_matchup_inputs,
    finish_calibration,
    phase_effect_calibration,
    state_calibration as v2_state_calibration,
    zero_transition_effect_calibration,
)


# ---------------------------------------------------------------------------
# V3 fatigue workload calibration
# ---------------------------------------------------------------------------

def state_calibration(
    candidate: Candidate,
) -> DynamicStateCalibration:
    """Preserve V2 adversity while making fatigue activity-dependent."""

    base = v2_state_calibration(candidate)

    # Passive cost of spending one 30-second segment in each phase/role.
    # These values are intentionally lower than the V1/V2 universal 0.05
    # because realized activity now contributes additional workload.
    phase_workload = PhaseWorkloadCalibration(
        distance=0.032,
        clinch_owner=0.034,
        clinch_defender=0.038,
        ground_owner=0.036,
        ground_defender=0.042,
    )

    # Additional cost from work actually performed during the segment.
    # Units are normalized raw workload before fighter-specific resistance.
    activity_workload = ActivityWorkloadCalibration(
        strike_attempt=0.0025,
        control_second=0.0008,
        submission_attempt=0.0120,
        position_advancement=0.0080,
        escape_attempt=0.0060,
        reversal_attempt=0.0100,
        scramble_attempt=0.0080,
    )

    return replace(
        base,
        phase_workload=phase_workload,
        activity_workload=activity_workload,
    )
