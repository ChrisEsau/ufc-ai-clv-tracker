"""Unified phase-activity dispatch for RFS Monte Carlo V2.

Each segment has one authoritative shared phase. This module routes activity
generation to exactly one phase-specific engine:

- DISTANCE -> distance activity
- CLINCH   -> clinch activity
- GROUND   -> ground owner/defender activity

The dispatcher does not select or change the fight phase. Phase transitions
remain the responsibility of the shared transition engine.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchSegmentActivity,
    generate_clinch_segment_activity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceSegmentActivity,
    generate_distance_segment_activity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
    generate_ground_segment_activity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    FighterPhaseParameters,
)


PhaseSegmentActivity: TypeAlias = (
    DistanceSegmentActivity
    | ClinchSegmentActivity
    | GroundSegmentActivity
)


def generate_phase_segment_activity(
    state: SharedFightState,
    red: FighterPhaseParameters,
    blue: FighterPhaseParameters,
    rng: np.random.Generator,
) -> PhaseSegmentActivity:
    """Generate legal activity for the current authoritative phase.

    Static fighter parameters are supplied here. Later, dynamic-state logic
    will create temporary effective parameter bundles before calling this
    dispatcher.
    """

    if state.phase is FightPhase.DISTANCE:
        return generate_distance_segment_activity(
            state,
            red.distance,
            blue.distance,
            rng,
        )

    if state.phase is FightPhase.CLINCH:
        if state.phase_owner is None:
            raise ValueError(
                "clinch activity requires a phase owner"
            )

        return generate_clinch_segment_activity(
            state,
            red.clinch,
            blue.clinch,
            rng,
        )

    if state.phase is FightPhase.GROUND:
        if state.phase_owner is FighterSide.RED:
            owner_parameters = red.ground_owner
            defender_parameters = blue.ground_defender

        elif state.phase_owner is FighterSide.BLUE:
            owner_parameters = blue.ground_owner
            defender_parameters = red.ground_defender

        else:
            raise ValueError(
                "ground activity requires a phase owner"
            )

        return generate_ground_segment_activity(
            state,
            owner_parameters,
            defender_parameters,
            rng,
        )

    raise ValueError(
        f"unsupported fight phase: {state.phase}"
    )
