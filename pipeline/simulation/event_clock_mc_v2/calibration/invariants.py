"""Consolidated path invariant checks."""

from __future__ import annotations
import math
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase
from pipeline.simulation.event_clock_mc_v2.causal.legality import (
    CLINCH_ACTIONS,
    GROUND_BOTTOM_ACTIONS,
    GROUND_TOP_ACTIONS,
    STANDING_ACTIONS,
)

PHASE_ACTIONS = {
    Phase.STANDING: frozenset(STANDING_ACTIONS),
    Phase.CLINCH: frozenset(CLINCH_ACTIONS),
    Phase.GROUND: frozenset(GROUND_TOP_ACTIONS + GROUND_BOTTOM_ACTIONS),
}


def inspect_path(result) -> dict[str, int]:
    exposure = sum(segment.duration for segment in result.timeline_segments)
    values = [result.final_state.fight_time_seconds]
    for side in ("red", "blue"):
        row = getattr(result.final_state.physiology, side)
        values.extend((row.stamina, row.cumulative_trauma, row.acute_vulnerability))
    return {
        "illegal_cross_phase_actions": sum(
            e.selected_action not in PHASE_ACTIONS[e.source_phase]
            for e in result.events
        ),
        "timeline_exposure_mismatch": int(
            not math.isclose(exposure, result.reported_through_seconds, abs_tol=1e-9)
        ),
        "post_finish_events": (
            sum(
                e.timestamp_seconds > result.reported_through_seconds
                for e in result.events
            )
            if result.termination
            else 0
        ),
        "invalid_state_transitions": int(
            any(segment.duration < 0 for segment in result.timeline_segments)
        ),
        "nan_or_impossible_state_values": int(
            any(not math.isfinite(v) for v in values)
            or any(
                row.stamina < 0
                or row.stamina > 1
                or row.cumulative_trauma < 0
                or row.acute_vulnerability < 0
                or row.knockdowns_suffered < 0
                for row in (
                    result.final_state.physiology.red,
                    result.final_state.physiology.blue,
                )
            )
        ),
    }


def status(counts: dict[str, int], deterministic_replay_mismatch: int = 0) -> dict:
    merged = {**counts, "deterministic_replay_mismatch": deterministic_replay_mismatch}
    return {"status": "PASS" if not any(merged.values()) else "FAIL", "counts": merged}
