"""Consolidated path invariant checks."""

from __future__ import annotations
import math
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily

STANDING = {
    ActionFamily.STAND_ATTACK,
    ActionFamily.STAND_COUNTER,
    ActionFamily.TAKEDOWN_ENTRY,
}


def inspect_path(result) -> dict[str, int]:
    exposure = sum(segment.duration for segment in result.timeline_segments)
    values = [result.final_state.fight_time_seconds]
    for side in ("red", "blue"):
        row = getattr(result.final_state.physiology, side)
        values.extend((row.stamina, row.cumulative_trauma, row.acute_vulnerability))
    return {
        "illegal_cross_phase_actions": sum(
            e.source_phase is Phase.GROUND and e.selected_action in STANDING
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
                getattr(result.final_state.physiology, s).stamina < 0
                or getattr(result.final_state.physiology, s).stamina > 1
                for s in ("red", "blue")
            )
        ),
    }


def status(counts: dict[str, int], deterministic_replay_mismatch: int = 0) -> dict:
    merged = {**counts, "deterministic_replay_mismatch": deterministic_replay_mismatch}
    return {"status": "PASS" if not any(merged.values()) else "FAIL", "counts": merged}
