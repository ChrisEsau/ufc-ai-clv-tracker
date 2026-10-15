"""Stage 9 causal fight-memory decay, update, and decision-context projection."""

from __future__ import annotations
from dataclasses import dataclass, replace
from math import exp, isfinite, log
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import (
    FightMemory,
    FightState,
    FighterMemory,
    Phase,
    Side,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import (
    ActionOutcome,
    ActionResolution,
)
from .policy import BrainDecisionContext

STRIKES = frozenset(
    {
        ActionFamily.STAND_ATTACK,
        ActionFamily.STAND_COUNTER,
        ActionFamily.CLINCH_STRIKE,
        ActionFamily.GROUND_STRIKE,
        ActionFamily.BOTTOM_STRIKE,
    }
)
TAKEDOWNS = frozenset({ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN})


@dataclass(frozen=True)
class FightMemoryConfig:
    half_life_seconds: float = 60.0
    evidence_increment: float = 0.35
    miss_penalty_fraction: float = 0.20

    def __post_init__(self):
        if not isfinite(self.half_life_seconds) or self.half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be finite and positive")
        if not 0 < self.evidence_increment <= 1:
            raise ValueError("evidence_increment must be in (0, 1]")
        if not 0 <= self.miss_penalty_fraction <= 1:
            raise ValueError("miss_penalty_fraction must be in [0, 1]")


DEFAULT_FIGHT_MEMORY_CONFIG = FightMemoryConfig()


def decay_memory(
    memory: FightMemory,
    timestamp: float,
    config: FightMemoryConfig = DEFAULT_FIGHT_MEMORY_CONFIG,
) -> FightMemory:
    if timestamp < memory.updated_at_seconds:
        raise ValueError("memory cannot decay backwards")
    factor = exp(
        -log(2) * (timestamp - memory.updated_at_seconds) / config.half_life_seconds
    )

    def one(row):
        return FighterMemory(
            row.striking_edge * factor,
            row.td_success_recent * factor,
            row.td_failure_recent * factor,
            row.td_defense_success_recent * factor,
            row.control_success_recent * factor,
            row.score_state,
        )

    return FightMemory(one(memory.red), one(memory.blue), timestamp)


def update_memory(
    memory: FightMemory,
    resolution: ActionResolution,
    config: FightMemoryConfig = DEFAULT_FIGHT_MEMORY_CONFIG,
) -> FightMemory:
    """Apply evidence only after mechanics resolves the current event."""
    event = resolution.event
    current = decay_memory(memory, event.timestamp_seconds, config)
    rows = {Side.RED: current.red, Side.BLUE: current.blue}
    actor = event.actor
    other = actor.opponent
    a, o = rows[actor], rows[other]
    inc = config.evidence_increment
    if event.action_family in STRIKES:
        delta = (
            inc
            if resolution.outcome is ActionOutcome.LANDED
            else -inc * config.miss_penalty_fraction
        )
        a = replace(a, striking_edge=_signed(a.striking_edge + delta))
        o = replace(o, striking_edge=_signed(o.striking_edge - delta))
    if event.action_family in TAKEDOWNS:
        if resolution.transition is not None:
            a = replace(
                a,
                td_success_recent=_rise(a.td_success_recent, inc),
                td_failure_recent=a.td_failure_recent * (1 - inc),
            )
        else:
            a = replace(
                a,
                td_failure_recent=_rise(a.td_failure_recent, inc),
                td_success_recent=a.td_success_recent * (1 - inc),
            )
            o = replace(
                o, td_defense_success_recent=_rise(o.td_defense_success_recent, inc)
            )
    if (
        event.action_family in {ActionFamily.CONTROL, ActionFamily.CLINCH_CONTROL}
        and resolution.outcome is ActionOutcome.CONTROLLED
    ):
        a = replace(a, control_success_recent=_rise(a.control_success_recent, inc))
    rows[actor], rows[other] = a, o
    return FightMemory(rows[Side.RED], rows[Side.BLUE], event.timestamp_seconds)


def decision_context(
    state: FightState, side: Side, base: BrainDecisionContext, total_horizon: float
) -> BrainDecisionContext:
    row = state.memory.fighter(side)
    late = max(
        base.late_urgency, min(1, state.fight_time_seconds / max(total_horizon, 1))
    )
    own = state.physiology.fighter(side)
    opponent = state.physiology.fighter(side.opponent)
    own_hurt = min(1, own.cumulative_trauma / 80 + own.acute_vulnerability / 2)
    opponent_hurt = min(
        1, opponent.cumulative_trauma / 80 + opponent.acute_vulnerability / 2
    )
    return replace(
        base,
        striking_edge=_signed(base.striking_edge + row.striking_edge),
        own_hurt=max(base.own_hurt, own_hurt),
        opponent_hurt=max(base.opponent_hurt, opponent_hurt),
        td_success_recent=max(base.td_success_recent, row.td_success_recent),
        td_failure_recent=max(base.td_failure_recent, row.td_failure_recent),
        td_defense_success_recent=max(
            base.td_defense_success_recent, row.td_defense_success_recent
        ),
        control_success_recent=max(
            base.control_success_recent, row.control_success_recent
        ),
        fatigue=max(base.fatigue, 1 - own.stamina),
        score_state=_signed(base.score_state + row.score_state),
        late_urgency=late,
        bad_bottom_position=max(
            base.bad_bottom_position,
            float(
                state.phase is Phase.GROUND and state.ground_controller is side.opponent
            ),
        ),
        dominant_top_position=max(
            base.dominant_top_position,
            float(state.phase is Phase.GROUND and state.ground_controller is side),
        ),
    )


def _rise(value, inc):
    return min(1, value + (1 - value) * inc)


def _signed(value):
    return max(-1, min(1, value))
