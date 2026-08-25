"""First authoritative brain-driven single-path Event Clock V2 engine."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Callable

import numpy as np

from pipeline.simulation.event_clock_mc_v2.brain.capabilities import BrainCapabilities
from pipeline.simulation.event_clock_mc_v2.brain.memory import (
    DEFAULT_FIGHT_MEMORY_CONFIG,
    FightMemoryConfig,
    decision_context,
    decay_memory,
    update_memory,
)
from pipeline.simulation.event_clock_mc_v2.brain.policy import (
    BrainDecisionContext,
    BrainPolicyConfig,
    DEFAULT_BRAIN_POLICY_CONFIG,
    choose_action,
)
from pipeline.simulation.event_clock_mc_v2.brain.timing import (
    BrainTimingConfig,
    BrainTimingContext,
    DEFAULT_BRAIN_TIMING_CONFIG,
    sample_next_action_delay,
)
from pipeline.simulation.event_clock_mc_v2.causal.events import (
    ActionEvent,
    ActionFamily,
)
from pipeline.simulation.event_clock_mc_v2.causal.state import (
    FightState,
    FighterMemory,
    Phase,
    Side,
)
from pipeline.simulation.event_clock_mc_v2.causal.timeline import (
    PhaseSegment,
    PhaseTimeline,
)
from pipeline.simulation.event_clock_mc_v2.causal.transitions import (
    clinch_takedown,
    direct_takedown,
    enter_clinch,
    escape_ground,
    reverse_ground,
    separate_clinch,
    start_next_round,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.config import (
    DEFAULT_MECHANICS_CALIBRATION_CONFIG,
    FighterMechanics,
    MechanicsCalibrationConfig,
    MechanicsInputs,
    StructuralMVPPlaceholders,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import (
    ActionOutcome,
    ActionResolution,
    FightTerminationRequest,
    StrikeConsequence,
    TransitionKind,
    TransitionRequest,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action
from pipeline.simulation.event_clock_mc_v2.mechanics.physiology import (
    advance_physiology,
    apply_action_consequence,
    recover_round,
)


@dataclass(frozen=True)
class PendingAction:
    """One fighter readiness time; action family is deliberately absent."""

    actor: Side
    scheduled_time_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.actor, Side):
            raise ValueError("actor must be a Side")
        if (
            isinstance(self.scheduled_time_seconds, bool)
            or not isinstance(self.scheduled_time_seconds, (int, float))
            or not math.isfinite(self.scheduled_time_seconds)
            or self.scheduled_time_seconds < 0.0
        ):
            raise ValueError("scheduled_time_seconds must be finite and non-negative")


@dataclass(frozen=True)
class FighterEngineInputs:
    capabilities: BrainCapabilities
    timing_context: BrainTimingContext
    decision_context: BrainDecisionContext
    mechanics: FighterMechanics


@dataclass(frozen=True)
class EngineInputs:
    red: FighterEngineInputs
    blue: FighterEngineInputs
    timing_config: BrainTimingConfig = DEFAULT_BRAIN_TIMING_CONFIG
    policy_config: BrainPolicyConfig = DEFAULT_BRAIN_POLICY_CONFIG
    mechanics_placeholders: StructuralMVPPlaceholders = StructuralMVPPlaceholders()
    mechanics_calibration: MechanicsCalibrationConfig = (
        DEFAULT_MECHANICS_CALIBRATION_CONFIG
    )

    def fighter(self, side: Side) -> FighterEngineInputs:
        if not isinstance(side, Side):
            raise ValueError("side must be a Side")
        return self.red if side is Side.RED else self.blue

    @property
    def mechanics_inputs(self) -> MechanicsInputs:
        return MechanicsInputs(
            self.red.mechanics, self.blue.mechanics, self.mechanics_calibration
        )


@dataclass(frozen=True)
class EngineConfig:
    round_length_seconds: float = 300.0
    number_of_rounds: int = 3
    memory_config: FightMemoryConfig = DEFAULT_FIGHT_MEMORY_CONFIG

    def __post_init__(self) -> None:
        if (
            isinstance(self.round_length_seconds, bool)
            or not isinstance(self.round_length_seconds, (int, float))
            or not math.isfinite(self.round_length_seconds)
            or self.round_length_seconds <= 0.0
        ):
            raise ValueError("round_length_seconds must be finite and positive")
        if isinstance(self.number_of_rounds, bool) or not isinstance(
            self.number_of_rounds, int
        ):
            raise ValueError("number_of_rounds must be an integer")
        if self.number_of_rounds < 1:
            raise ValueError("number_of_rounds must be at least 1")


@dataclass(frozen=True)
class EngineRNGs:
    """Five fixed independent streams: RED/BLUE timing, RED/BLUE choice, mechanics."""

    red_timing: np.random.Generator
    blue_timing: np.random.Generator
    red_selection: np.random.Generator
    blue_selection: np.random.Generator
    mechanics: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> EngineRNGs:
        streams = np.random.SeedSequence(seed).spawn(5)
        return cls(*(np.random.default_rng(stream) for stream in streams))

    def timing(self, side: Side) -> np.random.Generator:
        return self.red_timing if side is Side.RED else self.blue_timing

    def selection(self, side: Side) -> np.random.Generator:
        return self.red_selection if side is Side.RED else self.blue_selection


TimingSampler = Callable[
    [FightState, BrainTimingContext, np.random.Generator, BrainTimingConfig], float
]
ActionChooser = Callable[
    [
        FightState,
        Side,
        BrainCapabilities,
        BrainDecisionContext,
        np.random.Generator,
        BrainPolicyConfig,
    ],
    ActionFamily,
]
MechanicsResolver = Callable[
    [
        ActionEvent,
        FightState,
        MechanicsInputs,
        np.random.Generator,
        StructuralMVPPlaceholders,
    ],
    ActionResolution,
]


@dataclass(frozen=True)
class EngineFunctions:
    """Narrow composition seams; production defaults are the approved Stage 3-5 functions."""

    timing_sampler: TimingSampler = sample_next_action_delay
    action_chooser: ActionChooser = choose_action
    mechanics_resolver: MechanicsResolver = resolve_action

    def __post_init__(self) -> None:
        for field in fields(self):
            if not callable(getattr(self, field.name)):
                raise ValueError(f"{field.name} must be callable")


@dataclass(frozen=True)
class CausalEventRecord:
    timestamp_seconds: float
    actor: Side
    source_phase: Phase
    selected_action: ActionFamily
    outcome: ActionOutcome
    transition_kind: TransitionKind | None
    resulting_phase: Phase
    resulting_controller: Side | None
    pre_decision_context: BrainDecisionContext
    resulting_actor_memory: FighterMemory
    impact: float
    knockdown: bool


@dataclass(frozen=True)
class RoundBoundaryRecord:
    timestamp_seconds: float
    round_started: int


@dataclass(frozen=True)
class CausalPathResult:
    final_state: FightState
    timeline_segments: tuple[PhaseSegment, ...]
    events: tuple[CausalEventRecord, ...]
    round_boundaries: tuple[RoundBoundaryRecord, ...]
    termination: FightTerminationRequest | None
    horizon_seconds: float
    reported_through_seconds: float
    reached_horizon: bool
    final_pending_actions: tuple[PendingAction, ...]


def initialize_pending_actions(
    state: FightState,
    inputs: EngineInputs,
    rngs: EngineRNGs,
    functions: EngineFunctions = EngineFunctions(),
) -> tuple[PendingAction, PendingAction]:
    """Sample exactly one actor-only readiness record for each fighter."""
    return tuple(_sample_pending(state, side, inputs, rngs, functions) for side in Side)


def run_causal_path(
    inputs: EngineInputs,
    *,
    seed: int,
    horizon_seconds: float,
    initial_state: FightState = FightState(),
    config: EngineConfig = EngineConfig(),
    functions: EngineFunctions = EngineFunctions(),
) -> CausalPathResult:
    """Run one bounded brain-generated causal path with fixed supplied contexts."""
    _validate_run_inputs(inputs, initial_state, horizon_seconds, config, functions)
    effective_horizon = min(
        float(horizon_seconds), config.round_length_seconds * config.number_of_rounds
    )
    state = initial_state
    timeline = PhaseTimeline.from_state(state)
    rngs = EngineRNGs.from_seed(seed)
    pending = {
        item.actor: item
        for item in initialize_pending_actions(state, inputs, rngs, functions)
    }
    events: list[CausalEventRecord] = []
    boundaries: list[RoundBoundaryRecord] = []
    termination: FightTerminationRequest | None = None

    while not state.finished and state.fight_time_seconds < effective_horizon:
        next_pending = min(
            pending.values(),
            key=lambda item: (
                item.scheduled_time_seconds,
                0 if item.actor is Side.RED else 1,
            ),
        )
        round_end = state.round_number * config.round_length_seconds

        # Boundary wins exact timestamp ties with fighter actions.
        if (
            round_end <= effective_horizon
            and round_end <= next_pending.scheduled_time_seconds
        ):
            if state.round_number >= config.number_of_rounds:
                break
            state = advance_physiology(state, round_end, inputs.mechanics_calibration)
            state = start_next_round(state, timeline, round_end)
            state = recover_round(state, inputs.mechanics_calibration)
            state = replace(
                state,
                memory=decay_memory(state.memory, round_end, config.memory_config),
            )
            boundaries.append(RoundBoundaryRecord(round_end, state.round_number))
            pending = {
                item.actor: item
                for item in initialize_pending_actions(state, inputs, rngs, functions)
            }
            continue

        if next_pending.scheduled_time_seconds > effective_horizon:
            break

        actor = next_pending.actor
        timestamp = next_pending.scheduled_time_seconds
        state = advance_physiology(state, timestamp, inputs.mechanics_calibration)
        state = replace(
            state, memory=decay_memory(state.memory, timestamp, config.memory_config)
        )
        fighter = inputs.fighter(actor)
        current_context = decision_context(
            state, actor, fighter.decision_context, effective_horizon
        )
        selected = functions.action_chooser(
            state,
            actor,
            fighter.capabilities,
            current_context,
            rngs.selection(actor),
            inputs.policy_config,
        )
        event = ActionEvent(timestamp, actor, selected, state.phase)
        resolution = functions.mechanics_resolver(
            event,
            state,
            inputs.mechanics_inputs,
            rngs.mechanics,
            inputs.mechanics_placeholders,
        )
        state = apply_action_consequence(
            state,
            actor,
            selected,
            resolution.consequence,
            fighter.mechanics,
            inputs.mechanics_calibration,
        )
        material_change = resolution.transition is not None
        if resolution.transition is not None:
            state = apply_transition_request(
                state, timeline, resolution.transition, timestamp
            )
        state = replace(
            state,
            memory=update_memory(state.memory, resolution, config.memory_config),
        )

        requested_termination = (
            resolution.consequence
            if isinstance(resolution.consequence, FightTerminationRequest)
            else (
                resolution.consequence.termination
                if isinstance(resolution.consequence, StrikeConsequence)
                else None
            )
        )
        if requested_termination is not None:
            termination = requested_termination
            state = replace(
                state,
                finished=True,
                winner=termination.winner,
                finish_method=termination.finish_method.value,
            )

        events.append(
            CausalEventRecord(
                timestamp,
                actor,
                event.source_phase,
                selected,
                resolution.outcome,
                resolution.transition.kind if resolution.transition else None,
                state.phase,
                _controller(state),
                current_context,
                state.memory.fighter(actor),
                (
                    resolution.consequence.impact
                    if isinstance(resolution.consequence, StrikeConsequence)
                    else 0.0
                ),
                (
                    resolution.consequence.knockdown
                    if isinstance(resolution.consequence, StrikeConsequence)
                    else False
                ),
            )
        )

        if termination is not None:
            pending = {}
            break
        if material_change:
            pending = {
                item.actor: item
                for item in initialize_pending_actions(state, inputs, rngs, functions)
            }
        else:
            pending[actor] = _sample_pending(state, actor, inputs, rngs, functions)

    reported_through = state.fight_time_seconds if state.finished else effective_horizon
    timeline_segments = timeline.segments_through(reported_through)
    timeline.validate()
    return CausalPathResult(
        final_state=state,
        timeline_segments=timeline_segments,
        events=tuple(events),
        round_boundaries=tuple(boundaries),
        termination=termination,
        horizon_seconds=effective_horizon,
        reported_through_seconds=reported_through,
        reached_horizon=not state.finished,
        final_pending_actions=tuple(
            sorted(pending.values(), key=lambda item: item.actor.value)
        ),
    )


def apply_transition_request(
    state: FightState,
    timeline: PhaseTimeline,
    request: TransitionRequest,
    timestamp: float,
) -> FightState:
    """Exhaustively map typed mechanics intent to approved Stage 1 operations."""
    if request.kind is TransitionKind.ENTER_CLINCH:
        return enter_clinch(state, timeline, timestamp, request.controller)
    if request.kind is TransitionKind.DIRECT_TAKEDOWN:
        return direct_takedown(state, timeline, timestamp, request.controller)
    if request.kind is TransitionKind.CLINCH_TAKEDOWN:
        return clinch_takedown(state, timeline, timestamp, request.controller)
    if request.kind is TransitionKind.BREAK_CLINCH:
        return separate_clinch(state, timeline, timestamp)
    if request.kind in {TransitionKind.ESCAPE_GROUND, TransitionKind.DISENGAGE_GROUND}:
        return escape_ground(state, timeline, timestamp)
    if request.kind is TransitionKind.REVERSE_GROUND:
        return reverse_ground(state, timeline, timestamp, request.controller)
    raise ValueError(f"unsupported transition kind: {request.kind!r}")


def _sample_pending(
    state: FightState,
    side: Side,
    inputs: EngineInputs,
    rngs: EngineRNGs,
    functions: EngineFunctions,
) -> PendingAction:
    fighter = inputs.fighter(side)
    delay = functions.timing_sampler(
        state, fighter.timing_context, rngs.timing(side), inputs.timing_config
    )
    if not math.isfinite(delay) or delay <= 0.0:
        raise ValueError("timing sampler must return a finite positive delay")
    return PendingAction(side, state.fight_time_seconds + delay)


def _controller(state: FightState) -> Side | None:
    if state.phase is Phase.CLINCH:
        return state.clinch_controller
    if state.phase is Phase.GROUND:
        return state.ground_controller
    return None


def _validate_run_inputs(
    inputs: EngineInputs,
    state: FightState,
    horizon_seconds: float,
    config: EngineConfig,
    functions: EngineFunctions,
) -> None:
    if not isinstance(inputs, EngineInputs):
        raise ValueError("inputs must be EngineInputs")
    if not isinstance(state, FightState):
        raise ValueError("initial_state must be FightState")
    if state.finished:
        raise ValueError("initial_state cannot already be finished")
    if (
        isinstance(horizon_seconds, bool)
        or not isinstance(horizon_seconds, (int, float))
        or not math.isfinite(horizon_seconds)
        or horizon_seconds < state.fight_time_seconds
    ):
        raise ValueError("horizon_seconds must be finite and not precede initial state")
    if not isinstance(config, EngineConfig):
        raise ValueError("config must be EngineConfig")
    if not isinstance(functions, EngineFunctions):
        raise ValueError("functions must be EngineFunctions")
