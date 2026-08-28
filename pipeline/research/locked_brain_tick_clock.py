"""Research-only 1-second global availability clock for the locked Brain harness.

This module is not an executable runner. It is a scheduler implementation used only
by pipeline.research.locked_brain_mc. Production Event Clock V2 mechanics remain
unchanged.

At every whole-second tick each currently admissible action receives an independent
rate-driven availability draw. If no action is available the fight advances one
second. If one or more actions are available, one event is selected from the
available set using their hazard weights. The same per-15 rate therefore drives
both availability and same-tick competition.

Standing actions:
  STAND_ATTACK, TAKEDOWN_ENTRY, CLINCH_ENTRY for each fighter.
Ground actions:
  controller: GROUND_STRIKE, SUBMISSION_ATTACK
  bottom:     SUBMISSION_ATTACK, ESCAPE_STAND
The research ground menu deliberately excludes CONTROL, BOTTOM_STRIKE, REVERSAL,
DISENGAGE, IMPROVE_POSITION and ADVANCE_POSITION.

Clinch remains mechanically frozen; its existing mean timing is converted to a
one-second availability probability and its existing intent-prior chooser supplies
the selected clinch action.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math

import numpy as np

from pipeline.research import allen_shahbazyan_ground_opportunity_submission_trace as sub_shadow
from pipeline.research import allen_shahbazyan_fighter_level_submission_trace as sub_mod
from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import action_probabilities_with_intent_priors
from pipeline.simulation.event_clock_mc_v2.brain.memory import decision_context, decay_memory, update_memory
from pipeline.simulation.event_clock_mc_v2.brain.timing import expected_action_delay
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionEvent, ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side
from pipeline.simulation.event_clock_mc_v2.causal.timeline import PhaseTimeline
from pipeline.simulation.event_clock_mc_v2.causal.transitions import start_next_round
from pipeline.simulation.event_clock_mc_v2.engine.causal_engine import (
    CausalEventRecord,
    CausalPathResult,
    EngineConfig,
    EngineFunctions,
    EngineInputs,
    EngineRNGs,
    RoundBoundaryRecord,
    _controller,
    apply_transition_request,
)
from pipeline.simulation.event_clock_mc_v2.judging import score_decision
from pipeline.simulation.event_clock_mc_v2.mechanics.physiology import (
    advance_physiology,
    apply_action_consequence,
    recover_round,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import (
    ActionOutcome,
    ActionResolution,
    FinishMethod,
    FightTerminationRequest,
    StrikeConsequence,
    SubmissionConsequence,
    TransitionKind,
    TransitionRequest,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.resolver import resolve_action

EPS = 1e-12
TICK_SECONDS = 1.0
GROUND_RATE_BY_SIDE: dict[Side, float] = {}
GROUND_BURST_BY_SIDE: dict[Side, float] = {}
ESCAPE_MEAN_SECONDS_BY_CONTROLLER: dict[Side, float] = {}
STANDING_RATE_FN = None

REMOVED_GROUND_ACTIONS = {
    ActionFamily.CONTROL,
    ActionFamily.BOTTOM_STRIKE,
    ActionFamily.REVERSAL,
    ActionFamily.DISENGAGE,
    ActionFamily.IMPROVE_POSITION,
    ActionFamily.ADVANCE_POSITION,
}


def configure(*, standing_rate_fn, ground_rate_by_side, ground_burst_by_side):
    global STANDING_RATE_FN, GROUND_RATE_BY_SIDE, GROUND_BURST_BY_SIDE
    STANDING_RATE_FN = standing_rate_fn
    GROUND_RATE_BY_SIDE = {side: float(value) for side, value in ground_rate_by_side.items()}
    GROUND_BURST_BY_SIDE = {side: float(value) for side, value in ground_burst_by_side.items()}


def _availability_probability(rate_15m: float) -> float:
    rate = max(float(rate_15m), 0.0)
    return float(1.0 - math.exp(-rate * TICK_SECONDS / 900.0))


def _submission_ground_rate(side: Side) -> float:
    return max(
        float(sub_mod.RATE_PER_15_BY_SIDE.get(side, 0.0))
        * float(sub_shadow.GROUND_HAZARD_MULTIPLIER),
        0.0,
    )


def _escape_mean_seconds(controller: Side, resolver=None) -> float:
    if resolver is not None and hasattr(resolver, "escape_mean_seconds"):
        return max(float(resolver.escape_mean_seconds(controller)), 1.0)
    if controller in ESCAPE_MEAN_SECONDS_BY_CONTROLLER:
        return max(float(ESCAPE_MEAN_SECONDS_BY_CONTROLLER[controller]), 1.0)
    raise RuntimeError("escape mean unavailable for locked tick clock")


def _append_trace_decision(brain, state, actor, context, options, selected):
    if brain is None or not hasattr(brain, "decisions"):
        return
    total = sum(max(float(row["weight"]), 0.0) for row in options)
    rows = []
    for row in options:
        weight = max(float(row["weight"]), 0.0)
        rows.append({
            "action": row["action"].value,
            "actor": row["actor"].value,
            "rate_15m": float(row.get("rate_15m", 0.0)),
            "availability_probability_1s": float(row.get("availability_probability", 0.0)),
            "probability": weight / total if total > 0 else 0.0,
        })
    brain.decisions.append({
        "decision_index": len(brain.decisions),
        "timestamp_before_action": float(state.fight_time_seconds),
        "round": int(state.round_number),
        "phase": state.phase.value,
        "phase_started_at": float(state.phase_started_at),
        "ground_control_elapsed_seconds": (
            float(state.fight_time_seconds - state.phase_started_at)
            if state.phase is Phase.GROUND else None
        ),
        "ground_controller": None if state.ground_controller is None else state.ground_controller.value,
        "clinch_controller": None if state.clinch_controller is None else state.clinch_controller.value,
        "actor": actor.value,
        "context": asdict(context),
        "brain_options": rows,
        "selected_action": selected.value,
        "global_tick_seconds": TICK_SECONDS,
        "dynamic_pressure": 0.0,
    })


class AlwaysEscapeResolver:
    """Escape timing lives in the tick hazard; a selected escape therefore succeeds."""
    def __init__(self, model, seed):
        del seed
        self.model = model
        self.escape_checks = []
        ESCAPE_MEAN_SECONDS_BY_CONTROLLER[Side.RED] = self.escape_mean_seconds(Side.RED)
        ESCAPE_MEAN_SECONDS_BY_CONTROLLER[Side.BLUE] = self.escape_mean_seconds(Side.BLUE)

    def escape_mean_seconds(self, controller: Side) -> float:
        key = "red_controls_blue" if controller is Side.RED else "blue_controls_red"
        return max(float(self.model["matchups"][key]["expected_control_seconds"]), 1.0)

    def __call__(self, event, state, inputs, rng, placeholders, ko_kd_rng=None, submission_rng=None):
        if event.action_family is ActionFamily.ESCAPE_STAND:
            mean = self.escape_mean_seconds(state.ground_controller)
            self.escape_checks.append({
                "timestamp": float(event.timestamp_seconds),
                "actor": event.actor.value,
                "controller": state.ground_controller.value,
                "elapsed_control_seconds": float(state.fight_time_seconds - state.phase_started_at),
                "escape_mean_seconds": mean,
                "success": True,
                "tick_hazard_semantics": True,
            })
            return ActionResolution(
                event,
                ActionOutcome.ESCAPED,
                TransitionRequest(TransitionKind.ESCAPE_GROUND, Phase.GROUND, Phase.STANDING),
            )
        return resolve_action(
            event, state, inputs, rng, placeholders, ko_kd_rng, submission_rng
        )


def _ground_candidates(state, brain, rngs, resolver, burst_remaining):
    controller = state.ground_controller
    bottom = controller.opponent
    out = []
    burst_key = (int(state.round_number), controller, round(float(state.phase_started_at), 9))
    remaining = burst_remaining.get(burst_key)
    if remaining is None:
        remaining = int(rngs.selection(controller).poisson(max(GROUND_BURST_BY_SIDE.get(controller, 0.0), 0.0)))
        burst_remaining[burst_key] = remaining

    strike_rate = max(GROUND_RATE_BY_SIDE.get(controller, 0.0), 0.0)
    strike_p = _availability_probability(strike_rate)
    strike_available = remaining > 0 or rngs.selection(controller).random() < strike_p
    if strike_available:
        out.append({
            "actor": controller,
            "action": ActionFamily.GROUND_STRIKE,
            "rate_15m": strike_rate,
            "availability_probability": 1.0 if remaining > 0 else strike_p,
            "weight": max(strike_rate, EPS),
            "burst_key": burst_key if remaining > 0 else None,
        })

    for side in (controller, bottom):
        sub_rate = _submission_ground_rate(side)
        p = _availability_probability(sub_rate)
        if rngs.selection(side).random() < p:
            out.append({
                "actor": side,
                "action": ActionFamily.SUBMISSION_ATTACK,
                "rate_15m": sub_rate,
                "availability_probability": p,
                "weight": max(sub_rate, EPS),
            })

    mean_escape = _escape_mean_seconds(controller, resolver)
    escape_rate = 900.0 / mean_escape
    p_escape = _availability_probability(escape_rate)
    if rngs.selection(bottom).random() < p_escape:
        out.append({
            "actor": bottom,
            "action": ActionFamily.ESCAPE_STAND,
            "rate_15m": escape_rate,
            "availability_probability": p_escape,
            "weight": max(escape_rate, EPS),
        })
    return out


def _standing_candidates(state, brain, inputs, rngs):
    if STANDING_RATE_FN is None:
        raise RuntimeError("locked tick clock not configured with standing_rate_fn")
    out = []
    for side in Side:
        fighter = inputs.fighter(side)
        context = decision_context(state, side, fighter.decision_context, math.inf)
        rates, _ = STANDING_RATE_FN(
            state, side, fighter.capabilities, context, brain.priors[side], inputs.policy_config
        )
        for action, rate in rates.items():
            p = _availability_probability(rate)
            if rngs.selection(side).random() < p:
                out.append({
                    "actor": side,
                    "action": action,
                    "rate_15m": float(rate),
                    "availability_probability": p,
                    "weight": max(float(rate), EPS),
                })
    return out


def _clinch_candidates(state, brain, inputs, rngs):
    out = []
    for side in Side:
        fighter = inputs.fighter(side)
        mean = expected_action_delay(state, fighter.timing_context, inputs.timing_config)
        total_rate = 900.0 / max(mean, EPS)
        p = _availability_probability(total_rate)
        if rngs.selection(side).random() >= p:
            continue
        context = decision_context(state, side, fighter.decision_context, math.inf)
        rows = action_probabilities_with_intent_priors(
            state, side, fighter.capabilities, context, brain.priors[side], inputs.policy_config
        )
        probs = np.asarray([row.probability for row in rows], float)
        probs /= probs.sum()
        selected = rows[int(rngs.selection(side).choice(len(rows), p=probs))].action_family
        out.append({
            "actor": side,
            "action": selected,
            "rate_15m": total_rate,
            "availability_probability": p,
            "weight": max(total_rate, EPS),
        })
    return out


def run_causal_path(
    inputs: EngineInputs,
    *,
    seed: int,
    horizon_seconds: float,
    initial_state: FightState = FightState(),
    config: EngineConfig = EngineConfig(),
    functions: EngineFunctions = EngineFunctions(),
) -> CausalPathResult:
    """Run one path on the locked one-second global action-availability clock."""
    effective_horizon = min(
        float(horizon_seconds), config.round_length_seconds * config.number_of_rounds
    )
    state = initial_state
    timeline = PhaseTimeline.from_state(state)
    rngs = EngineRNGs.from_seed(seed)
    competition_rng = np.random.default_rng((int(seed) ^ 0x31434C4F434B) & ((1 << 63) - 1))
    brain = getattr(functions.action_chooser, "__self__", None)
    if brain is None or not hasattr(brain, "priors"):
        raise RuntimeError("locked tick clock requires the locked TraceBrain bound chooser")
    resolver = functions.mechanics_resolver

    events = []
    boundaries = []
    termination = None
    burst_remaining = {}

    while not state.finished and state.fight_time_seconds < effective_horizon:
        round_end = state.round_number * config.round_length_seconds
        next_tick = min(state.fight_time_seconds + TICK_SECONDS, effective_horizon, round_end)

        if next_tick >= round_end - 1e-12:
            state = advance_physiology(state, round_end, inputs.mechanics_calibration)
            if state.round_number >= config.number_of_rounds:
                break
            state = start_next_round(state, timeline, round_end)
            state = recover_round(state, inputs.mechanics_calibration)
            state = replace(
                state,
                memory=decay_memory(state.memory, round_end, config.memory_config),
            )
            boundaries.append(RoundBoundaryRecord(round_end, state.round_number))
            burst_remaining.clear()
            continue

        state = advance_physiology(state, next_tick, inputs.mechanics_calibration)
        state = replace(
            state, memory=decay_memory(state.memory, next_tick, config.memory_config)
        )

        if state.phase is Phase.STANDING:
            candidates = _standing_candidates(state, brain, inputs, rngs)
        elif state.phase is Phase.GROUND:
            candidates = _ground_candidates(state, brain, rngs, resolver, burst_remaining)
        else:
            candidates = _clinch_candidates(state, brain, inputs, rngs)

        if not candidates:
            continue

        weights = np.asarray([max(float(row["weight"]), EPS) for row in candidates], float)
        weights /= weights.sum()
        chosen_index = int(competition_rng.choice(len(candidates), p=weights))
        chosen = candidates[chosen_index]
        actor = chosen["actor"]
        selected = chosen["action"]
        fighter = inputs.fighter(actor)
        context = decision_context(state, actor, fighter.decision_context, effective_horizon)
        _append_trace_decision(brain, state, actor, context, candidates, selected)

        burst_key = chosen.get("burst_key")
        if burst_key is not None:
            burst_remaining[burst_key] = max(0, int(burst_remaining.get(burst_key, 0)) - 1)

        event = ActionEvent(float(state.fight_time_seconds), actor, selected, state.phase)
        resolution = functions.mechanics_resolver(
            event,
            state,
            inputs.mechanics_inputs,
            rngs.mechanics,
            inputs.mechanics_placeholders,
            rngs.ko_kd,
            rngs.submission,
        )
        state = apply_action_consequence(
            state,
            actor,
            selected,
            resolution.consequence,
            fighter.mechanics,
            inputs.mechanics_calibration,
        )
        if resolution.transition is not None:
            state = apply_transition_request(
                state, timeline, resolution.transition, float(state.fight_time_seconds)
            )
        state = replace(
            state, memory=update_memory(state.memory, resolution, config.memory_config)
        )

        requested_termination = (
            resolution.consequence
            if isinstance(resolution.consequence, FightTerminationRequest)
            else (
                resolution.consequence.termination
                if isinstance(resolution.consequence, (StrikeConsequence, SubmissionConsequence))
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

        events.append(CausalEventRecord(
            float(event.timestamp_seconds), actor, event.source_phase, selected,
            resolution.outcome,
            resolution.transition.kind if resolution.transition else None,
            state.phase, _controller(state), context, state.memory.fighter(actor),
            resolution.consequence.impact if isinstance(resolution.consequence, StrikeConsequence) else 0.0,
            resolution.consequence.knockdown if isinstance(resolution.consequence, StrikeConsequence) else False,
            resolution.consequence.ko_probability if isinstance(resolution.consequence, StrikeConsequence) else 0.0,
            resolution.consequence.knockdown_probability if isinstance(resolution.consequence, StrikeConsequence) else 0.0,
            resolution.consequence.prior_defender_kds if isinstance(resolution.consequence, StrikeConsequence) else 0,
            bool(isinstance(resolution.consequence, StrikeConsequence) and resolution.consequence.termination is not None and resolution.consequence.termination.finish_method is FinishMethod.KO_TKO),
            resolution.consequence.ko_kd_architecture if isinstance(resolution.consequence, StrikeConsequence) else None,
            isinstance(resolution.consequence, SubmissionConsequence),
            resolution.consequence.conversion_probability if isinstance(resolution.consequence, SubmissionConsequence) else 0.0,
            bool(isinstance(resolution.consequence, SubmissionConsequence) and resolution.consequence.success),
        ))

    reached_scheduled_horizon = not state.finished
    reported_through = state.fight_time_seconds if state.finished else effective_horizon
    timeline_segments = timeline.segments_through(reported_through)
    timeline.validate()
    decision = None
    if reached_scheduled_horizon and math.isclose(
        effective_horizon, config.round_length_seconds * config.number_of_rounds
    ):
        decision = score_decision(
            events,
            timeline_segments,
            rounds=config.number_of_rounds,
            round_length=config.round_length_seconds,
            model=inputs.judge_model,
            rng=rngs.judging,
        )
        termination = FightTerminationRequest(decision.winner, FinishMethod.DECISION)
        state = replace(
            state,
            fight_time_seconds=effective_horizon,
            finished=True,
            winner=decision.winner,
            finish_method=decision.classification,
        )

    return CausalPathResult(
        final_state=state,
        timeline_segments=timeline_segments,
        events=tuple(events),
        round_boundaries=tuple(boundaries),
        termination=termination,
        horizon_seconds=effective_horizon,
        reported_through_seconds=reported_through,
        reached_horizon=reached_scheduled_horizon,
        final_pending_actions=tuple(),
        decision=decision,
    )
