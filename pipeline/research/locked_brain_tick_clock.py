"""Research-only 1-second global probability clock for the locked Brain harness.

This module is not an executable runner. It is used only by
pipeline.research.locked_brain_mc; production Event Clock V2 remains unchanged.

Every tick first evaluates the validated time-survival KO/TKO competing risk, then,
if the fight survives, evaluates the currently admissible Brain actions. Action rates
control availability exactly once. Simultaneous available actions are resolved by
conditional Brain chooser weights across only those available actions.
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
KO_HAZARDS_BY_SIDE: dict[Side, np.ndarray] = {}
KO_FIGHTER_NAMES_BY_SIDE: dict[Side, str] = {}
STANDING_RATE_FN = None

REMOVED_GROUND_ACTIONS = {
    ActionFamily.CONTROL,
    ActionFamily.BOTTOM_STRIKE,
    ActionFamily.REVERSAL,
    ActionFamily.DISENGAGE,
    ActionFamily.IMPROVE_POSITION,
    ActionFamily.ADVANCE_POSITION,
}


def configure(
    *,
    standing_rate_fn,
    ground_rate_by_side,
    ground_burst_by_side,
    ko_hazards_by_side=None,
    ko_fighter_names_by_side=None,
):
    global STANDING_RATE_FN, GROUND_RATE_BY_SIDE, GROUND_BURST_BY_SIDE
    global KO_HAZARDS_BY_SIDE, KO_FIGHTER_NAMES_BY_SIDE
    STANDING_RATE_FN = standing_rate_fn
    GROUND_RATE_BY_SIDE = {side: float(value) for side, value in ground_rate_by_side.items()}
    GROUND_BURST_BY_SIDE = {side: float(value) for side, value in ground_burst_by_side.items()}
    KO_HAZARDS_BY_SIDE = {
        side: np.asarray(value, dtype=float)
        for side, value in (ko_hazards_by_side or {}).items()
    }
    KO_FIGHTER_NAMES_BY_SIDE = {
        side: str(value) for side, value in (ko_fighter_names_by_side or {}).items()
    }


def _availability_probability(rate_15m: float, exposure_seconds: float = TICK_SECONDS) -> float:
    rate = max(float(rate_15m), 0.0)
    exposure = max(float(exposure_seconds), 0.0)
    return float(1.0 - math.exp(-rate * exposure / 900.0))


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


def _trace_option(actor, action, rate_15m, probability, draw, available, *, source="rate"):
    return {
        "actor": actor.value,
        "action": None if action is None else action.value,
        "rate_15m": float(rate_15m),
        "availability_probability_1s": float(probability),
        "availability_draw": None if draw is None else float(draw),
        "available": bool(available),
        "source": source,
    }


def _ko_piece_index(timestamp: float, pieces: int) -> int:
    return min(max(int(math.ceil(max(float(timestamp), EPS) / 300.0)) - 1, 0), pieces - 1)


def _ko_tick_probability(tick_end: float, exposure_seconds: float, rng):
    if not KO_HAZARDS_BY_SIDE:
        return None, None
    if set(KO_HAZARDS_BY_SIDE) != {Side.RED, Side.BLUE}:
        raise RuntimeError("embedded KO clock requires hazards for both sides")

    hazards = {}
    for side in Side:
        pieces = KO_HAZARDS_BY_SIDE[side]
        if pieces.size < 1:
            raise RuntimeError(f"empty KO hazard vector for {side.value}")
        hazards[side] = max(float(pieces[_ko_piece_index(tick_end, len(pieces))]), 0.0)

    total_hazard = float(hazards[Side.RED] + hazards[Side.BLUE])
    exposure = max(float(exposure_seconds), 0.0)
    any_probability = float(1.0 - math.exp(-total_hazard * exposure)) if total_hazard > 0 else 0.0
    any_draw = float(rng.random())
    fires = any_draw < any_probability
    winner = None
    cause_draw = None
    if fires:
        cause_draw = float(rng.random())
        red_share = hazards[Side.RED] / total_hazard if total_hazard > 0 else 0.5
        winner = Side.RED if cause_draw < red_share else Side.BLUE

    ko = {}
    for side in Side:
        cause_probability = (
            any_probability * hazards[side] / total_hazard if total_hazard > 0 else 0.0
        )
        ko[side.value] = {
            "fighter": KO_FIGHTER_NAMES_BY_SIDE.get(side),
            "hazard_per_second": hazards[side],
            "exposure_seconds": exposure,
            "probability_in_interval": cause_probability,
            "probability_next_1s": (
                (1.0 - math.exp(-total_hazard)) * hazards[side] / total_hazard
                if total_hazard > 0 else 0.0
            ),
            "any_ko_probability_in_interval": any_probability,
            "any_ko_draw": any_draw,
            "cause_draw_if_ko": cause_draw,
            "sampled_clock_time": float(tick_end) if winner is side else None,
            "fires_in_this_tick_interval": winner is side,
        }
    return winner, ko


def _collision_policy_weights(state, brain, inputs, candidates):
    if not candidates:
        return
    policy_by_side = {}
    for side in {row["actor"] for row in candidates}:
        fighter = inputs.fighter(side)
        context = decision_context(state, side, fighter.decision_context, math.inf)
        rows = action_probabilities_with_intent_priors(
            state, side, fighter.capabilities, context, brain.priors[side], inputs.policy_config
        )
        policy_by_side[side] = {
            row.action_family: max(float(row.probability), 0.0) for row in rows
        }
    weights = []
    for candidate in candidates:
        weight = float(policy_by_side.get(candidate["actor"], {}).get(candidate["action"], 0.0))
        candidate["collision_weight"] = weight
        weights.append(weight)
    total = float(sum(weights))
    if total <= EPS:
        probability = 1.0 / float(len(candidates))
        for candidate in candidates:
            candidate["collision_probability"] = probability
            candidate["collision_weight_fallback"] = True
    else:
        for candidate in candidates:
            candidate["collision_probability"] = float(candidate["collision_weight"] / total)
            candidate["collision_weight_fallback"] = False


def _append_tick_trace(brain, state, diagnostics, candidates, selected, exposure_seconds, ko=None, ko_winner=None):
    if brain is None:
        return
    if not hasattr(brain, "tick_trace"):
        brain.tick_trace = []
    candidate_lookup = {(row["actor"].value, row["action"].value): row for row in candidates}
    traced_options = []
    for diagnostic in diagnostics:
        row = dict(diagnostic)
        candidate = candidate_lookup.get((row.get("actor"), row.get("action")))
        row["collision_weight"] = None if candidate is None else candidate.get("collision_weight")
        row["selection_probability_given_available"] = None if candidate is None else candidate.get("collision_probability")
        traced_options.append(row)
    ko_event = ko_winner is not None
    brain.tick_trace.append({
        "tick": len(brain.tick_trace) + 1,
        "timestamp": float(state.fight_time_seconds),
        "exposure_seconds": float(exposure_seconds),
        "round": int(state.round_number),
        "phase": state.phase.value,
        "ground_controller": None if state.ground_controller is None else state.ground_controller.value,
        "clinch_controller": None if state.clinch_controller is None else state.clinch_controller.value,
        "options": traced_options,
        "available_count": len(candidates),
        "collision": len(candidates) > 1,
        "collision_rule": "embedded_ko_first_then_brain_policy_weights_among_available",
        "selected_actor": ko_winner.value if ko_event else (None if selected is None else selected["actor"].value),
        "selected_action": "ko_clock" if ko_event else (None if selected is None else selected["action"].value),
        "selected_probability_given_available": 1.0 if ko_event else (None if selected is None else float(selected["collision_probability"])),
        "ko_clock_event": ko_event,
        "ko": ko or {},
    })


def _append_trace_decision(brain, state, actor, context, options, selected):
    if brain is None or not hasattr(brain, "decisions"):
        return
    rows = []
    for row in options:
        rows.append({
            "action": row["action"].value,
            "actor": row["actor"].value,
            "rate_15m": float(row.get("rate_15m", 0.0)),
            "availability_probability_1s": float(row.get("availability_probability", 0.0)),
            "collision_weight": float(row.get("collision_weight", 0.0)),
            "probability": float(row.get("collision_probability", 0.0)),
        })
    brain.decisions.append({
        "decision_index": len(brain.decisions),
        "timestamp_before_action": float(state.fight_time_seconds),
        "round": int(state.round_number),
        "phase": state.phase.value,
        "phase_started_at": float(state.phase_started_at),
        "ground_control_elapsed_seconds": float(state.fight_time_seconds - state.phase_started_at) if state.phase is Phase.GROUND else None,
        "ground_controller": None if state.ground_controller is None else state.ground_controller.value,
        "clinch_controller": None if state.clinch_controller is None else state.clinch_controller.value,
        "actor": actor.value,
        "context": asdict(context),
        "brain_options": rows,
        "selected_action": selected.value,
        "global_tick_seconds": TICK_SECONDS,
        "collision_rule": "brain_policy_weights_among_available",
        "dynamic_pressure": 0.0,
    })


class AlwaysEscapeResolver:
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
            return ActionResolution(event, ActionOutcome.ESCAPED, TransitionRequest(TransitionKind.ESCAPE_GROUND, Phase.GROUND, Phase.STANDING))
        return resolve_action(event, state, inputs, rng, placeholders, ko_kd_rng, submission_rng)


def _ground_candidates(state, brain, rngs, resolver, burst_remaining, exposure_seconds):
    del brain
    controller = state.ground_controller
    bottom = controller.opponent
    out, diagnostics = [], []
    burst_key = (int(state.round_number), controller, round(float(state.phase_started_at), 9))
    remaining = burst_remaining.get(burst_key)
    if remaining is None:
        remaining = int(rngs.selection(controller).poisson(max(GROUND_BURST_BY_SIDE.get(controller, 0.0), 0.0)))
        burst_remaining[burst_key] = remaining
    strike_rate = max(GROUND_RATE_BY_SIDE.get(controller, 0.0), 0.0)
    strike_p = _availability_probability(strike_rate, exposure_seconds)
    if remaining > 0:
        strike_draw, strike_available, effective_p, source = None, True, 1.0, "burst"
    else:
        strike_draw = float(rngs.selection(controller).random())
        strike_available = strike_draw < strike_p
        effective_p, source = strike_p, "rate"
    diagnostics.append(_trace_option(controller, ActionFamily.GROUND_STRIKE, strike_rate, effective_p, strike_draw, strike_available, source=source))
    if strike_available:
        out.append({"actor": controller, "action": ActionFamily.GROUND_STRIKE, "rate_15m": strike_rate, "availability_probability": effective_p, "burst_key": burst_key if remaining > 0 else None})
    for side in (controller, bottom):
        sub_rate = _submission_ground_rate(side)
        p = _availability_probability(sub_rate, exposure_seconds)
        draw = float(rngs.selection(side).random())
        available = draw < p
        diagnostics.append(_trace_option(side, ActionFamily.SUBMISSION_ATTACK, sub_rate, p, draw, available))
        if available:
            out.append({"actor": side, "action": ActionFamily.SUBMISSION_ATTACK, "rate_15m": sub_rate, "availability_probability": p})
    mean_escape = _escape_mean_seconds(controller, resolver)
    escape_rate = 900.0 / mean_escape
    p_escape = _availability_probability(escape_rate, exposure_seconds)
    escape_draw = float(rngs.selection(bottom).random())
    escape_available = escape_draw < p_escape
    diagnostics.append(_trace_option(bottom, ActionFamily.ESCAPE_STAND, escape_rate, p_escape, escape_draw, escape_available))
    if escape_available:
        out.append({"actor": bottom, "action": ActionFamily.ESCAPE_STAND, "rate_15m": escape_rate, "availability_probability": p_escape})
    return out, diagnostics


def _standing_candidates(state, brain, inputs, rngs, exposure_seconds):
    if STANDING_RATE_FN is None:
        raise RuntimeError("locked tick clock not configured with standing_rate_fn")
    out, diagnostics = [], []
    for side in Side:
        fighter = inputs.fighter(side)
        context = decision_context(state, side, fighter.decision_context, math.inf)
        rates, _ = STANDING_RATE_FN(state, side, fighter.capabilities, context, brain.priors[side], inputs.policy_config)
        for action, rate in rates.items():
            p = _availability_probability(rate, exposure_seconds)
            draw = float(rngs.selection(side).random())
            available = draw < p
            diagnostics.append(_trace_option(side, action, rate, p, draw, available))
            if available:
                out.append({"actor": side, "action": action, "rate_15m": float(rate), "availability_probability": p})
    return out, diagnostics


def _clinch_candidates(state, brain, inputs, rngs, exposure_seconds):
    out, diagnostics = [], []
    for side in Side:
        fighter = inputs.fighter(side)
        mean = expected_action_delay(state, fighter.timing_context, inputs.timing_config)
        total_rate = 900.0 / max(mean, EPS)
        p = _availability_probability(total_rate, exposure_seconds)
        draw = float(rngs.selection(side).random())
        available = draw < p
        if not available:
            diagnostics.append(_trace_option(side, None, total_rate, p, draw, False, source="clinch_opportunity"))
            continue
        context = decision_context(state, side, fighter.decision_context, math.inf)
        rows = action_probabilities_with_intent_priors(state, side, fighter.capabilities, context, brain.priors[side], inputs.policy_config)
        probs = np.asarray([row.probability for row in rows], float)
        probs /= probs.sum()
        selected = rows[int(rngs.selection(side).choice(len(rows), p=probs))].action_family
        diagnostics.append(_trace_option(side, selected, total_rate, p, draw, True, source="clinch_opportunity"))
        out.append({"actor": side, "action": selected, "rate_15m": total_rate, "availability_probability": p})
    return out, diagnostics


def run_causal_path(
    inputs: EngineInputs,
    *,
    seed: int,
    horizon_seconds: float,
    initial_state: FightState = FightState(),
    config: EngineConfig = EngineConfig(),
    functions: EngineFunctions = EngineFunctions(),
) -> CausalPathResult:
    effective_horizon = min(float(horizon_seconds), config.round_length_seconds * config.number_of_rounds)
    state = initial_state
    timeline = PhaseTimeline.from_state(state)
    rngs = EngineRNGs.from_seed(seed)
    competition_rng = np.random.default_rng((int(seed) ^ 0x31434C4F434B) & ((1 << 63) - 1))
    ko_rng = np.random.default_rng((int(seed) ^ 0x4B4F425241494E) & ((1 << 63) - 1))
    brain = getattr(functions.action_chooser, "__self__", None)
    if brain is None or not hasattr(brain, "priors"):
        raise RuntimeError("locked tick clock requires the locked TraceBrain bound chooser")
    brain.tick_trace = []
    resolver = functions.mechanics_resolver
    events, boundaries = [], []
    termination = None
    burst_remaining = {}

    while not state.finished and state.fight_time_seconds < effective_horizon:
        round_end = state.round_number * config.round_length_seconds
        tick_start = float(state.fight_time_seconds)
        next_tick = min(tick_start + TICK_SECONDS, effective_horizon, round_end)
        exposure_seconds = max(float(next_tick - tick_start), 0.0)
        if exposure_seconds <= EPS:
            raise RuntimeError(f"non-positive tick exposure at t={tick_start} round={state.round_number}")
        at_round_end = next_tick >= round_end - 1e-12

        ko_winner, ko_trace = _ko_tick_probability(next_tick, exposure_seconds, ko_rng)
        state = advance_physiology(state, next_tick, inputs.mechanics_calibration)
        state = replace(state, memory=decay_memory(state.memory, next_tick, config.memory_config))

        if ko_winner is not None:
            termination = FightTerminationRequest(ko_winner, FinishMethod.KO_TKO)
            state = replace(state, finished=True, winner=ko_winner, finish_method=FinishMethod.KO_TKO.value)
            _append_tick_trace(brain, state, [], [], None, exposure_seconds, ko_trace, ko_winner)
            break

        if state.phase is Phase.STANDING:
            candidates, diagnostics = _standing_candidates(state, brain, inputs, rngs, exposure_seconds)
        elif state.phase is Phase.GROUND:
            candidates, diagnostics = _ground_candidates(state, brain, rngs, resolver, burst_remaining, exposure_seconds)
        else:
            candidates, diagnostics = _clinch_candidates(state, brain, inputs, rngs, exposure_seconds)

        chosen = None
        if candidates:
            _collision_policy_weights(state, brain, inputs, candidates)
            probabilities = np.asarray([row["collision_probability"] for row in candidates], dtype=float)
            probabilities /= probabilities.sum()
            chosen = candidates[int(competition_rng.choice(len(candidates), p=probabilities))]

        _append_tick_trace(brain, state, diagnostics, candidates, chosen, exposure_seconds, ko_trace, None)

        if chosen is not None:
            actor, selected = chosen["actor"], chosen["action"]
            fighter = inputs.fighter(actor)
            context = decision_context(state, actor, fighter.decision_context, effective_horizon)
            _append_trace_decision(brain, state, actor, context, candidates, selected)
            burst_key = chosen.get("burst_key")
            if burst_key is not None:
                burst_remaining[burst_key] = max(0, int(burst_remaining.get(burst_key, 0)) - 1)
            event = ActionEvent(float(state.fight_time_seconds), actor, selected, state.phase)
            resolution = functions.mechanics_resolver(event, state, inputs.mechanics_inputs, rngs.mechanics, inputs.mechanics_placeholders, rngs.ko_kd, rngs.submission)
            state = apply_action_consequence(state, actor, selected, resolution.consequence, fighter.mechanics, inputs.mechanics_calibration)
            if resolution.transition is not None:
                state = apply_transition_request(state, timeline, resolution.transition, float(state.fight_time_seconds))
            state = replace(state, memory=update_memory(state.memory, resolution, config.memory_config))
            requested_termination = (
                resolution.consequence if isinstance(resolution.consequence, FightTerminationRequest)
                else resolution.consequence.termination if isinstance(resolution.consequence, (StrikeConsequence, SubmissionConsequence))
                else None
            )
            if requested_termination is not None:
                termination = requested_termination
                state = replace(state, finished=True, winner=termination.winner, finish_method=termination.finish_method.value)
            events.append(CausalEventRecord(
                float(event.timestamp_seconds), actor, event.source_phase, selected,
                resolution.outcome, resolution.transition.kind if resolution.transition else None,
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

        if state.finished:
            break
        if at_round_end:
            if state.round_number >= config.number_of_rounds:
                break
            state = start_next_round(state, timeline, round_end)
            state = recover_round(state, inputs.mechanics_calibration)
            state = replace(state, memory=decay_memory(state.memory, round_end, config.memory_config))
            boundaries.append(RoundBoundaryRecord(round_end, state.round_number))
            burst_remaining.clear()

    reached_scheduled_horizon = not state.finished
    reported_through = state.fight_time_seconds if state.finished else effective_horizon
    timeline_segments = timeline.segments_through(reported_through)
    timeline.validate()
    decision = None
    if reached_scheduled_horizon and math.isclose(effective_horizon, config.round_length_seconds * config.number_of_rounds):
        decision = score_decision(events, timeline_segments, rounds=config.number_of_rounds, round_length=config.round_length_seconds, model=inputs.judge_model, rng=rngs.judging)
        termination = FightTerminationRequest(decision.winner, FinishMethod.DECISION)
        state = replace(state, fight_time_seconds=effective_horizon, finished=True, winner=decision.winner, finish_method=decision.classification)

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


run_causal_path.embedded_ko_clock = True
