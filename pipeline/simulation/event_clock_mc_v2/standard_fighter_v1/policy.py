from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp
from typing import Dict


class Phase(str, Enum):
    STANDING = "standing"
    CLINCH = "clinch"
    GROUND_TOP = "ground_top"
    GROUND_BOTTOM = "ground_bottom"


class Action(str, Enum):
    STAND_ATTACK = "stand_attack"
    STAND_COUNTER = "stand_counter"
    PRESSURE = "pressure"
    RESET_RANGE = "reset_range"
    CLINCH_ENTRY = "clinch_entry"
    TAKEDOWN_ENTRY = "takedown_entry"
    CLINCH_STRIKE = "clinch_strike"
    CLINCH_CONTROL = "clinch_control"
    CLINCH_TAKEDOWN = "clinch_takedown"
    BREAK_CLINCH = "break_clinch"
    GROUND_STRIKE = "ground_strike"
    ADVANCE_POSITION = "advance_position"
    SUBMISSION_ATTACK = "submission_attack"
    CONTROL = "control"
    DISENGAGE = "disengage"
    ESCAPE_STAND = "escape_stand"
    IMPROVE_POSITION = "improve_position"
    REVERSAL = "reversal"
    BOTTOM_STRIKE = "bottom_strike"


@dataclass(frozen=True)
class Capability:
    standing: float = 0.0
    counter: float = 0.0
    pressure: float = 0.0
    clinch: float = 0.0
    takedown: float = 0.0
    ground_top: float = 0.0
    submission: float = 0.0
    escape: float = 0.0
    reversal: float = 0.0


@dataclass(frozen=True)
class FightState:
    phase: Phase = Phase.STANDING
    striking_edge: float = 0.0          # + own fighter, - opponent
    damage_edge: float = 0.0            # + opponent more damaged, - own more damaged
    own_hurt: float = 0.0               # 0..1
    opponent_hurt: float = 0.0          # 0..1
    td_success_recent: float = 0.0       # 0..1
    td_failure_recent: float = 0.0       # 0..1
    td_defense_success_recent: float = 0.0
    control_success_recent: float = 0.0
    own_back_to_cage: float = 0.0
    opponent_back_to_cage: float = 0.0
    fatigue: float = 0.0                 # 0 fresh, 1 exhausted
    score_state: float = 0.0             # + ahead, - behind
    late_fight: float = 0.0              # 0..1
    bad_bottom_position: float = 0.0      # 0..1
    dominant_top_position: float = 0.0    # 0..1


def _legal(phase: Phase) -> tuple[Action, ...]:
    if phase == Phase.STANDING:
        return (
            Action.STAND_ATTACK, Action.STAND_COUNTER, Action.PRESSURE,
            Action.RESET_RANGE, Action.CLINCH_ENTRY, Action.TAKEDOWN_ENTRY,
        )
    if phase == Phase.CLINCH:
        return (
            Action.CLINCH_STRIKE, Action.CLINCH_CONTROL,
            Action.CLINCH_TAKEDOWN, Action.BREAK_CLINCH,
        )
    if phase == Phase.GROUND_TOP:
        return (
            Action.GROUND_STRIKE, Action.ADVANCE_POSITION,
            Action.SUBMISSION_ATTACK, Action.CONTROL, Action.DISENGAGE,
        )
    return (
        Action.ESCAPE_STAND, Action.IMPROVE_POSITION, Action.REVERSAL,
        Action.SUBMISSION_ATTACK, Action.BOTTOM_STRIKE,
    )


def utilities(state: FightState, cap: Capability) -> Dict[Action, float]:
    """Return transparent, intentionally simple V1 tactical utilities.

    Inputs are normalized research signals rather than production runtime fields.
    This prototype exists only to validate directionality of the standard-fighter
    logic before any Event Clock integration.
    """
    u = {a: 0.0 for a in _legal(state.phase)}

    if state.phase == Phase.STANDING:
        u[Action.STAND_ATTACK] = 0.35 + 0.85 * cap.standing
        u[Action.STAND_COUNTER] = 0.20 + 0.70 * cap.counter
        u[Action.PRESSURE] = 0.10 + 0.65 * cap.pressure
        u[Action.RESET_RANGE] = 0.00
        u[Action.CLINCH_ENTRY] = -0.10 + 0.70 * cap.clinch
        u[Action.TAKEDOWN_ENTRY] = -0.15 + 0.90 * cap.takedown

        # "I'm touching him up": remain in the profitable phase.
        if state.striking_edge > 0:
            edge = state.striking_edge
            u[Action.STAND_ATTACK] += 0.85 * edge
            u[Action.STAND_COUNTER] += 0.35 * edge
            u[Action.TAKEDOWN_ENTRY] -= 0.45 * edge
            u[Action.CLINCH_ENTRY] -= 0.30 * edge

        # "I'm getting hit": alternatives gain value in proportion to viability.
        if state.striking_edge < 0:
            trouble = -state.striking_edge
            u[Action.RESET_RANGE] += 0.70 * trouble
            u[Action.CLINCH_ENTRY] += 0.65 * trouble * (0.35 + max(cap.clinch, 0.0))
            u[Action.TAKEDOWN_ENTRY] += 0.85 * trouble * (0.30 + max(cap.takedown, 0.0))
            u[Action.STAND_ATTACK] -= 0.45 * trouble

        # Hurt logic.
        u[Action.STAND_ATTACK] -= 1.05 * state.own_hurt
        u[Action.PRESSURE] -= 0.65 * state.own_hurt
        u[Action.RESET_RANGE] += 1.15 * state.own_hurt
        u[Action.CLINCH_ENTRY] += 0.65 * state.own_hurt * (0.35 + max(cap.clinch, 0.0))
        u[Action.TAKEDOWN_ENTRY] += 0.70 * state.own_hurt * (0.30 + max(cap.takedown, 0.0))

        u[Action.STAND_ATTACK] += 1.10 * state.opponent_hurt
        u[Action.PRESSURE] += 0.95 * state.opponent_hurt
        u[Action.RESET_RANGE] -= 0.60 * state.opponent_hurt

        # Wrestling reinforcement / abandonment.
        u[Action.TAKEDOWN_ENTRY] += 0.75 * state.td_success_recent
        u[Action.TAKEDOWN_ENTRY] -= 0.70 * state.td_failure_recent

        # Successful defense creates a brief offensive opening.
        u[Action.STAND_ATTACK] += 0.30 * state.td_defense_success_recent
        u[Action.TAKEDOWN_ENTRY] += 0.25 * state.td_defense_success_recent * max(cap.takedown, 0.0)

        # Cage context.
        u[Action.PRESSURE] += 0.55 * state.opponent_back_to_cage
        u[Action.CLINCH_ENTRY] += 0.55 * state.opponent_back_to_cage * (0.40 + max(cap.clinch, 0.0))
        u[Action.TAKEDOWN_ENTRY] += 0.55 * state.opponent_back_to_cage * (0.35 + max(cap.takedown, 0.0))
        u[Action.RESET_RANGE] += 0.85 * state.own_back_to_cage
        u[Action.PRESSURE] -= 0.40 * state.own_back_to_cage

        # Fatigue discourages high-cost pressure and repeated shots.
        u[Action.PRESSURE] -= 0.55 * state.fatigue
        u[Action.TAKEDOWN_ENTRY] -= 0.45 * state.fatigue
        u[Action.RESET_RANGE] += 0.35 * state.fatigue

        # Score/time urgency. Behind late -> seek higher-variance offense.
        urgency = state.late_fight * max(-state.score_state, 0.0)
        safety = state.late_fight * max(state.score_state, 0.0)
        u[Action.STAND_ATTACK] += 0.65 * urgency
        u[Action.PRESSURE] += 0.55 * urgency
        u[Action.TAKEDOWN_ENTRY] += 0.35 * urgency * (0.25 + max(cap.takedown, 0.0))
        u[Action.RESET_RANGE] -= 0.45 * urgency
        u[Action.RESET_RANGE] += 0.55 * safety
        u[Action.STAND_ATTACK] -= 0.35 * safety
        u[Action.PRESSURE] -= 0.25 * safety

    elif state.phase == Phase.CLINCH:
        u[Action.CLINCH_STRIKE] = 0.20 + 0.70 * cap.clinch + 0.40 * state.opponent_hurt
        u[Action.CLINCH_CONTROL] = 0.10 + 0.55 * cap.clinch + 0.50 * state.control_success_recent
        u[Action.CLINCH_TAKEDOWN] = 0.00 + 0.80 * cap.takedown + 0.55 * state.td_success_recent - 0.55 * state.td_failure_recent
        u[Action.BREAK_CLINCH] = 0.05 + 0.45 * cap.standing
        u[Action.BREAK_CLINCH] += 0.55 * max(state.striking_edge, 0.0)
        u[Action.CLINCH_CONTROL] += 0.70 * state.own_hurt

    elif state.phase == Phase.GROUND_TOP:
        u[Action.GROUND_STRIKE] = 0.20 + 0.75 * cap.ground_top + 0.70 * state.opponent_hurt
        u[Action.ADVANCE_POSITION] = 0.15 + 0.55 * cap.ground_top
        u[Action.SUBMISSION_ATTACK] = 0.00 + 0.80 * cap.submission + 0.35 * state.opponent_hurt
        u[Action.CONTROL] = 0.10 + 0.65 * cap.ground_top + 0.50 * state.control_success_recent
        u[Action.DISENGAGE] = -0.20 + 0.55 * cap.standing
        u[Action.GROUND_STRIKE] += 0.65 * state.dominant_top_position
        u[Action.SUBMISSION_ATTACK] += 0.55 * state.dominant_top_position
        u[Action.DISENGAGE] -= 0.90 * state.dominant_top_position
        safety = state.late_fight * max(state.score_state, 0.0)
        u[Action.CONTROL] += 0.60 * safety
        u[Action.DISENGAGE] -= 0.20 * safety

    else:  # ground bottom
        u[Action.ESCAPE_STAND] = 0.15 + 0.80 * cap.escape
        u[Action.IMPROVE_POSITION] = 0.20 + 0.55 * cap.escape
        u[Action.REVERSAL] = -0.05 + 0.70 * cap.reversal
        u[Action.SUBMISSION_ATTACK] = -0.10 + 0.75 * cap.submission
        u[Action.BOTTOM_STRIKE] = -0.20 + 0.35 * cap.standing
        danger = state.bad_bottom_position
        u[Action.ESCAPE_STAND] += 0.65 * danger
        u[Action.IMPROVE_POSITION] += 1.00 * danger
        u[Action.REVERSAL] += 0.45 * danger * (0.25 + max(cap.reversal, 0.0))
        u[Action.SUBMISSION_ATTACK] -= 0.40 * danger
        u[Action.BOTTOM_STRIKE] -= 0.70 * danger
        u[Action.ESCAPE_STAND] += 0.45 * state.own_hurt
        u[Action.IMPROVE_POSITION] += 0.55 * state.own_hurt

    return u


def action_probabilities(state: FightState, cap: Capability, temperature: float = 0.55) -> Dict[Action, float]:
    raw = utilities(state, cap)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    m = max(raw.values())
    weights = {a: exp((v - m) / temperature) for a, v in raw.items()}
    z = sum(weights.values())
    return {a: w / z for a, w in weights.items()}
