"""V2-native faithful migration of frozen damage, KD, finish, and stamina formulas."""

from __future__ import annotations
from dataclasses import replace
from math import exp, log
import numpy as np
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import (
    FightPhysiology,
    FightState,
    FighterPhysiology,
    Phase,
    Side,
)
from .config import MechanicsInputs
from .resolution import FightTerminationRequest, FinishMethod, StrikeConsequence

ACTION_COSTS = {
    ActionFamily.STAND_ATTACK: 0.70,
    ActionFamily.STAND_COUNTER: 0.70,
    ActionFamily.CLINCH_STRIKE: 0.70,
    ActionFamily.GROUND_STRIKE: 0.70,
    ActionFamily.BOTTOM_STRIKE: 0.70,
    ActionFamily.TAKEDOWN_ENTRY: 3.0,
    ActionFamily.CLINCH_TAKEDOWN: 3.0,
    ActionFamily.CLINCH_ENTRY: 1.0,
    ActionFamily.SUBMISSION_ATTACK: 2.5,
    ActionFamily.ESCAPE_STAND: 1.5,
    ActionFamily.REVERSAL: 2.5,
}


def resolve_landed_strike(
    event,
    state: FightState,
    inputs: MechanicsInputs,
    landed: bool,
    rng: np.random.Generator,
) -> StrikeConsequence:
    if not landed:
        return StrikeConsequence(False)
    attacker = inputs.fighter(event.actor)
    defender = inputs.fighter(event.actor.opponent)
    target = state.physiology.fighter(event.actor.opponent)
    power = exp((attacker.striking_power - 50) / 55)
    severity = rng.gamma(1, 2)
    if rng.random() < 0.06:
        severity += rng.gamma(1.25, 4.8)
    impact = max(1e-9, power * severity * 0.5)
    trauma = impact * exp(-(defender.damage_durability - 50) / 40)
    new_trauma = target.cumulative_trauma + trauma
    resistance = max(
        1e-6,
        exp((defender.knockdown_resistance - 50) / 32)
        * exp(-new_trauma / 80)
        * exp(-target.acute_vulnerability),
    )
    p_kd = _sigmoid(2 * (log(impact / resistance) - log(36)))
    kd = bool(rng.random() < p_kd)
    acute = 0.5 if kd else 0.0
    finish_resistance = max(
        1e-9,
        exp(
            ((defender.damage_durability + defender.knockdown_resistance) / 2 - 50) / 32
        )
        * exp(-new_trauma / 120)
        * exp(-(target.acute_vulnerability + acute)),
    )
    logit = 2 * (log(max(impact / finish_resistance, 1e-12)) - log(36)) + (
        1 if kd else 0
    )
    finished = bool(rng.random() < _sigmoid(logit))
    term = (
        FightTerminationRequest(event.actor, FinishMethod.KO_TKO) if finished else None
    )
    return StrikeConsequence(True, impact, trauma, p_kd, kd, acute, term)


def advance_physiology(state: FightState, timestamp: float) -> FightState:
    dt = timestamp - state.fight_time_seconds
    if dt < 0:
        raise ValueError("cannot advance physiology backwards")
    decay = exp(-log(2) * dt / 30)

    def one(side):
        row = state.physiology.fighter(side)
        cost = 0.0
        if state.phase is Phase.GROUND and state.ground_controller is not None:
            cost = (0.00025 if state.ground_controller is side else 0.00035) * dt
        return replace(
            row,
            stamina=max(0, row.stamina - cost),
            acute_vulnerability=row.acute_vulnerability * decay,
        )

    return replace(
        state,
        fight_time_seconds=timestamp,
        physiology=FightPhysiology(one(Side.RED), one(Side.BLUE)),
    )


def apply_action_consequence(
    state: FightState, actor: Side, action: ActionFamily, consequence, mechanics
) -> FightState:
    rows = {Side.RED: state.physiology.red, Side.BLUE: state.physiology.blue}
    attacker = rows[actor]
    cost = action_stamina_cost(mechanics, action)
    attacker = replace(attacker, stamina=max(0, attacker.stamina - cost))
    rows[actor] = attacker
    if isinstance(consequence, StrikeConsequence) and consequence.landed:
        target = rows[actor.opponent]
        rows[actor.opponent] = replace(
            target,
            cumulative_trauma=target.cumulative_trauma + consequence.trauma_increment,
            acute_vulnerability=target.acute_vulnerability
            + consequence.acute_increment,
            knockdowns_suffered=target.knockdowns_suffered + int(consequence.knockdown),
        )
    return replace(state, physiology=FightPhysiology(rows[Side.RED], rows[Side.BLUE]))


def action_stamina_cost(mechanics, action=None):
    base = ACTION_COSTS.get(action, 0.0) if action is not None else 0.0
    multiplier = float(
        np.clip(exp(-(mechanics.stamina_depletion_resistance - 50) / 80), 0.65, 1.45)
    )
    return min(1, base * multiplier / mechanics.stamina_capacity)


def recover_round(state):
    def one(row):
        return replace(row, stamina=min(1, row.stamina + (1 - row.stamina) * 0.4))

    return replace(
        state,
        physiology=FightPhysiology(
            one(state.physiology.red), one(state.physiology.blue)
        ),
    )


def _sigmoid(x):
    return 1 / (1 + exp(-float(np.clip(x, -20, 20))))
