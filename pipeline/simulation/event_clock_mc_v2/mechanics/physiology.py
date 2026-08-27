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
from .config import (
    DEFAULT_MECHANICS_CALIBRATION_CONFIG,
    MechanicsCalibrationConfig,
    MechanicsInputs,
    KOKDArchitecture,
)
from .resolution import FightTerminationRequest, FinishMethod, StrikeConsequence
from .ko_kd_empirical import resolve_landed_strike as resolve_empirical_ko_kd

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

# Research-only bridge from the empirical Event2 KD result into the Brain's
# existing decaying hurt context.  Legacy Stage 10 already assigned 0.5 acute
# vulnerability on a KD; reusing that established state magnitude lets a KD
# activate follow-up behavior without changing the empirical KO/KD hazards.
EMPIRICAL_KD_HURT_ACUTE_INCREMENT = 0.5


def resolve_landed_strike(
    event,
    state: FightState,
    inputs: MechanicsInputs,
    landed: bool,
    rng: np.random.Generator,
    config: MechanicsCalibrationConfig = DEFAULT_MECHANICS_CALIBRATION_CONFIG,
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
    impact = max(1e-9, power * severity * config.impact_scale)
    trauma = impact * exp(
        -(defender.damage_durability - 50) / config.trauma_durability_divisor
    )
    new_trauma = target.cumulative_trauma + trauma
    resistance = max(
        1e-6,
        exp((defender.knockdown_resistance - 50) / 32)
        * exp(-new_trauma / 80)
        * exp(-target.acute_vulnerability),
    )
    p_kd = _sigmoid(
        config.kd_slope * (log(impact / resistance) - log(config.kd_midpoint))
    )
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
    logit = config.finish_slope * (
        log(max(impact / finish_resistance, 1e-12)) - log(config.finish_midpoint)
    ) + (config.post_kd_finish_logit_bonus if kd else 0)
    p_ko = _sigmoid(logit)
    finished = bool(rng.random() < p_ko)
    term = (
        FightTerminationRequest(event.actor, FinishMethod.KO_TKO) if finished else None
    )
    return StrikeConsequence(
        landed=True,
        impact=impact,
        trauma_increment=trauma,
        knockdown_probability=p_kd,
        knockdown=kd,
        acute_increment=acute,
        termination=term,
        ko_probability=p_ko,
        prior_defender_kds=target.knockdowns_suffered,
        ko_kd_architecture=KOKDArchitecture.LEGACY_STAGE10.value,
    )


def resolve_strike_consequence(
    event,
    state: FightState,
    inputs: MechanicsInputs,
    landed: bool,
    ko_kd_rng: np.random.Generator,
    config: MechanicsCalibrationConfig = DEFAULT_MECHANICS_CALIBRATION_CONFIG,
) -> StrikeConsequence:
    """Dispatch only the KO/KD consequence architecture at the strike seam."""
    if not landed:
        return StrikeConsequence(False)
    # Direct resolver callers that predate the engine selector retain legacy
    # behavior. EngineInputs explicitly selects the production architecture.
    architecture = inputs.ko_kd_architecture or KOKDArchitecture.LEGACY_STAGE10
    if architecture is KOKDArchitecture.LEGACY_STAGE10:
        return resolve_landed_strike(event, state, inputs, landed, ko_kd_rng, config)
    if architecture is not KOKDArchitecture.EMPIRICAL_EVENT2:
        raise ValueError(f"unsupported KO/KD architecture: {architecture}")
    result = resolve_empirical_ko_kd(
        state=state,
        attacker_side=event.actor,
        attacker=inputs.fighter(event.actor),
        defender=inputs.fighter(event.actor.opponent),
        rng=ko_kd_rng,
    )
    return StrikeConsequence(
        landed=True,
        knockdown_probability=result.kd_probability,
        knockdown=result.knockdown,
        # Event2 previously incremented the cumulative KD counter but left
        # acute_vulnerability at zero.  Brain derives own_hurt/opponent_hurt
        # from acute_vulnerability, so it could not react to a knockdown.
        # Feed the KD into that existing state; advance_physiology already
        # decays it with a 30-second half-life.
        acute_increment=(
            EMPIRICAL_KD_HURT_ACUTE_INCREMENT if result.knockdown else 0.0
        ),
        termination=(
            FightTerminationRequest(event.actor, FinishMethod.KO_TKO)
            if result.ko_tko
            else None
        ),
        ko_probability=result.ko_probability,
        prior_defender_kds=result.prior_defender_kds,
        ko_kd_architecture=KOKDArchitecture.EMPIRICAL_EVENT2.value,
    )


def advance_physiology(
    state: FightState,
    timestamp: float,
    config: MechanicsCalibrationConfig = DEFAULT_MECHANICS_CALIBRATION_CONFIG,
) -> FightState:
    dt = timestamp - state.fight_time_seconds
    if dt < 0:
        raise ValueError("cannot advance physiology backwards")
    decay = exp(-log(2) * dt / 30)

    def one(side):
        row = state.physiology.fighter(side)
        cost = 0.0
        if state.phase is Phase.GROUND and state.ground_controller is not None:
            cost = (
                config.top_position_cost_per_second
                if state.ground_controller is side
                else config.bottom_position_cost_per_second
            ) * dt
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
    state: FightState,
    actor: Side,
    action: ActionFamily,
    consequence,
    mechanics,
    config: MechanicsCalibrationConfig = DEFAULT_MECHANICS_CALIBRATION_CONFIG,
) -> FightState:
    rows = {Side.RED: state.physiology.red, Side.BLUE: state.physiology.blue}
    attacker = rows[actor]
    cost = action_stamina_cost(mechanics, action, config)
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


def action_stamina_cost(
    mechanics,
    action=None,
    config: MechanicsCalibrationConfig = DEFAULT_MECHANICS_CALIBRATION_CONFIG,
):
    base = ACTION_COSTS.get(action, 0.0) if action is not None else 0.0
    multiplier = float(
        np.clip(exp(-(mechanics.stamina_depletion_resistance - 50) / 80), 0.65, 1.45)
    )
    return min(
        1, base * multiplier / mechanics.stamina_capacity * config.action_cost_scale
    )


def recover_round(
    state, config: MechanicsCalibrationConfig = DEFAULT_MECHANICS_CALIBRATION_CONFIG
):
    def one(row):
        return replace(
            row,
            stamina=min(
                1, row.stamina + (1 - row.stamina) * config.round_recovery_fraction
            ),
        )

    return replace(
        state,
        physiology=FightPhysiology(
            one(state.physiology.red), one(state.physiology.blue)
        ),
    )


def _sigmoid(x):
    return 1 / (1 + exp(-float(np.clip(x, -20, 20))))