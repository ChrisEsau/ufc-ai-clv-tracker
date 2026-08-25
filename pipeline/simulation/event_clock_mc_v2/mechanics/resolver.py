"""Explicit dispatch from legal action attempts to immutable mechanics results."""

from __future__ import annotations

import numpy as np

from pipeline.simulation.event_clock_mc_v2.causal.events import (
    ActionEvent,
    ActionFamily,
)
from pipeline.simulation.event_clock_mc_v2.causal.legality import validate_action_event
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase

from .config import (
    DEFAULT_MECHANICS_CALIBRATION_CONFIG,
    MechanicsCalibrationConfig,
    MechanicsInputs,
    StructuralMVPPlaceholders,
)
from .resolution import (
    ActionOutcome,
    ActionResolution,
    FinishMethod,
    FightTerminationRequest,
    SubmissionConsequence,
    StrikeConsequence,
    TransitionKind,
    TransitionRequest,
)
from .physiology import resolve_strike_consequence
from .submission import resolve_submission


def resolve_action(
    event: ActionEvent,
    state: FightState,
    inputs: MechanicsInputs,
    rng: np.random.Generator,
    placeholders: StructuralMVPPlaceholders = StructuralMVPPlaceholders(),
    ko_kd_rng: np.random.Generator | None = None,
    submission_rng: np.random.Generator | None = None,
) -> ActionResolution:
    """Resolve one legal attempt without mutating authoritative state or timeline."""
    validate_action_event(event, state)
    if not isinstance(inputs, MechanicsInputs):
        raise ValueError("inputs must be MechanicsInputs")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    if not isinstance(placeholders, StructuralMVPPlaceholders):
        raise ValueError("placeholders must be StructuralMVPPlaceholders")
    if ko_kd_rng is None:
        ko_kd_rng = rng
    if not isinstance(ko_kd_rng, np.random.Generator):
        raise ValueError("ko_kd_rng must be a numpy.random.Generator")
    submission_rng = rng if submission_rng is None else submission_rng
    if not isinstance(submission_rng, np.random.Generator):
        raise ValueError("submission_rng must be a numpy.random.Generator")

    family = event.action_family
    fighter = inputs.fighter(event.actor)
    calibration = inputs.calibration or DEFAULT_MECHANICS_CALIBRATION_CONFIG

    if family in {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}:
        return _strike(event, state, inputs, fighter.standing_strike_landing_probability, rng, calibration, ko_kd_rng)
    if family is ActionFamily.PRESSURE or family is ActionFamily.RESET_RANGE:
        return ActionResolution(event, ActionOutcome.TACTICAL)
    if family is ActionFamily.CLINCH_ENTRY:
        return _binary_transition(
            event,
            placeholders.clinch_entry_success_probability,
            rng,
            success_outcome=ActionOutcome.SUCCESS,
            failure_outcome=ActionOutcome.FAILURE,
            transition=TransitionRequest(TransitionKind.ENTER_CLINCH, Phase.STANDING, Phase.CLINCH, event.actor),
        )
    if family is ActionFamily.TAKEDOWN_ENTRY:
        return _binary_transition(
            event,
            fighter.takedown_completion_probability,
            rng,
            success_outcome=ActionOutcome.SUCCESS,
            failure_outcome=ActionOutcome.STUFFED,
            transition=TransitionRequest(TransitionKind.DIRECT_TAKEDOWN, Phase.STANDING, Phase.GROUND, event.actor),
        )
    if family is ActionFamily.CLINCH_STRIKE:
        return _strike(event, state, inputs, placeholders.clinch_strike_landing_probability, rng, calibration, ko_kd_rng)
    if family is ActionFamily.CLINCH_CONTROL:
        return ActionResolution(event, ActionOutcome.CONTROLLED)
    if family is ActionFamily.CLINCH_TAKEDOWN:
        return _binary_transition(
            event,
            fighter.takedown_completion_probability,
            rng,
            success_outcome=ActionOutcome.SUCCESS,
            failure_outcome=ActionOutcome.STUFFED,
            transition=TransitionRequest(TransitionKind.CLINCH_TAKEDOWN, Phase.CLINCH, Phase.GROUND, event.actor),
        )
    if family is ActionFamily.BREAK_CLINCH:
        return _binary_transition(
            event,
            placeholders.break_clinch_success_probability,
            rng,
            success_outcome=ActionOutcome.SEPARATED,
            failure_outcome=ActionOutcome.FAILURE,
            transition=TransitionRequest(TransitionKind.BREAK_CLINCH, Phase.CLINCH, Phase.STANDING),
        )
    if family in {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}:
        return _strike(event, state, inputs, fighter.ground_strike_landing_probability, rng, calibration, ko_kd_rng)
    if family in {ActionFamily.ADVANCE_POSITION, ActionFamily.IMPROVE_POSITION}:
        return ActionResolution(event, ActionOutcome.MAINTAINED)
    if family is ActionFamily.SUBMISSION_ATTACK:
        defender = inputs.fighter(event.actor.opponent)
        probability, succeeded = resolve_submission(
            fighter.submission_conversion_baseline,
            fighter.submission_conversion_offset,
            submission_rng,
            attacker_offense=fighter.submission_offense,
            defender_defense=defender.submission_defense,
        )
        return ActionResolution(
            event,
            ActionOutcome.SUCCESS if succeeded else ActionOutcome.FAILURE,
            consequence=SubmissionConsequence(
                attempted=True,
                conversion_probability=probability,
                success=succeeded,
                termination=FightTerminationRequest(event.actor, FinishMethod.SUBMISSION) if succeeded else None,
            ),
        )
    if family is ActionFamily.CONTROL:
        return ActionResolution(event, ActionOutcome.CONTROLLED)
    if family is ActionFamily.DISENGAGE:
        return ActionResolution(
            event,
            ActionOutcome.ESCAPED,
            TransitionRequest(TransitionKind.DISENGAGE_GROUND, Phase.GROUND, Phase.STANDING),
        )
    if family is ActionFamily.ESCAPE_STAND:
        return _binary_transition(
            event,
            fighter.ground_escape_probability,
            rng,
            success_outcome=ActionOutcome.ESCAPED,
            failure_outcome=ActionOutcome.FAILURE,
            transition=TransitionRequest(TransitionKind.ESCAPE_GROUND, Phase.GROUND, Phase.STANDING),
        )
    if family is ActionFamily.REVERSAL:
        return _binary_transition(
            event,
            fighter.ground_reversal_probability,
            rng,
            success_outcome=ActionOutcome.REVERSED,
            failure_outcome=ActionOutcome.FAILURE,
            transition=TransitionRequest(TransitionKind.REVERSE_GROUND, Phase.GROUND, Phase.GROUND, event.actor),
        )
    raise ValueError(f"no mechanics resolver for {family.value}")


def _strike(
    event: ActionEvent,
    state: FightState,
    inputs: MechanicsInputs,
    probability: float,
    rng: np.random.Generator,
    calibration: MechanicsCalibrationConfig = DEFAULT_MECHANICS_CALIBRATION_CONFIG,
    ko_kd_rng: np.random.Generator | None = None,
) -> ActionResolution:
    landed = _succeeds(probability, rng)
    return ActionResolution(
        event,
        ActionOutcome.LANDED if landed else ActionOutcome.MISSED,
        consequence=resolve_strike_consequence(event, state, inputs, landed, ko_kd_rng or rng, calibration),
    )


def _binary_transition(
    event: ActionEvent,
    probability: float,
    rng: np.random.Generator,
    *,
    success_outcome: ActionOutcome,
    failure_outcome: ActionOutcome,
    transition: TransitionRequest,
) -> ActionResolution:
    succeeded = _succeeds(probability, rng)
    return ActionResolution(
        event,
        success_outcome if succeeded else failure_outcome,
        transition if succeeded else None,
    )


def _succeeds(probability: float, rng: np.random.Generator) -> bool:
    return bool(rng.random() < probability)
