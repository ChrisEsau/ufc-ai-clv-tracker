"""Shared transition-probability engine for RFS Monte Carlo V2.

The engine creates one normalized end-of-segment transition distribution.
Takedowns are explicitly modeled as wrestling sequences:

1. style propensity creates a takedown-sequence initiation;
2. matchup skill resolves each attempt as success or failure;
3. failed-shot persistence determines whether another shot follows;
4. one terminal outcome carries the total attempt count for the segment.

This keeps one deterministic transition sample per 30-second segment while
allowing chain wrestlers to generate multiple attempts inside that segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite, log

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TAKEDOWN_EVENTS,
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)

MAX_TAKEDOWN_PROPENSITY_MULTIPLIER = 8.0
MAX_SUPPORTED_CHAIN_ATTEMPTS = 6


@dataclass(frozen=True)
class TransitionProbability:
    """One possible shared transition and its probability.

    ``attempt_count`` is zero for non-wrestling transitions. For simulator-
    generated takedown outcomes it records the number of shots contained in
    the terminal wrestling sequence represented by this option.
    """

    event: TransitionEvent
    actor: FighterSide | None
    probability: float
    attempt_count: int = 0

    def __post_init__(self) -> None:
        if (
            not isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError(
                "transition probability must be between 0 and 1"
            )

        if type(self.attempt_count) is not int:
            raise TypeError("attempt_count must be an integer")
        if self.attempt_count < 0:
            raise ValueError("attempt_count cannot be negative")
        if self.event not in TAKEDOWN_EVENTS and self.attempt_count != 0:
            raise ValueError(
                "attempt_count is only valid for takedown events"
            )

        actor_required = {
            TransitionEvent.CLINCH_ENTRY,
            TransitionEvent.TAKEDOWN,
            TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
            TransitionEvent.OWNERSHIP_CHANGE,
            TransitionEvent.GROUND_ESCAPE,
            TransitionEvent.SCRAMBLE_TO_CLINCH,
            TransitionEvent.REVERSAL,
        }
        actor_forbidden = {
            TransitionEvent.STAY,
            TransitionEvent.CLINCH_BREAK,
        }

        if self.event in actor_required and self.actor is None:
            raise ValueError(
                f"{self.event.value} requires an actor"
            )
        if self.event in actor_forbidden and self.actor is not None:
            raise ValueError(
                f"{self.event.value} cannot have an actor"
            )


@dataclass(frozen=True)
class TransitionDistribution:
    """Normalized transition choices from one shared phase."""

    source_phase: FightPhase
    options: tuple[TransitionProbability, ...]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(
                "transition distribution cannot be empty"
            )

        total = fsum(option.probability for option in self.options)
        if abs(total - 1.0) > 1e-12:
            raise ValueError(
                "transition probabilities must sum to one"
            )

        keys = [
            (option.event, option.actor, option.attempt_count)
            for option in self.options
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "transition distribution contains duplicate options"
            )

    def probability(
        self,
        event: TransitionEvent,
        actor: FighterSide | None,
    ) -> float:
        """Return aggregate probability for an event/actor pair.

        Takedown chains may contain several terminal options distinguished by
        attempt count. Summing preserves the pre-chain public API.
        """

        matching = [
            option.probability
            for option in self.options
            if option.event is event and option.actor is actor
        ]
        if not matching:
            raise KeyError(
                f"transition option not found: {event.value}, actor={actor}"
            )
        return fsum(matching)


def _validate_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{name} must be finite and positive"
        )


def _validate_nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(
            f"{name} must be finite and nonnegative"
        )


def _validate_probability(name: str, value: float) -> None:
    if not isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            f"{name} must be finite and between zero and one"
        )


def _validate_chain_attempts(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= MAX_SUPPORTED_CHAIN_ATTEMPTS:
        raise ValueError(
            f"{name} must be between 1 and {MAX_SUPPORTED_CHAIN_ATTEMPTS}"
        )


@dataclass(frozen=True)
class DistanceTransitionCalibration:
    """Provisional relative weights for distance transitions.

    ``takedown_base_weight`` controls sequence-initiation opportunity mass.
    Success and chain persistence redistribute that mass among terminal
    outcomes; they do not change whether the sequence was initiated.
    """

    stay_base_weight: float = 6.0
    clinch_entry_base_weight: float = 1.0
    takedown_base_weight: float = 0.75
    matchup_effect_strength: float = 1.0
    takedown_success_base_probability: float = 0.36
    takedown_success_effect_strength: float = 1.0
    max_takedown_chain_attempts: int = 4

    def __post_init__(self) -> None:
        _validate_positive("stay_base_weight", self.stay_base_weight)
        _validate_positive(
            "clinch_entry_base_weight",
            self.clinch_entry_base_weight,
        )
        _validate_positive("takedown_base_weight", self.takedown_base_weight)
        _validate_nonnegative(
            "matchup_effect_strength",
            self.matchup_effect_strength,
        )
        _validate_probability(
            "takedown_success_base_probability",
            self.takedown_success_base_probability,
        )
        _validate_nonnegative(
            "takedown_success_effect_strength",
            self.takedown_success_effect_strength,
        )
        _validate_chain_attempts(
            "max_takedown_chain_attempts",
            self.max_takedown_chain_attempts,
        )


@dataclass(frozen=True)
class ClinchTransitionCalibration:
    """Provisional relative weights for clinch transitions."""

    stay_base_weight: float = 4.5
    break_base_weight: float = 2.5
    ownership_change_base_weight: float = 1.0
    owner_takedown_base_weight: float = 1.5
    defender_takedown_base_weight: float = 0.5
    matchup_effect_strength: float = 1.0
    takedown_success_base_probability: float = 0.36
    takedown_success_effect_strength: float = 1.0
    max_takedown_chain_attempts: int = 4

    def __post_init__(self) -> None:
        _validate_positive("stay_base_weight", self.stay_base_weight)
        _validate_positive("break_base_weight", self.break_base_weight)
        _validate_positive(
            "ownership_change_base_weight",
            self.ownership_change_base_weight,
        )
        _validate_positive(
            "owner_takedown_base_weight",
            self.owner_takedown_base_weight,
        )
        _validate_positive(
            "defender_takedown_base_weight",
            self.defender_takedown_base_weight,
        )
        _validate_nonnegative(
            "matchup_effect_strength",
            self.matchup_effect_strength,
        )
        _validate_probability(
            "takedown_success_base_probability",
            self.takedown_success_base_probability,
        )
        _validate_nonnegative(
            "takedown_success_effect_strength",
            self.takedown_success_effect_strength,
        )
        _validate_chain_attempts(
            "max_takedown_chain_attempts",
            self.max_takedown_chain_attempts,
        )


@dataclass(frozen=True)
class GroundTransitionCalibration:
    """Provisional relative weights for ground transitions."""

    stay_base_weight: float = 5.5
    escape_base_weight: float = 2.0
    scramble_base_weight: float = 1.5
    reversal_base_weight: float = 1.0
    matchup_effect_strength: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive("stay_base_weight", self.stay_base_weight)
        _validate_positive("escape_base_weight", self.escape_base_weight)
        _validate_positive("scramble_base_weight", self.scramble_base_weight)
        _validate_positive("reversal_base_weight", self.reversal_base_weight)
        _validate_nonnegative(
            "matchup_effect_strength",
            self.matchup_effect_strength,
        )


def _matchup_multiplier(score: float, *, strength: float) -> float:
    """Map a centered unit-interval score to a positive relative weight."""

    return exp(2.0 * strength * (score - 0.5))


def _probability_from_matchup_score(
    score: float,
    *,
    base_probability: float,
    strength: float,
) -> float:
    """Apply a centered matchup adjustment on the log-odds scale."""

    log_odds = (
        log(base_probability / (1.0 - base_probability))
        + 2.0 * strength * (score - 0.5)
    )
    if log_odds >= 0.0:
        return 1.0 / (1.0 + exp(-log_odds))
    odds = exp(log_odds)
    return odds / (1.0 + odds)


def _normalize_options(
    source_phase: FightPhase,
    raw_options: tuple[
        tuple[TransitionEvent, FighterSide | None, float, int], ...
    ],
) -> TransitionDistribution:
    total = fsum(weight for _, _, weight, _ in raw_options)
    if not isfinite(total) or total <= 0.0:
        raise ValueError(
            "transition option weights must sum to a finite positive value"
        )
    return TransitionDistribution(
        source_phase=source_phase,
        options=tuple(
            TransitionProbability(
                event=event,
                actor=actor,
                probability=weight / total,
                attempt_count=attempt_count,
            )
            for event, actor, weight, attempt_count in raw_options
            if weight > 0.0
        ),
    )


def _distance_stay_score(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
) -> float:
    return (
        red.distance_retention
        + blue.distance_retention
        + red.phase_resistance
        + blue.phase_resistance
    ) / 4.0


def _clinch_entry_score(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    return (
        0.45 * attacker.clinch_entry_tendency
        + 0.25 * attacker.phase_imposition
        + 0.20 * (1.0 - defender.clinch_entry_resistance)
        + 0.10 * (1.0 - defender.phase_resistance)
    )


def _takedown_propensity_score(
    attacker: FighterTransitionParameters,
) -> float:
    """Return initial-shot style.

    The V1.4 style adapter maps raw TD attempts/round as ``t / (t + 1)``.
    The entry tendency therefore represents sequence initiation while the two
    persistence fields are reserved for repeat-shot behavior inside a sequence.
    """

    return attacker.takedown_entry_tendency


def _takedown_propensity_multiplier(
    attacker: FighterTransitionParameters,
) -> float:
    """Convert bounded initial-shot style to an initiation multiplier."""

    propensity = _takedown_propensity_score(attacker)
    if propensity <= 0.0:
        return 0.0
    if propensity >= 1.0:
        return MAX_TAKEDOWN_PROPENSITY_MULTIPLIER
    return min(
        MAX_TAKEDOWN_PROPENSITY_MULTIPLIER,
        propensity / (1.0 - propensity),
    )


def _takedown_attempt_weight(
    attacker: FighterTransitionParameters,
    *,
    base_weight: float,
) -> float:
    return base_weight * _takedown_propensity_multiplier(attacker)


def _takedown_chain_persistence_probability(
    attacker: FighterTransitionParameters,
) -> float:
    """Return probability of another shot after a failed attempt.

    General takedown persistence contributes 35%; explicit failed-shot
    persistence contributes 65%. The cap avoids nearly deterministic six-shot
    chains while retaining strong Merab-like repeat-shot behavior.
    """

    persistence = (
        0.35 * attacker.takedown_persistence
        + 0.65 * attacker.failed_takedown_persistence
    )
    return max(0.0, min(0.95, persistence))


def _takedown_conversion_score(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    return (
        0.45 * attacker.takedown_completion_ability
        + 0.20 * attacker.phase_imposition
        + 0.25 * (1.0 - defender.takedown_resistance)
        + 0.10 * (1.0 - defender.phase_resistance)
    )


def _takedown_chain_options(
    *,
    actor: FighterSide,
    attempt_weight: float,
    success_probability: float,
    persistence_probability: float,
    max_attempts: int,
) -> tuple[tuple[TransitionEvent, FighterSide, float, int], ...]:
    """Expand one initiated wrestling sequence into terminal outcomes.

    Success on attempt ``k`` records ``k`` attempts and enters ground. A chain
    that stops after failure records all attempts made and retains the current
    broad phase. The terminal outcome probabilities sum to one, so the total
    sequence-initiation mass remains exactly ``attempt_weight``.
    """

    if attempt_weight <= 0.0:
        return ()

    outcomes: list[
        tuple[TransitionEvent, FighterSide, float, int]
    ] = []
    reach_probability = 1.0

    for attempt_number in range(1, max_attempts + 1):
        success_terminal = reach_probability * success_probability
        if success_terminal > 0.0:
            outcomes.append(
                (
                    TransitionEvent.TAKEDOWN,
                    actor,
                    attempt_weight * success_terminal,
                    attempt_number,
                )
            )

        failed_this_attempt = (
            reach_probability * (1.0 - success_probability)
        )

        if attempt_number == max_attempts:
            stop_terminal = failed_this_attempt
        else:
            stop_terminal = (
                failed_this_attempt
                * (1.0 - persistence_probability)
            )

        if stop_terminal > 0.0:
            outcomes.append(
                (
                    TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
                    actor,
                    attempt_weight * stop_terminal,
                    attempt_number,
                )
            )

        if attempt_number < max_attempts:
            reach_probability = (
                failed_this_attempt * persistence_probability
            )

    return tuple(outcomes)


def _distance_takedown_success_probability(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
    calibration: DistanceTransitionCalibration,
) -> float:
    return _probability_from_matchup_score(
        _takedown_conversion_score(attacker, defender),
        base_probability=calibration.takedown_success_base_probability,
        strength=calibration.takedown_success_effect_strength,
    )


def build_distance_transition_distribution(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    calibration: DistanceTransitionCalibration | None = None,
) -> TransitionDistribution:
    """Build one shared distance transition distribution."""

    selected = calibration or DistanceTransitionCalibration()
    strength = selected.matchup_effect_strength

    red_attempt = _takedown_attempt_weight(
        red,
        base_weight=selected.takedown_base_weight,
    )
    blue_attempt = _takedown_attempt_weight(
        blue,
        base_weight=selected.takedown_base_weight,
    )
    red_success = _distance_takedown_success_probability(
        red,
        blue,
        selected,
    )
    blue_success = _distance_takedown_success_probability(
        blue,
        red,
        selected,
    )

    base_options: tuple[
        tuple[TransitionEvent, FighterSide | None, float, int], ...
    ] = (
        (
            TransitionEvent.STAY,
            None,
            selected.stay_base_weight
            * _matchup_multiplier(
                _distance_stay_score(red, blue),
                strength=strength,
            ),
            0,
        ),
        (
            TransitionEvent.CLINCH_ENTRY,
            FighterSide.RED,
            selected.clinch_entry_base_weight
            * _matchup_multiplier(
                _clinch_entry_score(red, blue),
                strength=strength,
            ),
            0,
        ),
        (
            TransitionEvent.CLINCH_ENTRY,
            FighterSide.BLUE,
            selected.clinch_entry_base_weight
            * _matchup_multiplier(
                _clinch_entry_score(blue, red),
                strength=strength,
            ),
            0,
        ),
    )

    return _normalize_options(
        FightPhase.DISTANCE,
        (
            *base_options,
            *_takedown_chain_options(
                actor=FighterSide.RED,
                attempt_weight=red_attempt,
                success_probability=red_success,
                persistence_probability=(
                    _takedown_chain_persistence_probability(red)
                ),
                max_attempts=selected.max_takedown_chain_attempts,
            ),
            *_takedown_chain_options(
                actor=FighterSide.BLUE,
                attempt_weight=blue_attempt,
                success_probability=blue_success,
                persistence_probability=(
                    _takedown_chain_persistence_probability(blue)
                ),
                max_attempts=selected.max_takedown_chain_attempts,
            ),
        ),
    )


def _clinch_stay_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    return (
        0.45 * owner.clinch_retention
        + 0.25 * owner.phase_imposition
        + 0.20 * (1.0 - defender.clinch_escape_ability)
        + 0.10 * (1.0 - defender.phase_resistance)
    )


def _clinch_break_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    return (
        0.50 * defender.clinch_escape_ability
        + 0.25 * defender.phase_resistance
        + 0.15 * (1.0 - owner.clinch_retention)
        + 0.10 * (1.0 - owner.phase_imposition)
    )


def _clinch_ownership_change_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    return (
        0.35 * defender.phase_imposition
        + 0.25 * defender.clinch_retention
        + 0.20 * (1.0 - owner.clinch_retention)
        + 0.20 * (1.0 - owner.phase_resistance)
    )


def _clinch_takedown_conversion_score(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
    *,
    attacker_is_owner: bool,
) -> float:
    positional_trait = (
        attacker.clinch_retention
        if attacker_is_owner
        else attacker.phase_imposition
    )
    return (
        0.85 * _takedown_conversion_score(attacker, defender)
        + 0.15 * positional_trait
    )


def _clinch_success_probability(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
    *,
    attacker_is_owner: bool,
    calibration: ClinchTransitionCalibration,
) -> float:
    return _probability_from_matchup_score(
        _clinch_takedown_conversion_score(
            attacker,
            defender,
            attacker_is_owner=attacker_is_owner,
        ),
        base_probability=calibration.takedown_success_base_probability,
        strength=calibration.takedown_success_effect_strength,
    )


def build_clinch_transition_distribution(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    current_owner: FighterSide,
    calibration: ClinchTransitionCalibration | None = None,
) -> TransitionDistribution:
    """Build one shared clinch transition distribution."""

    selected = calibration or ClinchTransitionCalibration()

    if current_owner is FighterSide.RED:
        owner = red
        defender = blue
        owner_side = FighterSide.RED
        defender_side = FighterSide.BLUE
    elif current_owner is FighterSide.BLUE:
        owner = blue
        defender = red
        owner_side = FighterSide.BLUE
        defender_side = FighterSide.RED
    else:
        raise ValueError(
            "current_owner must be red or blue"
        )

    strength = selected.matchup_effect_strength
    owner_attempt = _takedown_attempt_weight(
        owner,
        base_weight=selected.owner_takedown_base_weight,
    )
    defender_attempt = _takedown_attempt_weight(
        defender,
        base_weight=selected.defender_takedown_base_weight,
    )
    owner_success = _clinch_success_probability(
        owner,
        defender,
        attacker_is_owner=True,
        calibration=selected,
    )
    defender_success = _clinch_success_probability(
        defender,
        owner,
        attacker_is_owner=False,
        calibration=selected,
    )

    base_options: tuple[
        tuple[TransitionEvent, FighterSide | None, float, int], ...
    ] = (
        (
            TransitionEvent.STAY,
            None,
            selected.stay_base_weight
            * _matchup_multiplier(
                _clinch_stay_score(owner, defender),
                strength=strength,
            ),
            0,
        ),
        (
            TransitionEvent.CLINCH_BREAK,
            None,
            selected.break_base_weight
            * _matchup_multiplier(
                _clinch_break_score(owner, defender),
                strength=strength,
            ),
            0,
        ),
        (
            TransitionEvent.OWNERSHIP_CHANGE,
            defender_side,
            selected.ownership_change_base_weight
            * _matchup_multiplier(
                _clinch_ownership_change_score(owner, defender),
                strength=strength,
            ),
            0,
        ),
    )

    return _normalize_options(
        FightPhase.CLINCH,
        (
            *base_options,
            *_takedown_chain_options(
                actor=owner_side,
                attempt_weight=owner_attempt,
                success_probability=owner_success,
                persistence_probability=(
                    _takedown_chain_persistence_probability(owner)
                ),
                max_attempts=selected.max_takedown_chain_attempts,
            ),
            *_takedown_chain_options(
                actor=defender_side,
                attempt_weight=defender_attempt,
                success_probability=defender_success,
                persistence_probability=(
                    _takedown_chain_persistence_probability(defender)
                ),
                max_attempts=selected.max_takedown_chain_attempts,
            ),
        ),
    )


def _ground_stay_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    return (
        0.50 * owner.ground_retention
        + 0.25 * owner.phase_imposition
        + 0.15 * (1.0 - defender.ground_escape_ability)
        + 0.10 * (1.0 - defender.reversal_ability)
    )


def _ground_escape_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    return (
        0.50 * defender.ground_escape_ability
        + 0.25 * defender.phase_resistance
        + 0.15 * (1.0 - owner.ground_retention)
        + 0.10 * (1.0 - owner.phase_imposition)
    )


def _ground_scramble_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    return (
        0.35 * defender.ground_escape_ability
        + 0.30 * defender.phase_imposition
        + 0.20 * (1.0 - owner.ground_retention)
        + 0.15 * (1.0 - owner.phase_resistance)
    )


def _ground_reversal_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    return (
        0.50 * defender.reversal_ability
        + 0.20 * defender.phase_imposition
        + 0.15 * (1.0 - owner.ground_retention)
        + 0.15 * (1.0 - owner.phase_resistance)
    )


def build_ground_transition_distribution(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    current_owner: FighterSide,
    calibration: GroundTransitionCalibration | None = None,
) -> TransitionDistribution:
    """Build one shared ground transition distribution."""

    selected = calibration or GroundTransitionCalibration()

    if current_owner is FighterSide.RED:
        owner = red
        defender = blue
        defender_side = FighterSide.BLUE
    elif current_owner is FighterSide.BLUE:
        owner = blue
        defender = red
        defender_side = FighterSide.RED
    else:
        raise ValueError(
            "current_owner must be red or blue"
        )

    strength = selected.matchup_effect_strength
    return _normalize_options(
        FightPhase.GROUND,
        (
            (
                TransitionEvent.STAY,
                None,
                selected.stay_base_weight
                * _matchup_multiplier(
                    _ground_stay_score(owner, defender),
                    strength=strength,
                ),
                0,
            ),
            (
                TransitionEvent.GROUND_ESCAPE,
                defender_side,
                selected.escape_base_weight
                * _matchup_multiplier(
                    _ground_escape_score(owner, defender),
                    strength=strength,
                ),
                0,
            ),
            (
                TransitionEvent.SCRAMBLE_TO_CLINCH,
                defender_side,
                selected.scramble_base_weight
                * _matchup_multiplier(
                    _ground_scramble_score(owner, defender),
                    strength=strength,
                ),
                0,
            ),
            (
                TransitionEvent.REVERSAL,
                defender_side,
                selected.reversal_base_weight
                * _matchup_multiplier(
                    _ground_reversal_score(owner, defender),
                    strength=strength,
                ),
                0,
            ),
        ),
    )
