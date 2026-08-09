"""Shared transition-probability engine for RFS Monte Carlo V2.

The engine produces one normalized probability distribution shared by both
fighters. It does not sample the transition or generate fight activity.
Dynamic fighter state is intentionally excluded from this module.

Takedowns are modeled in two conceptual stages inside the same end-of-segment
distribution:

1. wrestling propensity creates a takedown-attempt opportunity;
2. matchup conversion resolves that attempt as success or failure.

This keeps one deterministic transition sample per segment while preventing
wrestling skill from creating attempts for fighters who rarely initiate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite, log

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TransitionEvent,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


@dataclass(frozen=True)
class TransitionProbability:
    """One possible shared transition and its probability."""

    event: TransitionEvent
    actor: FighterSide | None
    probability: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError(
                "transition probability must be between 0 and 1"
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

        total_probability = fsum(
            option.probability
            for option in self.options
        )
        if abs(total_probability - 1.0) > 1e-12:
            raise ValueError(
                "transition probabilities must sum to one"
            )

        keys = [
            (option.event, option.actor)
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
        """Return the probability for one transition option."""

        for option in self.options:
            if option.event is event and option.actor is actor:
                return option.probability

        raise KeyError(
            f"transition option not found: "
            f"{event.value}, actor={actor}"
        )


@dataclass(frozen=True)
class DistanceTransitionCalibration:
    """Provisional relative weights for distance transitions.

    ``takedown_base_weight`` is now the total *attempt* opportunity weight.
    Neutral propensity preserves the old total wrestling opportunity mass,
    while ``takedown_success_base_probability`` controls how much of that mass
    becomes a completed takedown versus an explicit failed attempt.
    """

    stay_base_weight: float = 6.0
    clinch_entry_base_weight: float = 1.0
    takedown_base_weight: float = 0.75

    matchup_effect_strength: float = 1.0
    takedown_success_base_probability: float = 0.36
    takedown_success_effect_strength: float = 1.0

    def __post_init__(self) -> None:
        positive_fields = {
            "stay_base_weight": self.stay_base_weight,
            "clinch_entry_base_weight": self.clinch_entry_base_weight,
            "takedown_base_weight": self.takedown_base_weight,
        }
        for name, value in positive_fields.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        nonnegative_fields = {
            "matchup_effect_strength": self.matchup_effect_strength,
            "takedown_success_effect_strength": (
                self.takedown_success_effect_strength
            ),
        }
        for name, value in nonnegative_fields.items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and nonnegative"
                )

        if (
            not isfinite(self.takedown_success_base_probability)
            or not 0.0 < self.takedown_success_base_probability < 1.0
        ):
            raise ValueError(
                "takedown_success_base_probability must be "
                "finite and between zero and one"
            )


@dataclass(frozen=True)
class ClinchTransitionCalibration:
    """Provisional relative weights for clinch transitions.

    Owner and defender takedown base weights now represent attempt opportunity
    mass. Conversion splits each attempt between success and failure.
    """

    stay_base_weight: float = 4.5
    break_base_weight: float = 2.5
    ownership_change_base_weight: float = 1.0
    owner_takedown_base_weight: float = 1.5
    defender_takedown_base_weight: float = 0.5

    matchup_effect_strength: float = 1.0
    takedown_success_base_probability: float = 0.36
    takedown_success_effect_strength: float = 1.0

    def __post_init__(self) -> None:
        positive_fields = {
            "stay_base_weight": self.stay_base_weight,
            "break_base_weight": self.break_base_weight,
            "ownership_change_base_weight": (
                self.ownership_change_base_weight
            ),
            "owner_takedown_base_weight": (
                self.owner_takedown_base_weight
            ),
            "defender_takedown_base_weight": (
                self.defender_takedown_base_weight
            ),
        }
        for name, value in positive_fields.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        nonnegative_fields = {
            "matchup_effect_strength": self.matchup_effect_strength,
            "takedown_success_effect_strength": (
                self.takedown_success_effect_strength
            ),
        }
        for name, value in nonnegative_fields.items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and nonnegative"
                )

        if (
            not isfinite(self.takedown_success_base_probability)
            or not 0.0 < self.takedown_success_base_probability < 1.0
        ):
            raise ValueError(
                "takedown_success_base_probability must be "
                "finite and between zero and one"
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
        positive_fields = {
            "stay_base_weight": self.stay_base_weight,
            "escape_base_weight": self.escape_base_weight,
            "scramble_base_weight": self.scramble_base_weight,
            "reversal_base_weight": self.reversal_base_weight,
        }
        for name, value in positive_fields.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if (
            not isfinite(self.matchup_effect_strength)
            or self.matchup_effect_strength < 0.0
        ):
            raise ValueError(
                "matchup_effect_strength must be finite "
                "and nonnegative"
            )


def _matchup_multiplier(
    score: float,
    *,
    strength: float,
) -> float:
    """Convert a zero-to-one matchup score into a positive multiplier."""

    return exp(
        2.0 * strength * (score - 0.5)
    )


def _probability_from_matchup_score(
    score: float,
    *,
    base_probability: float,
    strength: float,
) -> float:
    """Shift a neutral base probability by a centered matchup score.

    A score of 0.5 returns ``base_probability`` exactly. The transformation is
    performed on log odds so matchup skill changes conversion probability but
    cannot change the total number of attempts created by propensity.
    """

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
        tuple[TransitionEvent, FighterSide | None, float], ...
    ],
) -> TransitionDistribution:
    """Normalize positive/zero relative option weights."""

    total_weight = fsum(
        weight
        for _, _, weight in raw_options
    )
    if not isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(
            "transition option weights must sum to a finite positive value"
        )

    return TransitionDistribution(
        source_phase=source_phase,
        options=tuple(
            TransitionProbability(
                event=event,
                actor=actor,
                probability=weight / total_weight,
            )
            for event, actor, weight in raw_options
        ),
    )


def _distance_stay_score(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
) -> float:
    """Calculate the shared tendency to remain at distance."""

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
    """Calculate one fighter's distance-to-clinch matchup score."""

    return (
        0.45 * attacker.clinch_entry_tendency
        + 0.25 * attacker.phase_imposition
        + 0.20 * (1.0 - defender.clinch_entry_resistance)
        + 0.10 * (1.0 - defender.phase_resistance)
    )


def _takedown_propensity_score(
    attacker: FighterTransitionParameters,
) -> float:
    """Return the attacker's willingness to initiate takedowns."""

    return (
        0.60 * attacker.takedown_entry_tendency
        + 0.25 * attacker.takedown_persistence
        + 0.15 * attacker.failed_takedown_persistence
    )


def _takedown_propensity_multiplier(
    attacker: FighterTransitionParameters,
) -> float:
    """Map neutral propensity to 1x and zero propensity to zero attempts."""

    return 2.0 * _takedown_propensity_score(attacker)


def _takedown_attempt_weight(
    attacker: FighterTransitionParameters,
    *,
    base_weight: float,
) -> float:
    """Return total takedown-attempt opportunity weight."""

    return (
        base_weight
        * _takedown_propensity_multiplier(attacker)
    )


def _takedown_conversion_score(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate conversion quality conditional on an initiated shot."""

    return (
        0.45 * attacker.takedown_completion_ability
        + 0.20 * attacker.phase_imposition
        + 0.25 * (1.0 - defender.takedown_resistance)
        + 0.10 * (1.0 - defender.phase_resistance)
    )


def _distance_takedown_success_probability(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
    calibration: DistanceTransitionCalibration,
) -> float:
    """Resolve one distance-shot attempt into success probability."""

    return _probability_from_matchup_score(
        _takedown_conversion_score(attacker, defender),
        base_probability=(
            calibration.takedown_success_base_probability
        ),
        strength=calibration.takedown_success_effect_strength,
    )


def build_distance_transition_distribution(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    calibration: DistanceTransitionCalibration | None = None,
) -> TransitionDistribution:
    """Build one shared transition distribution from distance.

    A takedown attempt's total probability is determined only by propensity.
    Conversion quality then splits that attempt mass between ``TAKEDOWN`` and
    ``TAKEDOWN_ATTEMPT_FAILED``.
    """

    selected = (
        calibration
        if calibration is not None
        else DistanceTransitionCalibration()
    )
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

    raw_options = (
        (
            TransitionEvent.STAY,
            None,
            selected.stay_base_weight
            * _matchup_multiplier(
                _distance_stay_score(red, blue),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.CLINCH_ENTRY,
            FighterSide.RED,
            selected.clinch_entry_base_weight
            * _matchup_multiplier(
                _clinch_entry_score(red, blue),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.CLINCH_ENTRY,
            FighterSide.BLUE,
            selected.clinch_entry_base_weight
            * _matchup_multiplier(
                _clinch_entry_score(blue, red),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.TAKEDOWN,
            FighterSide.RED,
            red_attempt * red_success,
        ),
        (
            TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
            FighterSide.RED,
            red_attempt * (1.0 - red_success),
        ),
        (
            TransitionEvent.TAKEDOWN,
            FighterSide.BLUE,
            blue_attempt * blue_success,
        ),
        (
            TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
            FighterSide.BLUE,
            blue_attempt * (1.0 - blue_success),
        ),
    )

    return _normalize_options(
        FightPhase.DISTANCE,
        raw_options,
    )


def _clinch_stay_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate the owner's ability to retain the clinch."""

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
    """Calculate the shared tendency to break back to distance."""

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
    """Calculate the defender's ability to take clinch ownership."""

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
    """Calculate clinch conversion quality after initiation."""

    positional_trait = (
        attacker.clinch_retention
        if attacker_is_owner
        else attacker.phase_imposition
    )

    return (
        0.85 * _takedown_conversion_score(attacker, defender)
        + 0.15 * positional_trait
    )


def _clinch_takedown_success_probability(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
    *,
    attacker_is_owner: bool,
    calibration: ClinchTransitionCalibration,
) -> float:
    """Resolve one clinch-shot attempt into success probability."""

    return _probability_from_matchup_score(
        _clinch_takedown_conversion_score(
            attacker,
            defender,
            attacker_is_owner=attacker_is_owner,
        ),
        base_probability=(
            calibration.takedown_success_base_probability
        ),
        strength=calibration.takedown_success_effect_strength,
    )


def build_clinch_transition_distribution(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    current_owner: FighterSide,
    calibration: ClinchTransitionCalibration | None = None,
) -> TransitionDistribution:
    """Build one shared transition distribution from the clinch."""

    selected = (
        calibration
        if calibration is not None
        else ClinchTransitionCalibration()
    )

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
    owner_success = _clinch_takedown_success_probability(
        owner,
        defender,
        attacker_is_owner=True,
        calibration=selected,
    )
    defender_success = _clinch_takedown_success_probability(
        defender,
        owner,
        attacker_is_owner=False,
        calibration=selected,
    )

    raw_options = (
        (
            TransitionEvent.STAY,
            None,
            selected.stay_base_weight
            * _matchup_multiplier(
                _clinch_stay_score(owner, defender),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.CLINCH_BREAK,
            None,
            selected.break_base_weight
            * _matchup_multiplier(
                _clinch_break_score(owner, defender),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.OWNERSHIP_CHANGE,
            defender_side,
            selected.ownership_change_base_weight
            * _matchup_multiplier(
                _clinch_ownership_change_score(owner, defender),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.TAKEDOWN,
            owner_side,
            owner_attempt * owner_success,
        ),
        (
            TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
            owner_side,
            owner_attempt * (1.0 - owner_success),
        ),
        (
            TransitionEvent.TAKEDOWN,
            defender_side,
            defender_attempt * defender_success,
        ),
        (
            TransitionEvent.TAKEDOWN_ATTEMPT_FAILED,
            defender_side,
            defender_attempt * (1.0 - defender_success),
        ),
    )

    return _normalize_options(
        FightPhase.CLINCH,
        raw_options,
    )


def _ground_stay_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate the owner's ability to retain ground control."""

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
    """Calculate the defender's ability to stand up to distance."""

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
    """Calculate the defender's ability to scramble into a clinch."""

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
    """Calculate the defender's ability to reverse ground ownership."""

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
    """Build one shared transition distribution from the ground."""

    selected = (
        calibration
        if calibration is not None
        else GroundTransitionCalibration()
    )

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
    raw_options = (
        (
            TransitionEvent.STAY,
            None,
            selected.stay_base_weight
            * _matchup_multiplier(
                _ground_stay_score(owner, defender),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.GROUND_ESCAPE,
            defender_side,
            selected.escape_base_weight
            * _matchup_multiplier(
                _ground_escape_score(owner, defender),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.SCRAMBLE_TO_CLINCH,
            defender_side,
            selected.scramble_base_weight
            * _matchup_multiplier(
                _ground_scramble_score(owner, defender),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.REVERSAL,
            defender_side,
            selected.reversal_base_weight
            * _matchup_multiplier(
                _ground_reversal_score(owner, defender),
                strength=strength,
            ),
        ),
    )

    return _normalize_options(
        FightPhase.GROUND,
        raw_options,
    )
