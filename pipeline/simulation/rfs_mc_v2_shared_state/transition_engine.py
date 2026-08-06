"""Shared transition-probability engine for RFS Monte Carlo V2.

Milestone 2A implements only transitions from the distance phase.

The engine produces one normalized probability distribution shared by both
fighters. It does not sample the transition or generate fight activity.
Dynamic fighter state is intentionally excluded from this milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite

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
    """Uncalibrated starting weights for distance transitions.

    These are relative weights, not final probabilities. Neutral fighter
    parameters produce approximately:

    - 63% remain at distance
    - 11% red clinch entry
    - 11% blue clinch entry
    - 8% red takedown
    - 8% blue takedown

    These defaults are structural starting values and must later be
    calibrated against observed UFC phase transitions.
    """

    stay_base_weight: float = 6.0
    clinch_entry_base_weight: float = 1.0
    takedown_base_weight: float = 0.75

    matchup_effect_strength: float = 1.0

    def __post_init__(self) -> None:
        positive_fields = {
            "stay_base_weight": self.stay_base_weight,
            "clinch_entry_base_weight": (
                self.clinch_entry_base_weight
            ),
            "takedown_base_weight": self.takedown_base_weight,
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
    """Convert a zero-to-one matchup score into a positive multiplier.

    A neutral score of 0.5 produces a multiplier of 1.0.

    At the default strength:

    - score 0.0 produces approximately 0.37
    - score 0.5 produces 1.00
    - score 1.0 produces approximately 2.72
    """

    return exp(
        2.0 * strength * (score - 0.5)
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
        + 0.20 * (
            1.0 - defender.clinch_entry_resistance
        )
        + 0.10 * (
            1.0 - defender.phase_resistance
        )
    )


def _takedown_score(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate one fighter's distance-to-ground matchup score."""

    return (
        0.25 * attacker.takedown_entry_tendency
        + 0.20 * attacker.takedown_completion_ability
        + 0.15 * attacker.takedown_persistence
        + 0.10 * attacker.failed_takedown_persistence
        + 0.15 * attacker.phase_imposition
        + 0.10 * (
            1.0 - defender.takedown_resistance
        )
        + 0.05 * (
            1.0 - defender.phase_resistance
        )
    )


def build_distance_transition_distribution(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    calibration: DistanceTransitionCalibration | None = None,
) -> TransitionDistribution:
    """Build one shared transition distribution from distance.

    Red and blue do not sample phases independently. Their competing
    matchup scores are converted into one normalized distribution.
    """

    selected_calibration = (
        calibration
        if calibration is not None
        else DistanceTransitionCalibration()
    )

    strength = selected_calibration.matchup_effect_strength

    raw_options = (
        (
            TransitionEvent.STAY,
            None,
            selected_calibration.stay_base_weight
            * _matchup_multiplier(
                _distance_stay_score(red, blue),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.CLINCH_ENTRY,
            FighterSide.RED,
            selected_calibration.clinch_entry_base_weight
            * _matchup_multiplier(
                _clinch_entry_score(red, blue),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.CLINCH_ENTRY,
            FighterSide.BLUE,
            selected_calibration.clinch_entry_base_weight
            * _matchup_multiplier(
                _clinch_entry_score(blue, red),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.TAKEDOWN,
            FighterSide.RED,
            selected_calibration.takedown_base_weight
            * _matchup_multiplier(
                _takedown_score(red, blue),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.TAKEDOWN,
            FighterSide.BLUE,
            selected_calibration.takedown_base_weight
            * _matchup_multiplier(
                _takedown_score(blue, red),
                strength=strength,
            ),
        ),
    )

    total_weight = fsum(
        weight
        for _, _, weight in raw_options
    )

    options = tuple(
        TransitionProbability(
            event=event,
            actor=actor,
            probability=weight / total_weight,
        )
        for event, actor, weight in raw_options
    )

    return TransitionDistribution(
        source_phase=FightPhase.DISTANCE,
        options=options,
    )


@dataclass(frozen=True)
class ClinchTransitionCalibration:
    """Provisional relative weights for clinch transitions.

    Neutral fighter parameters with a known clinch owner produce:

    - 45.0% remain in the clinch with the current owner
    - 25.0% break back to distance
    - 10.0% change clinch ownership
    - 15.0% current-owner takedown
    - 5.0% defender takedown

    These are structural starting values, not final UFC calibration.
    """

    stay_base_weight: float = 4.5
    break_base_weight: float = 2.5
    ownership_change_base_weight: float = 1.0
    owner_takedown_base_weight: float = 1.5
    defender_takedown_base_weight: float = 0.5

    matchup_effect_strength: float = 1.0

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

        if (
            not isfinite(self.matchup_effect_strength)
            or self.matchup_effect_strength < 0.0
        ):
            raise ValueError(
                "matchup_effect_strength must be finite "
                "and nonnegative"
            )


def _clinch_stay_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate the owner's ability to retain the clinch."""

    return (
        0.45 * owner.clinch_retention
        + 0.25 * owner.phase_imposition
        + 0.20 * (
            1.0 - defender.clinch_escape_ability
        )
        + 0.10 * (
            1.0 - defender.phase_resistance
        )
    )


def _clinch_break_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate the shared tendency to break back to distance."""

    return (
        0.50 * defender.clinch_escape_ability
        + 0.25 * defender.phase_resistance
        + 0.15 * (
            1.0 - owner.clinch_retention
        )
        + 0.10 * (
            1.0 - owner.phase_imposition
        )
    )


def _clinch_ownership_change_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate the defender's ability to take clinch ownership."""

    return (
        0.35 * defender.phase_imposition
        + 0.25 * defender.clinch_retention
        + 0.20 * (
            1.0 - owner.clinch_retention
        )
        + 0.20 * (
            1.0 - owner.phase_resistance
        )
    )


def _clinch_takedown_score(
    attacker: FighterTransitionParameters,
    defender: FighterTransitionParameters,
    *,
    attacker_is_owner: bool,
) -> float:
    """Calculate a takedown score from the clinch.

    Both fighters use their general takedown matchup score. The current
    owner receives additional value from clinch retention, while the
    defender relies on broad phase-imposition ability.
    """

    positional_trait = (
        attacker.clinch_retention
        if attacker_is_owner
        else attacker.phase_imposition
    )

    return (
        0.85 * _takedown_score(
            attacker,
            defender,
        )
        + 0.15 * positional_trait
    )


def build_clinch_transition_distribution(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    current_owner: FighterSide,
    calibration: ClinchTransitionCalibration | None = None,
) -> TransitionDistribution:
    """Build one shared transition distribution from the clinch."""

    selected_calibration = (
        calibration
        if calibration is not None
        else ClinchTransitionCalibration()
    )

    if current_owner is FighterSide.RED:
        owner = red
        defender = blue
        owner_side = FighterSide.RED
        defender_side = FighterSide.BLUE
    else:
        owner = blue
        defender = red
        owner_side = FighterSide.BLUE
        defender_side = FighterSide.RED

    strength = selected_calibration.matchup_effect_strength

    raw_options = (
        (
            TransitionEvent.STAY,
            None,
            selected_calibration.stay_base_weight
            * _matchup_multiplier(
                _clinch_stay_score(
                    owner,
                    defender,
                ),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.CLINCH_BREAK,
            None,
            selected_calibration.break_base_weight
            * _matchup_multiplier(
                _clinch_break_score(
                    owner,
                    defender,
                ),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.OWNERSHIP_CHANGE,
            defender_side,
            selected_calibration.ownership_change_base_weight
            * _matchup_multiplier(
                _clinch_ownership_change_score(
                    owner,
                    defender,
                ),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.TAKEDOWN,
            owner_side,
            selected_calibration.owner_takedown_base_weight
            * _matchup_multiplier(
                _clinch_takedown_score(
                    owner,
                    defender,
                    attacker_is_owner=True,
                ),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.TAKEDOWN,
            defender_side,
            selected_calibration.defender_takedown_base_weight
            * _matchup_multiplier(
                _clinch_takedown_score(
                    defender,
                    owner,
                    attacker_is_owner=False,
                ),
                strength=strength,
            ),
        ),
    )

    total_weight = fsum(
        weight
        for _, _, weight in raw_options
    )

    options = tuple(
        TransitionProbability(
            event=event,
            actor=actor,
            probability=weight / total_weight,
        )
        for event, actor, weight in raw_options
    )

    return TransitionDistribution(
        source_phase=FightPhase.CLINCH,
        options=options,
    )


@dataclass(frozen=True)
class GroundTransitionCalibration:
    """Provisional relative weights for ground transitions.

    Neutral fighter parameters with a known ground owner produce:

    - 55% remain grounded with the current owner
    - 20% defender escapes to distance
    - 15% defender scrambles into clinch ownership
    - 10% defender reverses ground ownership

    These are structural starting values, not final UFC calibration.
    """

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


def _ground_stay_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate the owner's ability to retain ground control."""

    return (
        0.50 * owner.ground_retention
        + 0.25 * owner.phase_imposition
        + 0.15 * (
            1.0 - defender.ground_escape_ability
        )
        + 0.10 * (
            1.0 - defender.reversal_ability
        )
    )


def _ground_escape_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate the defender's ability to stand up to distance."""

    return (
        0.50 * defender.ground_escape_ability
        + 0.25 * defender.phase_resistance
        + 0.15 * (
            1.0 - owner.ground_retention
        )
        + 0.10 * (
            1.0 - owner.phase_imposition
        )
    )


def _ground_scramble_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate the defender's ability to scramble into a clinch."""

    return (
        0.35 * defender.ground_escape_ability
        + 0.30 * defender.phase_imposition
        + 0.20 * (
            1.0 - owner.ground_retention
        )
        + 0.15 * (
            1.0 - owner.phase_resistance
        )
    )


def _ground_reversal_score(
    owner: FighterTransitionParameters,
    defender: FighterTransitionParameters,
) -> float:
    """Calculate the defender's ability to reverse ground ownership."""

    return (
        0.50 * defender.reversal_ability
        + 0.20 * defender.phase_imposition
        + 0.15 * (
            1.0 - owner.ground_retention
        )
        + 0.15 * (
            1.0 - owner.phase_resistance
        )
    )


def build_ground_transition_distribution(
    red: FighterTransitionParameters,
    blue: FighterTransitionParameters,
    *,
    current_owner: FighterSide,
    calibration: GroundTransitionCalibration | None = None,
) -> TransitionDistribution:
    """Build one shared transition distribution from the ground.

    The current owner may retain the position. All other V2 ground
    transitions are defensive actions by the current non-owner.
    """

    selected_calibration = (
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

    strength = selected_calibration.matchup_effect_strength

    raw_options = (
        (
            TransitionEvent.STAY,
            None,
            selected_calibration.stay_base_weight
            * _matchup_multiplier(
                _ground_stay_score(
                    owner,
                    defender,
                ),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.GROUND_ESCAPE,
            defender_side,
            selected_calibration.escape_base_weight
            * _matchup_multiplier(
                _ground_escape_score(
                    owner,
                    defender,
                ),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.SCRAMBLE_TO_CLINCH,
            defender_side,
            selected_calibration.scramble_base_weight
            * _matchup_multiplier(
                _ground_scramble_score(
                    owner,
                    defender,
                ),
                strength=strength,
            ),
        ),
        (
            TransitionEvent.REVERSAL,
            defender_side,
            selected_calibration.reversal_base_weight
            * _matchup_multiplier(
                _ground_reversal_score(
                    owner,
                    defender,
                ),
                strength=strength,
            ),
        ),
    )

    total_weight = fsum(
        weight
        for _, _, weight in raw_options
    )

    options = tuple(
        TransitionProbability(
            event=event,
            actor=actor,
            probability=weight / total_weight,
        )
        for event, actor, weight in raw_options
    )

    return TransitionDistribution(
        source_phase=FightPhase.GROUND,
        options=options,
    )
