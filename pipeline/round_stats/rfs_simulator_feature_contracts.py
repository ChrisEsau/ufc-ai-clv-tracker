"""Contracts for mapping Round Fighter State into Monte Carlo V2.

This module defines:

- the four authoritative simulator-oriented RFS feature families
- the value semantics required by simulator parameters
- the complete registry of 37 fighter simulator targets
- the standard reliability-shrunk estimate envelope

It intentionally contains no feature calculations or calibration formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SimulatorFeatureFamily(str, Enum):
    """Authoritative simulator-oriented RFS feature families."""

    PHASE_BASELINE = "phase_baseline"
    PHASE_INTERACTION = "phase_interaction"
    DYNAMIC_RESPONSE = "dynamic_response"
    FINISH_STATE = "finish_state"


class SimulatorValueKind(str, Enum):
    """Target value semantics expected by Monte Carlo V2."""

    UNIT_INTERVAL_STRENGTH = "unit_interval_strength"
    PROBABILITY = "probability"
    EXPECTED_EVENTS_PER_30_SECONDS = (
        "expected_events_per_30_seconds"
    )
    EXPECTED_CONTROL_SECONDS_PER_30_SECONDS = (
        "expected_control_seconds_per_30_seconds"
    )


class MappingSupportLevel(str, Enum):
    """How directly one simulator parameter is supported by source data."""

    DIRECT_OBSERVED = "direct_observed"
    DERIVED_OBSERVED = "derived_observed"
    LATENT_CALIBRATED = "latent_calibrated"


@dataclass(frozen=True)
class SimulatorTargetSpec:
    """Definition of one fighter-level simulator target."""

    target_parameter: str
    primary_family: SimulatorFeatureFamily
    value_kind: SimulatorValueKind
    support_level: MappingSupportLevel
    description: str

    def __post_init__(self) -> None:
        """Validate one target definition."""

        if not isinstance(self.target_parameter, str):
            raise TypeError(
                "target_parameter must be a string"
            )

        if not self.target_parameter.strip():
            raise ValueError(
                "target_parameter cannot be empty"
            )

        if not isinstance(
            self.primary_family,
            SimulatorFeatureFamily,
        ):
            raise TypeError(
                "primary_family must be "
                "SimulatorFeatureFamily"
            )

        if not isinstance(
            self.value_kind,
            SimulatorValueKind,
        ):
            raise TypeError(
                "value_kind must be SimulatorValueKind"
            )

        if not isinstance(
            self.support_level,
            MappingSupportLevel,
        ):
            raise TypeError(
                "support_level must be MappingSupportLevel"
            )

        if not isinstance(self.description, str):
            raise TypeError(
                "description must be a string"
            )

        if not self.description.strip():
            raise ValueError(
                "description cannot be empty"
            )


@dataclass(frozen=True)
class ReliabilityShrunkEstimate:
    """One auditable fighter-state estimate.

    Attributes:
        raw_estimate:
            Estimate calculated from available prior observations. It may be
            None when no usable fighter evidence exists.

        population_prior:
            Population or hierarchical-group prior used for shrinkage.

        sample_size:
            Integer count of qualifying historical observations.

        effective_sample_size:
            Weighted sample amount after recency or exposure weighting.

        reliability:
            Weight assigned to fighter-specific evidence in [0, 1].

        shrunk_estimate:
            Final estimate after combining fighter evidence and prior.

        used_fallback:
            Whether the output depended entirely on a fallback prior.

        source_columns:
            Raw or intermediate columns supporting the estimate.
    """

    raw_estimate: float | None
    population_prior: float
    sample_size: int
    effective_sample_size: float
    reliability: float
    shrunk_estimate: float
    used_fallback: bool
    source_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate estimate metadata and numerical integrity."""

        if (
            self.raw_estimate is not None
            and (
                not isinstance(
                    self.raw_estimate,
                    (int, float),
                )
                or not math.isfinite(
                    float(self.raw_estimate)
                )
            )
        ):
            raise ValueError(
                "raw_estimate must be finite or None"
            )

        numeric_fields = {
            "population_prior": self.population_prior,
            "effective_sample_size": (
                self.effective_sample_size
            ),
            "reliability": self.reliability,
            "shrunk_estimate": self.shrunk_estimate,
        }

        for name, value in numeric_fields.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be numeric"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite"
                )

        if type(self.sample_size) is not int:
            raise TypeError(
                "sample_size must be an integer"
            )

        if self.sample_size < 0:
            raise ValueError(
                "sample_size cannot be negative"
            )

        if self.effective_sample_size < 0.0:
            raise ValueError(
                "effective_sample_size cannot be negative"
            )

        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(
                "reliability must be between 0 and 1"
            )

        if type(self.used_fallback) is not bool:
            raise TypeError(
                "used_fallback must be boolean"
            )

        if not isinstance(self.source_columns, tuple):
            raise TypeError(
                "source_columns must be a tuple"
            )

        for column in self.source_columns:
            if not isinstance(column, str):
                raise TypeError(
                    "source_columns must contain strings"
                )

            if not column.strip():
                raise ValueError(
                    "source_columns cannot contain "
                    "empty names"
                )

        if self.raw_estimate is None and not self.used_fallback:
            raise ValueError(
                "missing raw_estimate requires "
                "used_fallback=True"
            )


UNIT = SimulatorValueKind.UNIT_INTERVAL_STRENGTH
PROBABILITY = SimulatorValueKind.PROBABILITY
EVENT_RATE = (
    SimulatorValueKind.EXPECTED_EVENTS_PER_30_SECONDS
)
CONTROL_SECONDS = (
    SimulatorValueKind
    .EXPECTED_CONTROL_SECONDS_PER_30_SECONDS
)

PHASE_BASELINE = (
    SimulatorFeatureFamily.PHASE_BASELINE
)
PHASE_INTERACTION = (
    SimulatorFeatureFamily.PHASE_INTERACTION
)
DYNAMIC_RESPONSE = (
    SimulatorFeatureFamily.DYNAMIC_RESPONSE
)
FINISH_STATE = (
    SimulatorFeatureFamily.FINISH_STATE
)

DIRECT = MappingSupportLevel.DIRECT_OBSERVED
DERIVED = MappingSupportLevel.DERIVED_OBSERVED
LATENT = MappingSupportLevel.LATENT_CALIBRATED


SIMULATOR_TARGET_SPECS: tuple[
    SimulatorTargetSpec,
    ...,
] = (
    # ------------------------------------------------------------------
    # Transition strengths
    # ------------------------------------------------------------------
    SimulatorTargetSpec(
        "transition.distance_retention",
        PHASE_INTERACTION,
        UNIT,
        LATENT,
        "Ability to preserve distance against competing phase pressure.",
    ),
    SimulatorTargetSpec(
        "transition.clinch_entry_tendency",
        PHASE_BASELINE,
        UNIT,
        LATENT,
        "Latent tendency to initiate or create clinch occupancy.",
    ),
    SimulatorTargetSpec(
        "transition.clinch_entry_resistance",
        PHASE_INTERACTION,
        UNIT,
        LATENT,
        "Ability to prevent an opponent from establishing a clinch.",
    ),
    SimulatorTargetSpec(
        "transition.takedown_entry_tendency",
        PHASE_BASELINE,
        UNIT,
        DERIVED,
        "Normalized tendency to initiate takedown sequences.",
    ),
    SimulatorTargetSpec(
        "transition.takedown_completion_ability",
        PHASE_BASELINE,
        UNIT,
        DIRECT,
        "Ability to convert takedown attempts into completed takedowns.",
    ),
    SimulatorTargetSpec(
        "transition.takedown_resistance",
        PHASE_INTERACTION,
        UNIT,
        DIRECT,
        "Ability to prevent opponent takedown attempts from succeeding.",
    ),
    SimulatorTargetSpec(
        "transition.takedown_persistence",
        PHASE_BASELINE,
        UNIT,
        LATENT,
        "Likelihood of maintaining takedown pressure across opportunities.",
    ),
    SimulatorTargetSpec(
        "transition.failed_takedown_persistence",
        PHASE_BASELINE,
        UNIT,
        LATENT,
        "Likelihood of continuing wrestling pressure after failed attempts.",
    ),
    SimulatorTargetSpec(
        "transition.clinch_retention",
        PHASE_INTERACTION,
        UNIT,
        LATENT,
        "Ability to keep ownership of an established clinch.",
    ),
    SimulatorTargetSpec(
        "transition.clinch_escape_ability",
        PHASE_INTERACTION,
        UNIT,
        LATENT,
        "Ability to return to distance from an opponent-owned clinch.",
    ),
    SimulatorTargetSpec(
        "transition.ground_retention",
        PHASE_INTERACTION,
        UNIT,
        LATENT,
        "Ability to maintain ground ownership and prevent phase exit.",
    ),
    SimulatorTargetSpec(
        "transition.ground_escape_ability",
        PHASE_INTERACTION,
        UNIT,
        DERIVED,
        "Ability to escape opponent ground control.",
    ),
    SimulatorTargetSpec(
        "transition.reversal_ability",
        PHASE_INTERACTION,
        UNIT,
        DIRECT,
        "Ability to reverse ground ownership.",
    ),
    SimulatorTargetSpec(
        "transition.phase_imposition",
        PHASE_INTERACTION,
        UNIT,
        LATENT,
        "Broad ability to make the fight occur in preferred phases.",
    ),
    SimulatorTargetSpec(
        "transition.phase_resistance",
        PHASE_INTERACTION,
        UNIT,
        LATENT,
        "Broad ability to resist the opponent's preferred phases.",
    ),

    # ------------------------------------------------------------------
    # Phase-specific activity
    # ------------------------------------------------------------------
    SimulatorTargetSpec(
        "phase.distance.sig_strike_attempt_rate",
        PHASE_BASELINE,
        EVENT_RATE,
        DERIVED,
        "Expected significant-strike attempts per occupied distance segment.",
    ),
    SimulatorTargetSpec(
        "phase.distance.sig_strike_accuracy",
        PHASE_INTERACTION,
        PROBABILITY,
        DIRECT,
        "Expected significant-strike accuracy while at distance.",
    ),
    SimulatorTargetSpec(
        "phase.distance.knockdown_probability_per_landed",
        FINISH_STATE,
        PROBABILITY,
        DERIVED,
        "Knockdown probability per landed distance significant strike.",
    ),
    SimulatorTargetSpec(
        "phase.clinch.clinch_strike_attempt_rate",
        PHASE_BASELINE,
        EVENT_RATE,
        DERIVED,
        "Expected significant-strike attempts per owned clinch segment.",
    ),
    SimulatorTargetSpec(
        "phase.clinch.clinch_strike_accuracy",
        PHASE_INTERACTION,
        PROBABILITY,
        DIRECT,
        "Expected significant-strike accuracy while clinching.",
    ),
    SimulatorTargetSpec(
        "phase.clinch.control_seconds_mean",
        PHASE_BASELINE,
        CONTROL_SECONDS,
        LATENT,
        "Expected controlled seconds in an owned clinch segment.",
    ),
    SimulatorTargetSpec(
        "phase.clinch.damaging_clinch_probability",
        FINISH_STATE,
        PROBABILITY,
        LATENT,
        "Probability that an owned clinch segment creates damaging adversity.",
    ),
    SimulatorTargetSpec(
        "phase.ground_owner.ground_strike_attempt_rate",
        PHASE_BASELINE,
        EVENT_RATE,
        DERIVED,
        "Expected ground significant-strike attempts per owned ground segment.",
    ),
    SimulatorTargetSpec(
        "phase.ground_owner.ground_strike_accuracy",
        PHASE_INTERACTION,
        PROBABILITY,
        DIRECT,
        "Expected significant-strike accuracy while owning ground position.",
    ),
    SimulatorTargetSpec(
        "phase.ground_owner.control_seconds_mean",
        PHASE_BASELINE,
        CONTROL_SECONDS,
        LATENT,
        "Expected controlled seconds in an owned ground segment.",
    ),
    SimulatorTargetSpec(
        "phase.ground_owner.submission_attempt_rate",
        FINISH_STATE,
        EVENT_RATE,
        DERIVED,
        "Expected submission attempts per owned ground segment.",
    ),
    SimulatorTargetSpec(
        "phase.ground_owner.position_advancement_probability",
        PHASE_INTERACTION,
        PROBABILITY,
        LATENT,
        "Probability of improving authoritative ground position.",
    ),
    SimulatorTargetSpec(
        "phase.ground_defender.escape_attempt_rate",
        PHASE_INTERACTION,
        EVENT_RATE,
        LATENT,
        "Expected ground escape attempts per defended ground segment.",
    ),
    SimulatorTargetSpec(
        "phase.ground_defender.reversal_attempt_rate",
        PHASE_INTERACTION,
        EVENT_RATE,
        DERIVED,
        "Expected reversal attempts per defended ground segment.",
    ),
    SimulatorTargetSpec(
        "phase.ground_defender.scramble_attempt_rate",
        PHASE_INTERACTION,
        EVENT_RATE,
        LATENT,
        "Expected scramble attempts per defended ground segment.",
    ),
    SimulatorTargetSpec(
        "phase.ground_defender.submission_defense",
        FINISH_STATE,
        PROBABILITY,
        LATENT,
        "Ability to prevent opponent submission attempts from converting.",
    ),

    # ------------------------------------------------------------------
    # Dynamic response traits
    # ------------------------------------------------------------------
    SimulatorTargetSpec(
        "dynamic.fatigue_accumulation_resistance",
        DYNAMIC_RESPONSE,
        UNIT,
        DERIVED,
        "Resistance to accumulating fatigue under equivalent workload.",
    ),
    SimulatorTargetSpec(
        "dynamic.fatigue_performance_resilience",
        DYNAMIC_RESPONSE,
        UNIT,
        DERIVED,
        "Ability to preserve output after fatigue accumulates.",
    ),
    SimulatorTargetSpec(
        "dynamic.recovery_ability",
        DYNAMIC_RESPONSE,
        UNIT,
        LATENT,
        "Ability to recover fatigue during lower workload and round breaks.",
    ),
    SimulatorTargetSpec(
        "dynamic.damage_resistance",
        FINISH_STATE,
        UNIT,
        LATENT,
        "Resistance to persistent damage from equivalent damaging events.",
    ),
    SimulatorTargetSpec(
        "dynamic.acute_stress_resistance",
        FINISH_STATE,
        UNIT,
        LATENT,
        "Resistance to immediate impairment following adversity.",
    ),
    SimulatorTargetSpec(
        "dynamic.acute_stress_recovery",
        DYNAMIC_RESPONSE,
        UNIT,
        LATENT,
        "Ability to recover from temporary adversity across later segments.",
    ),
)


SIMULATOR_TARGET_BY_NAME = {
    spec.target_parameter: spec
    for spec in SIMULATOR_TARGET_SPECS
}


def validate_simulator_target_registry() -> None:
    """Validate authoritative target count and unique names."""

    if len(SIMULATOR_TARGET_SPECS) != 37:
        raise RuntimeError(
            "simulator target registry must contain "
            "exactly 37 targets"
        )

    if (
        len(SIMULATOR_TARGET_BY_NAME)
        != len(SIMULATOR_TARGET_SPECS)
    ):
        raise RuntimeError(
            "simulator target registry contains "
            "duplicate parameter names"
        )


validate_simulator_target_registry()
