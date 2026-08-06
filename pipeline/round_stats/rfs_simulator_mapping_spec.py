"""Authoritative source-to-target mapping specification for RFS MC V2.

This module declares how the leakage-safe historical Round Fighter State
pipeline will eventually estimate all 37 fighter-specific simulator inputs.

It contains specifications only. It does not calculate fighter features,
fit calibration coefficients, create parquet artifacts, or alter production
feature views.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.round_stats.rfs_simulator_feature_contracts import (
    MappingSupportLevel,
    SIMULATOR_TARGET_BY_NAME,
)


AUTHORITATIVE_ROUND_STATS_PATH = (
    "data/fight_details/ufc_round_stats.parquet"
)

AUTHORITATIVE_ROUND_SOURCE_COLUMNS = frozenset(
    {
        "event_id",
        "event_name",
        "event_date",
        "fight_id",
        "fight_url",
        "event_url",
        "location",
        "division",
        "title_fight",
        "total_rounds",
        "fight_order",
        "corner",
        "fighter_name",
        "fighter_id",
        "fighter_url",
        "opponent_name",
        "opponent_id",
        "opponent_url",
        "round",
        "kd",
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "td_landed",
        "td_attempted",
        "sub_att",
        "rev",
        "ctrl_sec",
        "head_landed",
        "head_attempted",
        "body_landed",
        "body_attempted",
        "leg_landed",
        "leg_attempted",
        "distance_landed",
        "distance_attempted",
        "clinch_landed",
        "clinch_attempted",
        "ground_landed",
        "ground_attempted",
    }
)


class MappingTransformation(str, Enum):
    """High-level transformation required by one mapping."""

    DIRECT_RATIO = "direct_ratio"
    OPPONENT_COMPLEMENT_RATIO = "opponent_complement_ratio"
    OPPONENT_ADJUSTED_RATIO = "opponent_adjusted_ratio"
    EXPOSURE_NORMALIZED_RATE = "exposure_normalized_rate"
    PHASE_CONDITIONAL_RATE = "phase_conditional_rate"
    ROUND_TRAJECTORY_COMPOSITE = "round_trajectory_composite"
    ADVERSITY_RESPONSE_COMPOSITE = "adversity_response_composite"
    LATENT_PHASE_PROCESS = "latent_phase_process"
    LATENT_POSITIONAL_PROCESS = "latent_positional_process"
    FINISH_PROPENSITY_COMPOSITE = "finish_propensity_composite"


class OpponentInteractionMode(str, Enum):
    """How fighter and opponent evidence contribute to a target."""

    NONE = "none"
    OFFENSE_ADJUSTED_BY_DEFENSE = "offense_adjusted_by_defense"
    DEFENSE_ADJUSTED_BY_OFFENSE = "defense_adjusted_by_offense"
    JOINT_PHASE_COMPETITION = "joint_phase_competition"


class MappingSampleBasis(str, Enum):
    """Primary historical opportunity unit supporting an estimate."""

    PRIOR_FIGHTS = "prior_fights"
    PRIOR_ROUNDS = "prior_rounds"
    STRIKE_ATTEMPTS = "strike_attempts"
    LANDED_STRIKES = "landed_strikes"
    TAKEDOWN_ATTEMPTS = "takedown_attempts"
    CONTROL_OPPORTUNITIES = "control_opportunities"
    SUBMISSION_OPPORTUNITIES = "submission_opportunities"
    ADVERSITY_EVENTS = "adversity_events"


class MappingShrinkageBasis(str, Enum):
    """Evidence amount used to shrink an estimate toward its prior."""

    FIGHT_COUNT = "fight_count"
    ROUND_COUNT = "round_count"
    OPPORTUNITY_COUNT = "opportunity_count"
    EFFECTIVE_EXPOSURE = "effective_exposure"
    HIERARCHICAL_COMPOSITE = "hierarchical_composite"


class MappingFallbackLevel(str, Enum):
    """Ordered fallback levels available when fighter evidence is weak."""

    DIVISION = "division"
    GLOBAL = "global"


@dataclass(frozen=True)
class SimulatorTargetMappingSpec:
    """Complete source and estimation declaration for one simulator target."""

    target_parameter: str
    support_level: MappingSupportLevel

    source_columns: tuple[str, ...]
    opponent_source_columns: tuple[str, ...]

    transformation: MappingTransformation
    opponent_interaction: OpponentInteractionMode
    sample_basis: MappingSampleBasis
    shrinkage_basis: MappingShrinkageBasis
    fallback_hierarchy: tuple[MappingFallbackLevel, ...]

    requires_calibration: bool
    requires_outcome_join: bool
    rationale: str

    def __post_init__(self) -> None:
        """Validate one source-to-target mapping declaration."""

        if not isinstance(self.target_parameter, str):
            raise TypeError(
                "target_parameter must be a string"
            )

        if self.target_parameter not in SIMULATOR_TARGET_BY_NAME:
            raise ValueError(
                "target_parameter is not registered: "
                f"{self.target_parameter}"
            )

        if not isinstance(
            self.support_level,
            MappingSupportLevel,
        ):
            raise TypeError(
                "support_level must be MappingSupportLevel"
            )

        registered_support = SIMULATOR_TARGET_BY_NAME[
            self.target_parameter
        ].support_level

        if self.support_level is not registered_support:
            raise ValueError(
                "mapping support_level does not match "
                "the simulator target registry"
            )

        self._validate_source_columns(
            "source_columns",
            self.source_columns,
        )
        self._validate_source_columns(
            "opponent_source_columns",
            self.opponent_source_columns,
        )

        if (
            not self.source_columns
            and not self.opponent_source_columns
        ):
            raise ValueError(
                "mapping must declare at least one source column"
            )

        enum_fields = {
            "transformation": (
                self.transformation,
                MappingTransformation,
            ),
            "opponent_interaction": (
                self.opponent_interaction,
                OpponentInteractionMode,
            ),
            "sample_basis": (
                self.sample_basis,
                MappingSampleBasis,
            ),
            "shrinkage_basis": (
                self.shrinkage_basis,
                MappingShrinkageBasis,
            ),
        }

        for name, (value, expected_type) in enum_fields.items():
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} must be {expected_type.__name__}"
                )

        if not isinstance(
            self.fallback_hierarchy,
            tuple,
        ):
            raise TypeError(
                "fallback_hierarchy must be a tuple"
            )

        if not self.fallback_hierarchy:
            raise ValueError(
                "fallback_hierarchy cannot be empty"
            )

        for fallback in self.fallback_hierarchy:
            if not isinstance(
                fallback,
                MappingFallbackLevel,
            ):
                raise TypeError(
                    "fallback_hierarchy must contain "
                    "MappingFallbackLevel values"
                )

        if (
            len(set(self.fallback_hierarchy))
            != len(self.fallback_hierarchy)
        ):
            raise ValueError(
                "fallback_hierarchy cannot contain duplicates"
            )

        if (
            self.fallback_hierarchy[-1]
            is not MappingFallbackLevel.GLOBAL
        ):
            raise ValueError(
                "fallback_hierarchy must end with GLOBAL"
            )

        if type(self.requires_calibration) is not bool:
            raise TypeError(
                "requires_calibration must be boolean"
            )

        if type(self.requires_outcome_join) is not bool:
            raise TypeError(
                "requires_outcome_join must be boolean"
            )

        if (
            self.support_level
            is MappingSupportLevel.LATENT_CALIBRATED
            and not self.requires_calibration
        ):
            raise ValueError(
                "latent mappings must require calibration"
            )

        if not isinstance(self.rationale, str):
            raise TypeError(
                "rationale must be a string"
            )

        if not self.rationale.strip():
            raise ValueError(
                "rationale cannot be empty"
            )

    @staticmethod
    def _validate_source_columns(
        name: str,
        values: object,
    ) -> None:
        """Validate one tuple of authoritative raw source columns."""

        if not isinstance(values, tuple):
            raise TypeError(
                f"{name} must be a tuple"
            )

        if len(values) != len(set(values)):
            raise ValueError(
                f"{name} cannot contain duplicates"
            )

        for column in values:
            if not isinstance(column, str):
                raise TypeError(
                    f"{name} must contain strings"
                )

            if column not in AUTHORITATIVE_ROUND_SOURCE_COLUMNS:
                raise ValueError(
                    f"{name} contains unknown round source "
                    f"column: {column}"
                )


DIRECT = MappingSupportLevel.DIRECT_OBSERVED
DERIVED = MappingSupportLevel.DERIVED_OBSERVED
LATENT = MappingSupportLevel.LATENT_CALIBRATED

NONE = OpponentInteractionMode.NONE
OFFENSE_VS_DEFENSE = (
    OpponentInteractionMode.OFFENSE_ADJUSTED_BY_DEFENSE
)
DEFENSE_VS_OFFENSE = (
    OpponentInteractionMode.DEFENSE_ADJUSTED_BY_OFFENSE
)
JOINT = OpponentInteractionMode.JOINT_PHASE_COMPETITION

DIVISION_GLOBAL = (
    MappingFallbackLevel.DIVISION,
    MappingFallbackLevel.GLOBAL,
)


def _mapping(
    target_parameter: str,
    support_level: MappingSupportLevel,
    source_columns: tuple[str, ...],
    *,
    opponent_source_columns: tuple[str, ...] = (),
    transformation: MappingTransformation,
    opponent_interaction: OpponentInteractionMode,
    sample_basis: MappingSampleBasis,
    shrinkage_basis: MappingShrinkageBasis,
    requires_calibration: bool,
    requires_outcome_join: bool = False,
    rationale: str,
) -> SimulatorTargetMappingSpec:
    """Build one mapping with the locked fallback hierarchy."""

    return SimulatorTargetMappingSpec(
        target_parameter=target_parameter,
        support_level=support_level,
        source_columns=source_columns,
        opponent_source_columns=opponent_source_columns,
        transformation=transformation,
        opponent_interaction=opponent_interaction,
        sample_basis=sample_basis,
        shrinkage_basis=shrinkage_basis,
        fallback_hierarchy=DIVISION_GLOBAL,
        requires_calibration=requires_calibration,
        requires_outcome_join=requires_outcome_join,
        rationale=rationale,
    )


SIMULATOR_TARGET_MAPPING_SPECS: tuple[
    SimulatorTargetMappingSpec,
    ...,
] = (
    # ------------------------------------------------------------------
    # Transition strengths
    # ------------------------------------------------------------------
    _mapping(
        "transition.distance_retention",
        LATENT,
        (
            "distance_attempted",
            "clinch_attempted",
            "ground_attempted",
            "td_attempted",
            "ctrl_sec",
            "round",
        ),
        opponent_source_columns=(
            "clinch_attempted",
            "ground_attempted",
            "td_attempted",
            "td_landed",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.LATENT_PHASE_PROCESS,
        opponent_interaction=JOINT,
        sample_basis=MappingSampleBasis.PRIOR_ROUNDS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Distance retention is inferred from phase mix under "
            "opponent clinch and wrestling pressure."
        ),
    ),
    _mapping(
        "transition.clinch_entry_tendency",
        LATENT,
        (
            "clinch_attempted",
            "distance_attempted",
            "td_attempted",
            "ctrl_sec",
            "round",
        ),
        transformation=MappingTransformation.LATENT_PHASE_PROCESS,
        opponent_interaction=NONE,
        sample_basis=MappingSampleBasis.PRIOR_ROUNDS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "UFCStats exposes clinch occupancy signals but not "
            "individual clinch-entry events."
        ),
    ),
    _mapping(
        "transition.clinch_entry_resistance",
        LATENT,
        (
            "distance_attempted",
            "clinch_attempted",
            "ctrl_sec",
            "round",
        ),
        opponent_source_columns=(
            "clinch_attempted",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.LATENT_PHASE_PROCESS,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.PRIOR_ROUNDS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Resistance is inferred from opponent clinch occupancy "
            "and control relative to opponent historical pressure."
        ),
    ),
    _mapping(
        "transition.takedown_entry_tendency",
        DERIVED,
        (
            "td_attempted",
            "round",
            "total_rounds",
        ),
        transformation=MappingTransformation.EXPOSURE_NORMALIZED_RATE,
        opponent_interaction=NONE,
        sample_basis=MappingSampleBasis.PRIOR_ROUNDS,
        shrinkage_basis=MappingShrinkageBasis.EFFECTIVE_EXPOSURE,
        requires_calibration=True,
        rationale=(
            "Prior takedown attempts provide an observable entry "
            "signal that must be normalized into a unit strength."
        ),
    ),
    _mapping(
        "transition.takedown_completion_ability",
        DIRECT,
        (
            "td_landed",
            "td_attempted",
        ),
        transformation=MappingTransformation.DIRECT_RATIO,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.TAKEDOWN_ATTEMPTS,
        shrinkage_basis=MappingShrinkageBasis.OPPORTUNITY_COUNT,
        requires_calibration=False,
        rationale=(
            "Completion ability is directly supported by landed "
            "takedowns over attempted takedowns."
        ),
    ),
    _mapping(
        "transition.takedown_resistance",
        DIRECT,
        (),
        opponent_source_columns=(
            "td_landed",
            "td_attempted",
        ),
        transformation=MappingTransformation.OPPONENT_COMPLEMENT_RATIO,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.TAKEDOWN_ATTEMPTS,
        shrinkage_basis=MappingShrinkageBasis.OPPORTUNITY_COUNT,
        requires_calibration=False,
        rationale=(
            "Resistance is the shrunk complement of opponent "
            "takedown completion against the fighter."
        ),
    ),
    _mapping(
        "transition.takedown_persistence",
        LATENT,
        (
            "td_attempted",
            "td_landed",
            "ctrl_sec",
            "round",
        ),
        transformation=MappingTransformation.LATENT_POSITIONAL_PROCESS,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.PRIOR_ROUNDS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Persistence is inferred from repeated wrestling "
            "pressure across rounds and fights."
        ),
    ),
    _mapping(
        "transition.failed_takedown_persistence",
        LATENT,
        (
            "td_attempted",
            "td_landed",
            "ctrl_sec",
            "round",
        ),
        transformation=MappingTransformation.LATENT_POSITIONAL_PROCESS,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.TAKEDOWN_ATTEMPTS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Round totals cannot identify exact retry sequences "
            "after failed takedowns."
        ),
    ),
    _mapping(
        "transition.clinch_retention",
        LATENT,
        (
            "clinch_attempted",
            "clinch_landed",
            "ctrl_sec",
            "round",
        ),
        opponent_source_columns=(
            "distance_attempted",
            "clinch_attempted",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.LATENT_PHASE_PROCESS,
        opponent_interaction=JOINT,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Clinch retention requires latent allocation of control "
            "and occupancy to clinch sequences."
        ),
    ),
    _mapping(
        "transition.clinch_escape_ability",
        LATENT,
        (
            "distance_attempted",
            "clinch_attempted",
            "ctrl_sec",
            "round",
        ),
        opponent_source_columns=(
            "clinch_attempted",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.LATENT_PHASE_PROCESS,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Exact clinch exits are unavailable and must be inferred "
            "from phase recovery under opponent pressure."
        ),
    ),
    _mapping(
        "transition.ground_retention",
        LATENT,
        (
            "td_landed",
            "ground_attempted",
            "ctrl_sec",
            "sub_att",
            "round",
        ),
        opponent_source_columns=(
            "rev",
            "distance_attempted",
        ),
        transformation=MappingTransformation.LATENT_POSITIONAL_PROCESS,
        opponent_interaction=JOINT,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Ground retention is inferred from takedowns, ground "
            "activity, control, and opponent reversal pressure."
        ),
    ),
    _mapping(
        "transition.ground_escape_ability",
        DERIVED,
        (
            "rev",
            "distance_attempted",
            "round",
        ),
        opponent_source_columns=(
            "td_landed",
            "ground_attempted",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.OPPONENT_ADJUSTED_RATIO,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Escape ability is derived from control survived and "
            "returns toward neutral or reversed positions."
        ),
    ),
    _mapping(
        "transition.reversal_ability",
        DIRECT,
        (
            "rev",
        ),
        opponent_source_columns=(
            "td_landed",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.DIRECT_RATIO,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.OPPORTUNITY_COUNT,
        requires_calibration=False,
        rationale=(
            "Recorded reversals directly support reversal ability "
            "when normalized by ground-control opportunities."
        ),
    ),
    _mapping(
        "transition.phase_imposition",
        LATENT,
        (
            "distance_attempted",
            "clinch_attempted",
            "ground_attempted",
            "td_attempted",
            "td_landed",
            "ctrl_sec",
            "round",
        ),
        opponent_source_columns=(
            "distance_attempted",
            "clinch_attempted",
            "ground_attempted",
            "td_attempted",
            "td_landed",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.LATENT_PHASE_PROCESS,
        opponent_interaction=JOINT,
        sample_basis=MappingSampleBasis.PRIOR_FIGHTS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Broad phase imposition combines phase occupancy and "
            "wrestling-control evidence against opponent baselines."
        ),
    ),
    _mapping(
        "transition.phase_resistance",
        LATENT,
        (
            "distance_attempted",
            "clinch_attempted",
            "ground_attempted",
            "rev",
            "ctrl_sec",
            "round",
        ),
        opponent_source_columns=(
            "clinch_attempted",
            "ground_attempted",
            "td_attempted",
            "td_landed",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.LATENT_PHASE_PROCESS,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.PRIOR_FIGHTS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Broad phase resistance measures deviation from the "
            "opponent's usual ability to impose phases."
        ),
    ),

    # ------------------------------------------------------------------
    # Phase-specific activity
    # ------------------------------------------------------------------
    _mapping(
        "phase.distance.sig_strike_attempt_rate",
        DERIVED,
        (
            "distance_attempted",
            "round",
        ),
        transformation=MappingTransformation.PHASE_CONDITIONAL_RATE,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.PRIOR_ROUNDS,
        shrinkage_basis=MappingShrinkageBasis.EFFECTIVE_EXPOSURE,
        requires_calibration=True,
        rationale=(
            "Distance attempts are observable, but distance exposure "
            "must be estimated before conversion to a 30-second rate."
        ),
    ),
    _mapping(
        "phase.distance.sig_strike_accuracy",
        DIRECT,
        (
            "distance_landed",
            "distance_attempted",
        ),
        opponent_source_columns=(
            "distance_landed",
            "distance_attempted",
        ),
        transformation=MappingTransformation.DIRECT_RATIO,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.STRIKE_ATTEMPTS,
        shrinkage_basis=MappingShrinkageBasis.OPPORTUNITY_COUNT,
        requires_calibration=False,
        rationale=(
            "Distance accuracy is directly observed and then "
            "opponent-adjusted and shrunk."
        ),
    ),
    _mapping(
        "phase.distance.knockdown_probability_per_landed",
        DERIVED,
        (
            "kd",
            "distance_landed",
            "clinch_landed",
            "ground_landed",
            "head_landed",
        ),
        transformation=MappingTransformation.FINISH_PROPENSITY_COMPOSITE,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.LANDED_STRIKES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Knockdowns are observed, but their exact strike phase "
            "must be probabilistically attributed."
        ),
    ),
    _mapping(
        "phase.clinch.clinch_strike_attempt_rate",
        DERIVED,
        (
            "clinch_attempted",
            "round",
        ),
        transformation=MappingTransformation.PHASE_CONDITIONAL_RATE,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.PRIOR_ROUNDS,
        shrinkage_basis=MappingShrinkageBasis.EFFECTIVE_EXPOSURE,
        requires_calibration=True,
        rationale=(
            "Clinch attempts require estimated clinch exposure before "
            "conversion to a conditional segment rate."
        ),
    ),
    _mapping(
        "phase.clinch.clinch_strike_accuracy",
        DIRECT,
        (
            "clinch_landed",
            "clinch_attempted",
        ),
        opponent_source_columns=(
            "clinch_landed",
            "clinch_attempted",
        ),
        transformation=MappingTransformation.DIRECT_RATIO,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.STRIKE_ATTEMPTS,
        shrinkage_basis=MappingShrinkageBasis.OPPORTUNITY_COUNT,
        requires_calibration=False,
        rationale=(
            "Clinch accuracy is directly observed and can be "
            "opponent-adjusted."
        ),
    ),
    _mapping(
        "phase.clinch.control_seconds_mean",
        LATENT,
        (
            "ctrl_sec",
            "clinch_attempted",
            "ground_attempted",
            "td_landed",
        ),
        transformation=MappingTransformation.LATENT_PHASE_PROCESS,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Total control time must be split between clinch and "
            "ground ownership."
        ),
    ),
    _mapping(
        "phase.clinch.damaging_clinch_probability",
        LATENT,
        (
            "clinch_landed",
            "head_landed",
            "kd",
            "ctrl_sec",
        ),
        opponent_source_columns=(
            "kd",
            "head_landed",
        ),
        transformation=MappingTransformation.FINISH_PROPENSITY_COMPOSITE,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Damaging clinch events are latent and must be calibrated "
            "from clinch offense and adversity signals."
        ),
    ),
    _mapping(
        "phase.ground_owner.ground_strike_attempt_rate",
        DERIVED,
        (
            "ground_attempted",
            "td_landed",
            "ctrl_sec",
            "round",
        ),
        transformation=MappingTransformation.PHASE_CONDITIONAL_RATE,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.EFFECTIVE_EXPOSURE,
        requires_calibration=True,
        rationale=(
            "Ground attempts are observed, while owned-ground "
            "exposure must be estimated."
        ),
    ),
    _mapping(
        "phase.ground_owner.ground_strike_accuracy",
        DIRECT,
        (
            "ground_landed",
            "ground_attempted",
        ),
        opponent_source_columns=(
            "ground_landed",
            "ground_attempted",
        ),
        transformation=MappingTransformation.DIRECT_RATIO,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.STRIKE_ATTEMPTS,
        shrinkage_basis=MappingShrinkageBasis.OPPORTUNITY_COUNT,
        requires_calibration=False,
        rationale=(
            "Ground striking accuracy is directly observed and "
            "opponent-adjusted."
        ),
    ),
    _mapping(
        "phase.ground_owner.control_seconds_mean",
        LATENT,
        (
            "ctrl_sec",
            "td_landed",
            "ground_attempted",
            "sub_att",
        ),
        opponent_source_columns=(
            "rev",
        ),
        transformation=MappingTransformation.LATENT_POSITIONAL_PROCESS,
        opponent_interaction=JOINT,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Owned-ground control requires latent allocation of total "
            "control time."
        ),
    ),
    _mapping(
        "phase.ground_owner.submission_attempt_rate",
        DERIVED,
        (
            "sub_att",
            "td_landed",
            "ground_attempted",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.PHASE_CONDITIONAL_RATE,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.SUBMISSION_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.EFFECTIVE_EXPOSURE,
        requires_calibration=True,
        rationale=(
            "Submission attempts are observed, while ground-owner "
            "segment exposure must be estimated."
        ),
    ),
    _mapping(
        "phase.ground_owner.position_advancement_probability",
        LATENT,
        (
            "td_landed",
            "ctrl_sec",
            "ground_attempted",
            "sub_att",
        ),
        opponent_source_columns=(
            "rev",
        ),
        transformation=MappingTransformation.LATENT_POSITIONAL_PROCESS,
        opponent_interaction=JOINT,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "UFCStats does not record positional advancements, so "
            "they must be inferred from ground productivity."
        ),
    ),
    _mapping(
        "phase.ground_defender.escape_attempt_rate",
        LATENT,
        (
            "rev",
            "distance_attempted",
            "round",
        ),
        opponent_source_columns=(
            "td_landed",
            "ground_attempted",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.LATENT_POSITIONAL_PROCESS,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Escape attempts are not directly recorded and must be "
            "estimated from ground exposure and phase exits."
        ),
    ),
    _mapping(
        "phase.ground_defender.reversal_attempt_rate",
        DERIVED,
        (
            "rev",
            "round",
        ),
        opponent_source_columns=(
            "td_landed",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.OPPONENT_ADJUSTED_RATIO,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.EFFECTIVE_EXPOSURE,
        requires_calibration=True,
        rationale=(
            "Recorded reversals support an exposure-adjusted reversal "
            "attempt rate."
        ),
    ),
    _mapping(
        "phase.ground_defender.scramble_attempt_rate",
        LATENT,
        (
            "rev",
            "distance_attempted",
            "ground_attempted",
            "ctrl_sec",
            "round",
        ),
        opponent_source_columns=(
            "td_landed",
            "ground_attempted",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.LATENT_POSITIONAL_PROCESS,
        opponent_interaction=JOINT,
        sample_basis=MappingSampleBasis.CONTROL_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Scramble attempts are not directly recorded and require "
            "latent positional calibration."
        ),
    ),
    _mapping(
        "phase.ground_defender.submission_defense",
        LATENT,
        (
            "rev",
            "ctrl_sec",
            "round",
        ),
        opponent_source_columns=(
            "sub_att",
            "td_landed",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.FINISH_PROPENSITY_COMPOSITE,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.SUBMISSION_OPPORTUNITIES,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        requires_outcome_join=True,
        rationale=(
            "Submission defense needs opponent attempt pressure plus "
            "historical submission outcome information."
        ),
    ),

    # ------------------------------------------------------------------
    # Dynamic response traits
    # ------------------------------------------------------------------
    _mapping(
        "dynamic.fatigue_accumulation_resistance",
        DERIVED,
        (
            "round",
            "sig_str_attempted",
            "total_str_attempted",
            "td_attempted",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.ROUND_TRAJECTORY_COMPOSITE,
        opponent_interaction=NONE,
        sample_basis=MappingSampleBasis.PRIOR_ROUNDS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Workload and late-round pace trajectories support a "
            "normalized fatigue-accumulation resistance trait."
        ),
    ),
    _mapping(
        "dynamic.fatigue_performance_resilience",
        DERIVED,
        (
            "round",
            "sig_str_landed",
            "sig_str_attempted",
            "total_str_landed",
            "total_str_attempted",
            "td_landed",
            "td_attempted",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.ROUND_TRAJECTORY_COMPOSITE,
        opponent_interaction=OFFENSE_VS_DEFENSE,
        sample_basis=MappingSampleBasis.PRIOR_ROUNDS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Output and efficiency preservation across rounds support "
            "fatigue performance resilience."
        ),
    ),
    _mapping(
        "dynamic.recovery_ability",
        LATENT,
        (
            "round",
            "sig_str_landed",
            "sig_str_attempted",
            "td_attempted",
            "ctrl_sec",
        ),
        opponent_source_columns=(
            "sig_str_landed",
            "head_landed",
            "kd",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.ADVERSITY_RESPONSE_COMPOSITE,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.ADVERSITY_EVENTS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Recovery is inferred from between-round performance "
            "rebounds after workload and opponent pressure."
        ),
    ),
    _mapping(
        "dynamic.damage_resistance",
        LATENT,
        (
            "round",
            "sig_str_landed",
            "head_landed",
            "kd",
        ),
        opponent_source_columns=(
            "sig_str_landed",
            "head_landed",
            "kd",
            "ground_landed",
        ),
        transformation=MappingTransformation.FINISH_PROPENSITY_COMPOSITE,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.ADVERSITY_EVENTS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        requires_outcome_join=True,
        rationale=(
            "Damage resistance requires opponent damage exposure and "
            "historical survival or finish outcomes."
        ),
    ),
    _mapping(
        "dynamic.acute_stress_resistance",
        LATENT,
        (
            "round",
            "sig_str_landed",
            "sig_str_attempted",
            "td_attempted",
            "ctrl_sec",
        ),
        opponent_source_columns=(
            "kd",
            "head_landed",
            "ground_landed",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.ADVERSITY_RESPONSE_COMPOSITE,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.ADVERSITY_EVENTS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Immediate output preservation after adversity supports "
            "acute-stress resistance."
        ),
    ),
    _mapping(
        "dynamic.acute_stress_recovery",
        LATENT,
        (
            "round",
            "sig_str_landed",
            "sig_str_attempted",
            "td_landed",
            "td_attempted",
            "ctrl_sec",
        ),
        opponent_source_columns=(
            "kd",
            "head_landed",
            "ground_landed",
            "ctrl_sec",
        ),
        transformation=MappingTransformation.ADVERSITY_RESPONSE_COMPOSITE,
        opponent_interaction=DEFENSE_VS_OFFENSE,
        sample_basis=MappingSampleBasis.ADVERSITY_EVENTS,
        shrinkage_basis=MappingShrinkageBasis.HIERARCHICAL_COMPOSITE,
        requires_calibration=True,
        rationale=(
            "Later-round rebound after acute adversity supports the "
            "acute-stress recovery trait."
        ),
    ),
)


SIMULATOR_TARGET_MAPPING_BY_NAME = {
    spec.target_parameter: spec
    for spec in SIMULATOR_TARGET_MAPPING_SPECS
}


def validate_simulator_mapping_registry() -> None:
    """Validate complete and exact coverage of all simulator targets."""

    expected_targets = set(SIMULATOR_TARGET_BY_NAME)
    mapped_targets = {
        spec.target_parameter
        for spec in SIMULATOR_TARGET_MAPPING_SPECS
    }

    if len(SIMULATOR_TARGET_MAPPING_SPECS) != 37:
        raise RuntimeError(
            "simulator mapping registry must contain "
            "exactly 37 mappings"
        )

    if len(SIMULATOR_TARGET_MAPPING_BY_NAME) != 37:
        raise RuntimeError(
            "simulator mapping registry contains "
            "duplicate target names"
        )

    if mapped_targets != expected_targets:
        missing = sorted(
            expected_targets - mapped_targets
        )
        extra = sorted(
            mapped_targets - expected_targets
        )

        raise RuntimeError(
            "simulator mapping registry does not match "
            f"target contracts; missing={missing}, extra={extra}"
        )


validate_simulator_mapping_registry()
