"""Approved RFS-to-simulation-profile mappings for Monte Carlo V1.

This registry defines profile inputs only. It does not define simulation
mechanics, finish formulas, scoring weights, or calibration coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping


class RFSFamily(str, Enum):
    """Implemented Round Fighter State source families."""

    TRAJECTORY = "trajectory"
    OPENING_OFFENSE = "opening_offense"
    SUPPRESSION = "suppression"
    WRESTLING = "wrestling"
    DEFENSE = "defense"
    SUBMISSION_RESULTS = "submission_results"


@dataclass(frozen=True)
class ProfileParameterDefinition:
    """Definition of one pre-fight simulation-profile parameter."""

    name: str
    family: RFSFamily
    source_column: str
    prior_fight_count_column: str
    prior_valid_count_column: str
    description: str
    fallback_source_column: str | None = None


TRAJECTORY_PARAMETERS: tuple[ProfileParameterDefinition, ...] = (
    ProfileParameterDefinition(
        name="sig_attempt_trajectory",
        family=RFSFamily.TRAJECTORY,
        source_column="rfs_traj_ewm_sig_attempt_slope",
        prior_fight_count_column="rfs_traj_prior_fight_count",
        prior_valid_count_column="rfs_traj_prior_valid_trajectory_count",
        description="EWM significant-strike attempt trajectory.",
    ),
    ProfileParameterDefinition(
        name="total_attempt_trajectory",
        family=RFSFamily.TRAJECTORY,
        source_column="rfs_traj_ewm_total_attempt_slope",
        prior_fight_count_column="rfs_traj_prior_fight_count",
        prior_valid_count_column="rfs_traj_prior_valid_trajectory_count",
        description="EWM total-strike attempt trajectory.",
    ),
    ProfileParameterDefinition(
        name="sig_accuracy_trajectory",
        family=RFSFamily.TRAJECTORY,
        source_column="rfs_traj_ewm_sig_accuracy_slope",
        prior_fight_count_column="rfs_traj_prior_fight_count",
        prior_valid_count_column="rfs_traj_prior_valid_trajectory_count",
        description="EWM significant-strike accuracy trajectory.",
    ),
    ProfileParameterDefinition(
        name="late_sig_output_ratio",
        family=RFSFamily.TRAJECTORY,
        source_column="rfs_traj_ewm_sig_attempt_late_ratio",
        prior_fight_count_column="rfs_traj_prior_fight_count",
        prior_valid_count_column="rfs_traj_prior_valid_trajectory_count",
        description="Late-round significant-strike attempt ratio.",
    ),
    ProfileParameterDefinition(
        name="late_td_output_ratio",
        family=RFSFamily.TRAJECTORY,
        source_column="rfs_traj_ewm_td_attempt_late_ratio",
        prior_fight_count_column="rfs_traj_prior_fight_count",
        prior_valid_count_column="rfs_traj_prior_valid_trajectory_count",
        description="Late-round takedown attempt ratio.",
    ),
    ProfileParameterDefinition(
        name="late_control_ratio",
        family=RFSFamily.TRAJECTORY,
        source_column="rfs_traj_ewm_control_late_ratio",
        prior_fight_count_column="rfs_traj_prior_fight_count",
        prior_valid_count_column="rfs_traj_prior_valid_trajectory_count",
        description="Late-round control persistence ratio.",
    ),
)


OPENING_OFFENSE_PARAMETERS: tuple[ProfileParameterDefinition, ...] = (
    ProfileParameterDefinition(
        name="opening_sig_attempts",
        family=RFSFamily.OPENING_OFFENSE,
        source_column="rfs_open_ewm_round1_sig_attempted_per_min",
        prior_fight_count_column="rfs_open_prior_fight_count",
        prior_valid_count_column="rfs_open_prior_valid_opening_count",
        description="EWM Round 1 significant-strike attempts.",
    ),
    ProfileParameterDefinition(
        name="opening_sig_landed",
        family=RFSFamily.OPENING_OFFENSE,
        source_column="rfs_open_ewm_round1_sig_landed_per_min",
        prior_fight_count_column="rfs_open_prior_fight_count",
        prior_valid_count_column="rfs_open_prior_valid_opening_count",
        description="EWM Round 1 significant strikes landed.",
    ),
    ProfileParameterDefinition(
        name="opening_head_landed",
        family=RFSFamily.OPENING_OFFENSE,
        source_column="rfs_open_ewm_round1_head_landed_per_min",
        prior_fight_count_column="rfs_open_prior_fight_count",
        prior_valid_count_column="rfs_open_prior_valid_opening_count",
        description="EWM Round 1 head strikes landed.",
    ),
    ProfileParameterDefinition(
        name="opening_ground_landed",
        family=RFSFamily.OPENING_OFFENSE,
        source_column="rfs_open_ewm_round1_ground_landed_per_min",
        prior_fight_count_column="rfs_open_prior_fight_count",
        prior_valid_count_column="rfs_open_prior_valid_opening_count",
        description="EWM Round 1 ground strikes landed.",
    ),
    ProfileParameterDefinition(
        name="opening_knockdowns",
        family=RFSFamily.OPENING_OFFENSE,
        source_column="rfs_open_ewm_round1_kd_per_min",
        prior_fight_count_column="rfs_open_prior_fight_count",
        prior_valid_count_column="rfs_open_prior_valid_opening_count",
        description="EWM Round 1 knockdowns.",
    ),
    ProfileParameterDefinition(
        name="opening_kd_per_sig_landed",
        family=RFSFamily.OPENING_OFFENSE,
        source_column="rfs_open_ewm_round1_kd_per_sig_landed",
        prior_fight_count_column="rfs_open_prior_fight_count",
        prior_valid_count_column="rfs_open_prior_valid_opening_count",
        description="EWM Round 1 knockdowns per significant strike landed.",
    ),
)


SUPPRESSION_PARAMETERS: tuple[ProfileParameterDefinition, ...] = (
    ProfileParameterDefinition(
        name="opponent_sig_attempt_suppression",
        family=RFSFamily.SUPPRESSION,
        source_column="rfs_suppress_ewm_opp_sig_attempt_delta",
        prior_fight_count_column="rfs_suppress_prior_fight_count",
        prior_valid_count_column="rfs_suppress_prior_valid_suppression_count",
        description="Opponent significant-strike attempt suppression.",
    ),
    ProfileParameterDefinition(
        name="opponent_sig_accuracy_suppression",
        family=RFSFamily.SUPPRESSION,
        source_column="rfs_suppress_ewm_opp_sig_accuracy_delta",
        prior_fight_count_column="rfs_suppress_prior_fight_count",
        prior_valid_count_column="rfs_suppress_prior_valid_suppression_count",
        description="Opponent significant-strike accuracy suppression.",
    ),
    ProfileParameterDefinition(
        name="opponent_td_attempt_suppression",
        family=RFSFamily.SUPPRESSION,
        source_column="rfs_suppress_ewm_opp_td_attempt_delta",
        prior_fight_count_column="rfs_suppress_prior_fight_count",
        prior_valid_count_column="rfs_suppress_prior_valid_suppression_count",
        description="Opponent takedown attempt suppression.",
    ),
    ProfileParameterDefinition(
        name="opponent_td_accuracy_suppression",
        family=RFSFamily.SUPPRESSION,
        source_column="rfs_suppress_ewm_opp_td_accuracy_delta",
        prior_fight_count_column="rfs_suppress_prior_fight_count",
        prior_valid_count_column="rfs_suppress_prior_valid_suppression_count",
        description="Opponent takedown accuracy suppression.",
    ),
    ProfileParameterDefinition(
        name="opponent_control_suppression",
        family=RFSFamily.SUPPRESSION,
        source_column="rfs_suppress_ewm_opp_control_delta",
        prior_fight_count_column="rfs_suppress_prior_fight_count",
        prior_valid_count_column="rfs_suppress_prior_valid_suppression_count",
        description="Opponent control-time suppression.",
    ),
    ProfileParameterDefinition(
        name="phase_mix_disruption",
        family=RFSFamily.SUPPRESSION,
        source_column="rfs_suppress_ewm_opp_phase_mix_disruption",
        prior_fight_count_column="rfs_suppress_prior_fight_count",
        prior_valid_count_column="rfs_suppress_prior_valid_suppression_count",
        description="Opponent phase-mix disruption.",
    ),
)


WRESTLING_PARAMETERS: tuple[ProfileParameterDefinition, ...] = (
    ProfileParameterDefinition(
        name="control_per_td_attempt",
        family=RFSFamily.WRESTLING,
        source_column="rfs_wrestle_ewm_control_per_td_attempt",
        prior_fight_count_column="rfs_wrestle_prior_fight_count",
        prior_valid_count_column="rfs_wrestle_prior_valid_wrestling_count",
        description="Control seconds generated per takedown attempt.",
    ),
    ProfileParameterDefinition(
        name="td_to_control_conversion",
        family=RFSFamily.WRESTLING,
        source_column="rfs_wrestle_ewm_td_to_control_conversion",
        prior_fight_count_column="rfs_wrestle_prior_fight_count",
        prior_valid_count_column="rfs_wrestle_prior_valid_wrestling_count",
        description="Takedown-to-control conversion.",
    ),
    ProfileParameterDefinition(
        name="control_to_damage",
        family=RFSFamily.WRESTLING,
        source_column="rfs_wrestle_ewm_control_to_damage_score",
        prior_fight_count_column="rfs_wrestle_prior_fight_count",
        prior_valid_count_column="rfs_wrestle_prior_valid_wrestling_count",
        description="Ground offense generated from control.",
    ),
    ProfileParameterDefinition(
        name="submission_pressure",
        family=RFSFamily.WRESTLING,
        source_column="rfs_wrestle_ewm_submission_pressure_score",
        prior_fight_count_column="rfs_wrestle_prior_fight_count",
        prior_valid_count_column="rfs_wrestle_prior_valid_wrestling_count",
        description="Submission attempts generated from control.",
    ),
    ProfileParameterDefinition(
        name="control_stability",
        family=RFSFamily.WRESTLING,
        source_column="rfs_wrestle_ewm_control_stability_score",
        prior_fight_count_column="rfs_wrestle_prior_fight_count",
        prior_valid_count_column="rfs_wrestle_prior_valid_wrestling_count",
        description="Ability to maintain control without reversal.",
    ),
    ProfileParameterDefinition(
        name="wrestling_persistence",
        family=RFSFamily.WRESTLING,
        source_column="rfs_wrestle_ewm_td_persistence_score",
        prior_fight_count_column="rfs_wrestle_prior_fight_count",
        prior_valid_count_column="rfs_wrestle_prior_valid_wrestling_count",
        description="Persistence of takedown pressure.",
    ),
)


DEFENSE_PARAMETERS: tuple[ProfileParameterDefinition, ...] = (
    ProfileParameterDefinition(
        name="sig_absorption_trajectory",
        family=RFSFamily.DEFENSE,
        source_column="rfs_def_ewm_sig_absorbed_slope",
        prior_fight_count_column="rfs_def_prior_fight_count",
        prior_valid_count_column="rfs_def_prior_valid_defense_count",
        description="Trajectory of significant strikes absorbed.",
    ),
    ProfileParameterDefinition(
        name="head_absorption_trajectory",
        family=RFSFamily.DEFENSE,
        source_column="rfs_def_ewm_head_absorbed_slope",
        prior_fight_count_column="rfs_def_prior_fight_count",
        prior_valid_count_column="rfs_def_prior_valid_defense_count",
        description="Trajectory of head strikes absorbed.",
    ),
    ProfileParameterDefinition(
        name="opponent_output_acceleration",
        family=RFSFamily.DEFENSE,
        source_column="rfs_def_ewm_opp_output_acceleration",
        prior_fight_count_column="rfs_def_prior_fight_count",
        prior_valid_count_column="rfs_def_prior_valid_defense_count",
        description="Opponent output acceleration across rounds.",
    ),
    ProfileParameterDefinition(
        name="late_head_damage_allowed",
        family=RFSFamily.DEFENSE,
        source_column="rfs_def_ewm_late_head_damage_allowed_delta",
        prior_fight_count_column="rfs_def_prior_fight_count",
        prior_valid_count_column="rfs_def_prior_valid_defense_count",
        description="Late-round increase in head damage allowed.",
    ),
    ProfileParameterDefinition(
        name="late_td_defense_decay",
        family=RFSFamily.DEFENSE,
        source_column="rfs_def_ewm_late_td_defense_decay",
        prior_fight_count_column="rfs_def_prior_fight_count",
        prior_valid_count_column="rfs_def_prior_valid_defense_count",
        description="Late-round takedown-defense deterioration.",
    ),
    ProfileParameterDefinition(
        name="knockdowns_absorbed",
        family=RFSFamily.DEFENSE,
        source_column="rfs_def_ewm_kd_absorbed",
        prior_fight_count_column="rfs_def_prior_fight_count",
        prior_valid_count_column="rfs_def_prior_valid_defense_count",
        description="Historical EWM knockdowns absorbed.",
    ),
    ProfileParameterDefinition(
        name="defensive_deterioration",
        family=RFSFamily.DEFENSE,
        source_column="rfs_def_ewm_defensive_deterioration_score",
        prior_fight_count_column="rfs_def_prior_fight_count",
        prior_valid_count_column="rfs_def_prior_valid_defense_count",
        description="Composite defensive deterioration proxy.",
    ),
)



SUBMISSION_RESULT_PARAMETERS: tuple[
    ProfileParameterDefinition,
    ...
] = (
    ProfileParameterDefinition(
        name="prior_submission_wins",
        family=RFSFamily.SUBMISSION_RESULTS,
        source_column="rfs_sub_career_submission_wins",
        prior_fight_count_column="rfs_sub_career_fight_count",
        prior_valid_count_column=(
            "rfs_sub_career_valid_fight_count"
        ),
        description=(
            "Submission wins recorded before the target fight."
        ),
    ),
    ProfileParameterDefinition(
        name="prior_submission_win_rate",
        family=RFSFamily.SUBMISSION_RESULTS,
        source_column=(
            "rfs_sub_career_submission_win_rate"
        ),
        prior_fight_count_column="rfs_sub_career_fight_count",
        prior_valid_count_column=(
            "rfs_sub_career_valid_fight_count"
        ),
        description=(
            "Prior submission wins divided by prior fights."
        ),
    ),
    ProfileParameterDefinition(
        name="prior_submission_conversion_rate",
        family=RFSFamily.SUBMISSION_RESULTS,
        source_column=(
            "rfs_sub_career_smoothed_conversion_rate"
        ),
        prior_fight_count_column="rfs_sub_career_fight_count",
        prior_valid_count_column=(
            "rfs_sub_career_valid_fight_count"
        ),
        description=(
            "Beta-smoothed prior submission wins per "
            "recorded submission attempt."
        ),
    ),
    ProfileParameterDefinition(
        name="prior_submission_loss_rate",
        family=RFSFamily.SUBMISSION_RESULTS,
        source_column=(
            "rfs_sub_career_submission_loss_rate"
        ),
        prior_fight_count_column="rfs_sub_career_fight_count",
        prior_valid_count_column=(
            "rfs_sub_career_valid_fight_count"
        ),
        description=(
            "Prior submission losses divided by prior fights."
        ),
    ),
)


PROFILE_PARAMETER_DEFINITIONS: tuple[ProfileParameterDefinition, ...] = (
    *TRAJECTORY_PARAMETERS,
    *OPENING_OFFENSE_PARAMETERS,
    *SUPPRESSION_PARAMETERS,
    *WRESTLING_PARAMETERS,
    *DEFENSE_PARAMETERS,
    *SUBMISSION_RESULT_PARAMETERS,
)


LAST3_PROFILE_PARAMETER_DEFINITIONS: tuple[
    ProfileParameterDefinition,
    ...
] = tuple(
    replace(
        definition,
        source_column=definition.source_column.replace(
            "_ewm_",
            "_last3_",
        ),
        fallback_source_column=definition.source_column,
        description=definition.description.replace(
            "EWM ",
            "Last-3 ",
        ),
    )
    for definition in PROFILE_PARAMETER_DEFINITIONS
)


PARAMETER_DEFINITIONS_BY_NAME: Mapping[
    str,
    ProfileParameterDefinition,
] = {
    definition.name: definition
    for definition in PROFILE_PARAMETER_DEFINITIONS
}


def validate_parameter_registry() -> None:
    """Validate uniqueness and required naming contracts."""

    names = [definition.name for definition in PROFILE_PARAMETER_DEFINITIONS]
    if len(names) != len(set(names)):
        raise ValueError("Profile parameter names must be unique")

    source_columns = [
        definition.source_column
        for definition in PROFILE_PARAMETER_DEFINITIONS
    ]
    if len(source_columns) != len(set(source_columns)):
        raise ValueError("RFS source columns must be unique")

    for definition in PROFILE_PARAMETER_DEFINITIONS:
        if not definition.source_column.startswith("rfs_"):
            raise ValueError(
                f"Invalid RFS source column: {definition.source_column}"
            )

        if not definition.description.strip():
            raise ValueError(
                f"Missing description for parameter: {definition.name}"
            )
