"""Calibrate leakage-safe RFS profiles into Monte Carlo V2 parameters.

V0 calibration policy
---------------------

1. Never use current-fight observations.
2. Population calibration must be supplied from rows occurring BEFORE the
   simulated fight.
3. Direct observed rates/probabilities retain physical units.
4. Latent traits are population-percentile composites in [0, 1].
5. Low-history fighters are shrunk toward a population prior.
6. Every final target carries ReliabilityShrunkEstimate audit metadata.

This is intentionally a transparent first calibration layer. Calibration
constants can later be learned/tuned from historical simulation performance
without changing the RFS or Monte Carlo contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping

import numpy as np
import pandas as pd

from pipeline.round_stats.rfs_simulator_feature_contracts import (
    ReliabilityShrunkEstimate,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.rfs_target_resolver import (
    TARGET_EVIDENCE,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


SHRINKAGE_PRIOR_FIGHTS = 3.0


class RFSParameterResolutionError(RuntimeError):
    """Raised when simulator parameters cannot be calibrated safely."""


@dataclass(frozen=True)
class CalibratedFighterParameters:
    """Complete immutable simulator parameter set for one fighter."""

    transition: FighterTransitionParameters
    phase: FighterPhaseParameters
    dynamic: FighterDynamicParameters
    estimates: dict[str, ReliabilityShrunkEstimate]


# ---------------------------------------------------------------------------
# State-column resolution
# ---------------------------------------------------------------------------

FAMILY_PREFIX_MAP = (
    ("rfs_phase_base_fight_", "rfs_phase_base_"),
    ("rfs_phase_interact_fight_", "rfs_phase_interact_"),
    ("rfs_dynamic_response_fight_", "rfs_dynamic_response_"),
    ("rfs_finish_state_fight_", "rfs_finish_state_"),
)


def _state_candidates(
    fight_feature: str,
) -> tuple[str, ...]:
    """Translate an approved fight feature into prior-state candidates."""

    for fight_prefix, state_prefix in FAMILY_PREFIX_MAP:
        if fight_feature.startswith(fight_prefix):
            suffix = fight_feature.removeprefix(fight_prefix)

            return (
                f"{state_prefix}ewm_{suffix}",
                f"{state_prefix}exp_{suffix}",
                f"{state_prefix}last3_{suffix}",
            )

    raise RFSParameterResolutionError(
        f"unsupported RFS evidence feature: {fight_feature}"
    )


def _find_evidence(
    target: str,
    suffix: str,
) -> str:
    """Find one approved evidence feature by exact family-relative name."""

    matches = []

    for feature in TARGET_EVIDENCE[target]:
        for fight_prefix, _ in FAMILY_PREFIX_MAP:
            if not feature.startswith(fight_prefix):
                continue

            feature_suffix = feature.removeprefix(
                fight_prefix
            )

            if feature_suffix == suffix:
                matches.append(feature)

            break

    if len(matches) != 1:
        raise RFSParameterResolutionError(
            f"{target}: expected one exact evidence feature "
            f"{suffix!r}; found {matches}"
        )

    return matches[0]


def _state_column(
    target: str,
    suffix: str,
    available_columns: set[str],
) -> str | None:
    """Resolve preferred EWM -> expanding -> last3 prior-state column."""

    fight_feature = _find_evidence(
        target,
        suffix,
    )

    for candidate in _state_candidates(
        fight_feature
    ):
        if candidate in available_columns:
            return candidate

    return None


def _finite_float(
    value: object,
) -> float | None:
    """Return a finite float or None."""

    if value is None or pd.isna(value):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(numeric):
        return None

    return numeric


def _profile_value(
    profile: Mapping[str, object],
    target: str,
    suffix: str,
) -> tuple[float | None, str | None]:
    """Resolve one fighter prior-state value."""

    column = _state_column(
        target,
        suffix,
        set(profile),
    )

    if column is None:
        return None, None

    return (
        _finite_float(profile[column]),
        column,
    )


def _population_series(
    population: pd.DataFrame,
    target: str,
    suffix: str,
) -> tuple[pd.Series, str | None]:
    """Resolve one leakage-safe population prior-state distribution."""

    column = _state_column(
        target,
        suffix,
        set(population.columns),
    )

    if column is None:
        return pd.Series(dtype=float), None

    series = pd.to_numeric(
        population[column],
        errors="coerce",
    )

    series = series[
        np.isfinite(series)
    ].astype(float)

    return series, column


# ---------------------------------------------------------------------------
# Reliability / shrinkage
# ---------------------------------------------------------------------------

def _reliability(
    prior_fight_count: int,
) -> float:
    """Convert prior UFC fight count into fighter-specific reliability."""

    count = max(
        0.0,
        float(prior_fight_count),
    )

    return (
        count
        / (
            count
            + SHRINKAGE_PRIOR_FIGHTS
        )
    )


def _estimate(
    *,
    raw_estimate: float | None,
    population_prior: float,
    prior_fight_count: int,
    source_columns: tuple[str, ...],
    lower: float | None = None,
    upper: float | None = None,
) -> ReliabilityShrunkEstimate:
    """Apply reliability shrinkage and optional legal bounds."""

    reliability = _reliability(
        prior_fight_count
    )

    if raw_estimate is None:
        shrunk = float(population_prior)
        used_fallback = True
    else:
        shrunk = (
            reliability * float(raw_estimate)
            + (1.0 - reliability)
            * float(population_prior)
        )
        used_fallback = False

    if lower is not None:
        shrunk = max(
            float(lower),
            shrunk,
        )

    if upper is not None:
        shrunk = min(
            float(upper),
            shrunk,
        )

    return ReliabilityShrunkEstimate(
        raw_estimate=raw_estimate,
        population_prior=float(
            population_prior
        ),
        sample_size=int(
            prior_fight_count
        ),
        effective_sample_size=float(
            prior_fight_count
        ),
        reliability=reliability,
        shrunk_estimate=float(
            shrunk
        ),
        used_fallback=used_fallback,
        source_columns=source_columns,
    )


# ---------------------------------------------------------------------------
# Direct observed targets
# ---------------------------------------------------------------------------

# target:
#   evidence suffix
#   scale into engine units
#   legal lower
#   legal upper

DIRECT_RULES = {
    "transition.takedown_completion_ability": (
        "td_completion_rate",
        1.0,
        0.0,
        1.0,
    ),
    "transition.takedown_resistance": (
        "td_defense_rate",
        1.0,
        0.0,
        1.0,
    ),

    "phase.distance.sig_strike_attempt_rate": (
        "distance_attempts_per_round",
        0.1,  # five-minute round -> ten 30-second segments
        0.0,
        None,
    ),
    "phase.distance.sig_strike_accuracy": (
        "distance_accuracy",
        1.0,
        0.0,
        1.0,
    ),
    "phase.distance.knockdown_probability_per_landed": (
        "knockdowns_per_distance_strike_landed_proxy",
        1.0,
        0.0,
        1.0,
    ),

    "phase.clinch.clinch_strike_attempt_rate": (
        "clinch_attempts_per_round",
        0.1,  # five-minute round -> ten 30-second segments
        0.0,
        None,
    ),
    "phase.clinch.clinch_strike_accuracy": (
        "clinch_accuracy",
        1.0,
        0.0,
        1.0,
    ),
    "phase.clinch.control_seconds_mean": (
        "control_seconds_per_round",
        0.1,
        0.0,
        30.0,
    ),

    "phase.ground_owner.ground_strike_attempt_rate": (
        "ground_attempts_per_round",
        0.1,  # five-minute round -> ten 30-second segments
        0.0,
        None,
    ),
    "phase.ground_owner.ground_strike_accuracy": (
        "ground_accuracy",
        1.0,
        0.0,
        1.0,
    ),
    "phase.ground_owner.control_seconds_mean": (
        "control_seconds_per_round",
        0.1,
        0.0,
        30.0,
    ),
    "phase.ground_owner.submission_attempt_rate": (
        "submission_attempts_per_control_minute",
        0.5,
        0.0,
        None,
    ),

    "phase.ground_defender.reversal_attempt_rate": (
        "reversal_rate_per_opponent_control_min",
        0.5,
        0.0,
        None,
    ),
}


# ---------------------------------------------------------------------------
# Population-relative calibration for selected physical targets
# ---------------------------------------------------------------------------

# Compression controls how far a fighter may move away from the population
# median before being mapped back into physical simulator units.
#
# 1.0 = preserve the original percentile.
# 0.5 = compress percentile distance from the median by 50%.
#
# This first cohort is deliberately narrow so historical A/B validation can
# measure the effect before the calibration layer is expanded.
# ---------------------------------------------------------------------------
# V1 family-level population calibration
# ---------------------------------------------------------------------------
#
# Compression is expressed as retained distance from the population median:
#
#   1.00 = preserve the full population-relative difference
#   0.50 = retain half the distance from the median
#   0.25 = retain one quarter
#
# These values are provisional calibration knobs. They must eventually be
# fitted on a chronological development cohort rather than chosen from the
# final historical validation sample.
CALIBRATION_FAMILY_COMPRESSION = {
    # Current best baseline:
    # direct physical targets compressed to 0.50;
    # latent population-relative traits retain full differentiation.
    "direct_rate": 0.50,
    "direct_probability": 0.50,
    "transition_latent": 1.00,
    "phase_control": 1.00,
    "dynamic": 1.00,
    "finish_direct": 0.50,
    "finish_latent": 1.00,
}


# ---------------------------------------------------------------------------
# V1 empirical percentile-curve shape
#
# Learned from the 500-fight TRAIN cohort and checked against the
# 250-fight DEVELOPMENT cohort.
#
# Only families with acceptable first-pass development behavior are
# activated. Gamma=1.0 preserves the current linear mapping.
#
# This is a shadow calibration experiment, not a production lock.
# ---------------------------------------------------------------------------
CALIBRATION_FAMILY_GAMMA = {
    "direct_rate": 1.00,
    "direct_probability": 1.00,
    "transition_latent": 1.00,

    # Held linear pending stronger evidence.
    "phase_control": 1.00,
    "dynamic": 1.00,
    "finish_direct": 1.00,
    "finish_latent": 1.00,
}


FINISH_DIRECT_TARGETS = {
    "phase.distance.knockdown_probability_per_landed",
    "phase.ground_owner.submission_attempt_rate",
}


FINISH_LATENT_TARGETS = {
    "phase.clinch.damaging_clinch_probability",
    "phase.ground_defender.submission_defense",
    "dynamic.damage_resistance",
    "dynamic.acute_stress_resistance",
}


DIRECT_PROBABILITY_TARGETS = {
    "transition.takedown_completion_ability",
    "transition.takedown_resistance",
    "phase.distance.sig_strike_accuracy",
    "phase.clinch.clinch_strike_accuracy",
    "phase.ground_owner.ground_strike_accuracy",
}


def _calibration_family(target: str) -> str:
    """Return the locked V1 calibration family for one simulator target."""

    if target in FINISH_DIRECT_TARGETS:
        return "finish_direct"

    if target in FINISH_LATENT_TARGETS:
        return "finish_latent"

    if target in DIRECT_PROBABILITY_TARGETS:
        return "direct_probability"

    if target in DIRECT_RULES:
        return "direct_rate"

    if target.startswith("transition."):
        return "transition_latent"

    if target.startswith("dynamic."):
        return "dynamic"

    if target.startswith("phase."):
        return "phase_control"

    raise RFSParameterResolutionError(
        f"{target}: no calibration family assigned"
    )


# ---------------------------------------------------------------------------
# V1 target-specific influence pruning
#
# Historical TRAIN -> DEVELOPMENT audit showed six transition traits with
# repeatable winner signal. Seven weaker/inconsistent transition traits are
# temporarily compressed to 50% influence for a controlled shadow backtest.
#
# All unlisted targets continue to use their family compression.
# ---------------------------------------------------------------------------
TARGET_COMPRESSION_OVERRIDE = {}


def _target_compression(target: str) -> float:
    """Return validated family-level percentile compression."""

    family = _calibration_family(target)

    compression = TARGET_COMPRESSION_OVERRIDE.get(
        target,
        CALIBRATION_FAMILY_COMPRESSION[family],
    )

    if not 0.0 <= compression <= 1.0:
        raise RFSParameterResolutionError(
            f"{target}: compression must be in [0, 1]; "
            f"family={family}, value={compression}"
        )

    return float(compression)


def _shape_population_percentile(
    *,
    target: str,
    percentile: float,
) -> float:
    """Apply family compression and empirical nonlinear curve shape.

    The percentile is expressed relative to the population median.

    gamma < 1:
        differences rise quickly and then saturate

    gamma = 1:
        existing linear compression

    gamma > 1:
        small differences are suppressed while extreme differences
        retain more influence
    """

    family = _calibration_family(target)

    gamma = float(
        CALIBRATION_FAMILY_GAMMA[family]
    )

    compression = _target_compression(
        target
    )

    if gamma <= 0.0:
        raise RFSParameterResolutionError(
            f"{target}: gamma must be > 0; "
            f"family={family}, gamma={gamma}"
        )

    percentile = float(
        min(
            1.0,
            max(
                0.0,
                percentile,
            ),
        )
    )

    delta = (
        percentile
        - 0.5
    )

    if abs(delta) <= 1e-12:
        return 0.5

    sign = (
        1.0
        if delta > 0.0
        else -1.0
    )

    # Convert median distance to [0, 1].
    normalized_distance = min(
        1.0,
        abs(delta) / 0.5,
    )

    shaped_distance = (
        normalized_distance
        ** gamma
    )

    calibrated_percentile = (
        0.5
        + sign
        * 0.5
        * shaped_distance
        * compression
    )

    return float(
        min(
            1.0,
            max(
                0.0,
                calibrated_percentile,
            ),
        )
    )


# All direct physical targets use population-relative quantile calibration.
PHYSICAL_PERCENTILE_COMPRESSION = {
    target: _target_compression(target)
    for target in DIRECT_RULES
}


def _direct_estimate(
    *,
    target: str,
    profile: Mapping[str, object],
    population: pd.DataFrame,
    prior_fight_count: int,
) -> ReliabilityShrunkEstimate:
    """Resolve and shrink one direct physical target."""

    suffix, scale, lower, upper = (
        DIRECT_RULES[target]
    )

    value, profile_column = _profile_value(
        profile,
        target,
        suffix,
    )

    pop, population_column = _population_series(
        population,
        target,
        suffix,
    )

    raw = (
        None
        if value is None
        else float(value) * float(scale)
    )

    scaled_population = (
        pop * float(scale)
    )

    if lower is not None:
        scaled_population = (
            scaled_population.clip(
                lower=lower
            )
        )

    if upper is not None:
        scaled_population = (
            scaled_population.clip(
                upper=upper
            )
        )

    if scaled_population.empty:
        if lower == 0.0 and upper == 1.0:
            prior = 0.5
        else:
            prior = 0.0
    else:
        prior = float(
            scaled_population.median()
        )

    sources = tuple(
        column
        for column in (
            profile_column,
            population_column,
        )
        if column is not None
    )

    return _estimate(
        raw_estimate=raw,
        population_prior=prior,
        prior_fight_count=prior_fight_count,
        source_columns=tuple(
            dict.fromkeys(sources)
        ),
        lower=lower,
        upper=upper,
    )


def _normalized_direct_estimate(
    *,
    target: str,
    profile: Mapping[str, object],
    population: pd.DataFrame,
    prior_fight_count: int,
) -> ReliabilityShrunkEstimate:
    """Resolve a direct target, compress its population rank, and restore units."""

    base = _direct_estimate(
        target=target,
        profile=profile,
        population=population,
        prior_fight_count=prior_fight_count,
    )

    compression = _target_compression(
        target
    )

    suffix, scale, lower, upper = DIRECT_RULES[target]

    pop, _ = _population_series(
        population,
        target,
        suffix,
    )

    scaled_population = (
        pd.to_numeric(pop, errors="coerce")
        .dropna()
        * float(scale)
    )

    if lower is not None:
        scaled_population = scaled_population.clip(
            lower=lower
        )

    if upper is not None:
        scaled_population = scaled_population.clip(
            upper=upper
        )

    if scaled_population.empty:
        return base

    # Rank the already reliability-shrunk physical value against the
    # leakage-safe historical population.
    percentile = float(
        (
            scaled_population
            <= base.shrunk_estimate
        ).mean()
    )

    # Pull extreme population ranks toward the median.
    compressed_percentile = _shape_population_percentile(
        target=target,
        percentile=percentile,
    )

    normalized_value = float(
        scaled_population.quantile(
            compressed_percentile
        )
    )

    if lower is not None:
        normalized_value = max(
            normalized_value,
            lower,
        )

    if upper is not None:
        normalized_value = min(
            normalized_value,
            upper,
        )

    # Preserve the existing ReliabilityShrunkEstimate audit information.
    # raw_estimate remains the original resolved physical observation;
    # shrunk_estimate becomes the calibrated physical engine value.
    return replace(
        base,
        shrunk_estimate=normalized_value,
    )


# ---------------------------------------------------------------------------
# Population-relative latent targets
# ---------------------------------------------------------------------------

# Directions:
#   +1 = higher observed value supports a higher simulator target.
#   -1 = higher observed value supports a lower simulator target.
#
# Exposure/count-only evidence is deliberately omitted from scoring.

LATENT_SIGNALS = {
    "transition.distance_retention": (
        ("distance_attempt_share", +1),
        ("distance_pressure_share", +1),
        ("non_distance_attempt_share_allowed", -1),
        ("td_attempts_allowed_per_round", -1),
        ("control_seconds_allowed_per_round", -1),
        ("control_share", +1),
    ),
    "transition.clinch_entry_tendency": (
        ("clinch_attempts_per_round", +1),
        ("clinch_attempt_share", +1),
        ("non_distance_clinch_share", +1),
    ),
    "transition.clinch_entry_resistance": (
        ("clinch_attempt_share_allowed", -1),
        ("clinch_attempts_allowed_per_round", -1),
        ("control_seconds_allowed_per_round", -1),
        ("distance_pressure_share", +1),
    ),
    "transition.takedown_entry_tendency": (
        ("td_attempts_per_round", +1),
        ("failed_td_attempts_per_round", +1),
    ),
    "transition.takedown_persistence": (
        ("td_attempt_slope", +1),
        ("td_persistence_ratio", +1),
        ("td_attempts_per_round", +1),
    ),
    "transition.failed_takedown_persistence": (
        ("failed_td_attempt_slope", +1),
        ("failed_td_attempts_per_round", +1),
    ),
    "transition.clinch_retention": (
        ("clinch_pressure_share", +1),
        ("control_share", +1),
        ("control_exchange_balance", +1),
        ("reversal_allowed_per_control_min", -1),
    ),
    "transition.clinch_escape_ability": (
        ("clinch_attempt_share_allowed", -1),
        ("control_seconds_allowed_per_round", -1),
        ("reversal_rate_per_opponent_control_min", +1),
        ("distance_pressure_share", +1),
    ),
    "transition.ground_retention": (
        ("ground_pressure_share", +1),
        ("control_share", +1),
        ("ground_landed_per_control_min", +1),
        ("sub_attempts_per_control_min", +1),
        ("reversal_allowed_per_control_min", -1),
    ),
    "transition.ground_escape_ability": (
        ("control_seconds_allowed_per_round", -1),
        ("ground_landed_allowed_per_control_min", -1),
        ("sub_attempts_allowed_per_control_min", -1),
        ("reversal_rate_per_opponent_control_min", +1),
        ("distance_pressure_share", +1),
    ),
    "transition.reversal_ability": (
        ("reversal_rate_per_opponent_control_min", +1),
    ),
    "transition.phase_imposition": (
        ("non_distance_attempt_share", +1),
        ("clinch_pressure_share", +1),
        ("ground_pressure_share", +1),
        ("td_pressure_share", +1),
        ("control_share", +1),
    ),
    "transition.phase_resistance": (
        ("non_distance_attempt_share_allowed", -1),
        ("td_defense_rate", +1),
        ("distance_pressure_share", +1),
        ("control_seconds_allowed_per_round", -1),
        ("control_exchange_balance", +1),
    ),

    "phase.clinch.damaging_clinch_probability": (
        ("clinch_strike_accuracy", +1),
        ("clinch_damage_output_per_round", +1),
    ),
    "phase.ground_owner.position_advancement_probability": (
        ("ground_pressure_share", +1),
        ("control_share", +1),
        ("ground_landed_per_control_min", +1),
        ("sub_attempts_per_control_min", +1),
        ("reversal_allowed_per_control_min", -1),
    ),
    "phase.ground_defender.escape_attempt_rate": (
        ("reversal_rate_per_opponent_control_min", +1),
        ("distance_pressure_share", +1),
        ("ground_attempts_allowed_per_control_min", +1),
    ),
    "phase.ground_defender.scramble_attempt_rate": (
        ("combined_reversals_per_control_min", +1),
        ("reversal_rate_per_opponent_control_min", +1),
        ("control_exchange_balance", +1),
    ),
    "phase.ground_defender.submission_defense": (
        ("submission_loss_indicator", -1),
        ("submission_survival_indicator", +1),
    ),

    "dynamic.fatigue_accumulation_resistance": (
        ("sig_strike_attempt_slope", +1),
        ("total_strike_attempt_slope", +1),
        ("td_attempt_slope", +1),
        ("control_seconds_slope", +1),
        ("sig_strike_attempt_first_last_ratio", +1),
        ("total_strike_attempt_first_last_ratio", +1),
        ("late_early_workload_ratio", +1),
        ("late_early_workload_difference", +1),
    ),
    "dynamic.fatigue_performance_resilience": (
        ("sig_strike_landed_slope", +1),
        ("total_strike_landed_slope", +1),
        ("sig_strike_accuracy_change", +1),
        ("total_strike_accuracy_change", +1),
        ("late_early_output_ratio", +1),
        ("late_early_workload_ratio", +1),
    ),
    "dynamic.recovery_ability": (
        ("post_adversity_sig_strike_rebound", +1),
        ("post_adversity_output_rebound", +1),
        ("post_adversity_efficiency_preservation", +1),
        ("late_early_output_ratio", +1),
    ),
    "dynamic.damage_resistance": (
        ("ko_tko_loss_indicator", -1),
        ("ko_tko_survival_indicator", +1),
    ),
    "dynamic.acute_stress_resistance": (
        ("same_round_output_preservation", +1),
        ("same_round_efficiency_preservation", +1),
        ("knockdowns_absorbed", -1),
    ),
    "dynamic.acute_stress_recovery": (
        ("post_adversity_sig_strike_rebound", +1),
        ("post_adversity_output_rebound", +1),
        ("post_adversity_efficiency_preservation", +1),
    ),
}


# Latent targets whose engine semantics are not generic strengths.
# Population percentile is mapped into a conservative V0 physical range.

LATENT_OUTPUT_RANGES = {
    "phase.clinch.damaging_clinch_probability": (
        0.03,
        0.25,
    ),
    "phase.ground_owner.position_advancement_probability": (
        0.10,
        0.60,
    ),
    "phase.ground_defender.escape_attempt_rate": (
        0.10,
        0.90,
    ),
    "phase.ground_defender.scramble_attempt_rate": (
        0.05,
        0.70,
    ),
    "phase.ground_defender.submission_defense": (
        0.50,
        0.95,
    ),
}


def _percentile(
    value: float,
    population: pd.Series,
) -> float | None:
    """Return empirical percentile of one value."""

    if population.empty:
        return None

    return float(
        (
            population <= value
        ).mean()
    )


def _latent_raw_score(
    *,
    target: str,
    profile: Mapping[str, object],
    population: pd.DataFrame,
) -> tuple[float | None, tuple[str, ...]]:
    """Calculate population-relative composite score for one target."""

    component_scores: list[float] = []
    source_columns: list[str] = []

    for suffix, direction in LATENT_SIGNALS[target]:
        value, profile_column = _profile_value(
            profile,
            target,
            suffix,
        )

        pop, population_column = _population_series(
            population,
            target,
            suffix,
        )

        if (
            value is None
            or pop.empty
        ):
            continue

        percentile = _percentile(
            value,
            pop,
        )

        if percentile is None:
            continue

        if direction < 0:
            percentile = (
                1.0 - percentile
            )

        component_scores.append(
            percentile
        )

        for column in (
            profile_column,
            population_column,
        ):
            if (
                column is not None
                and column not in source_columns
            ):
                source_columns.append(
                    column
                )

    if not component_scores:
        return None, tuple(
            source_columns
        )

    return (
        float(
            np.mean(
                component_scores
            )
        ),
        tuple(
            source_columns
        ),
    )


def _latent_estimate(
    *,
    target: str,
    profile: Mapping[str, object],
    population: pd.DataFrame,
    prior_fight_count: int,
) -> ReliabilityShrunkEstimate:
    """Resolve one population-relative latent target."""

    score, sources = _latent_raw_score(
        target=target,
        profile=profile,
        population=population,
    )

    # Population percentile is centered on 0.5. Compress extremes so
    # historical state differences remain meaningful without translating
    # linearly into extreme simulator advantages.
    compression = _target_compression(
        target
    )

    calibrated_score = (
        None
        if score is None
        else _shape_population_percentile(
            target=target,
            percentile=score,
        )
    )

    output_range = LATENT_OUTPUT_RANGES.get(
        target
    )

    if output_range is None:
        raw = calibrated_score
        prior = 0.5
        lower = 0.0
        upper = 1.0
    else:
        lower, upper = output_range
        midpoint = (
            lower
            + upper
        ) / 2.0

        raw = (
            None
            if calibrated_score is None
            else (
                lower
                + calibrated_score
                * (
                    upper
                    - lower
                )
            )
        )

        prior = midpoint

    return _estimate(
        raw_estimate=raw,
        population_prior=prior,
        prior_fight_count=prior_fight_count,
        source_columns=sources,
        lower=lower,
        upper=upper,
    )


# ---------------------------------------------------------------------------
# Public calibration API
# ---------------------------------------------------------------------------

ALL_TARGETS = (
    set(DIRECT_RULES)
    | set(LATENT_SIGNALS)
)


def resolve_fighter_parameters(
    *,
    profile: Mapping[str, object],
    prior_fight_count: int,
    population_history: pd.DataFrame,
) -> CalibratedFighterParameters:
    """Convert one leakage-safe RFS profile into engine parameter objects."""

    if prior_fight_count < 0:
        raise ValueError(
            "prior_fight_count cannot be negative"
        )

    if population_history.empty:
        raise RFSParameterResolutionError(
            "population_history cannot be empty"
        )

    estimates: dict[
        str,
        ReliabilityShrunkEstimate,
    ] = {}

    for target in DIRECT_RULES:
        if target in PHYSICAL_PERCENTILE_COMPRESSION:
            estimates[target] = _normalized_direct_estimate(
                target=target,
                profile=profile,
                population=population_history,
                prior_fight_count=prior_fight_count,
            )
        else:
            estimates[target] = _direct_estimate(
                target=target,
                profile=profile,
                population=population_history,
                prior_fight_count=prior_fight_count,
            )

    for target in LATENT_SIGNALS:
        estimates[target] = _latent_estimate(
            target=target,
            profile=profile,
            population=population_history,
            prior_fight_count=prior_fight_count,
        )

    expected_target_count = 37

    if len(estimates) != expected_target_count:
        raise RFSParameterResolutionError(
            f"expected {expected_target_count} calibrated targets; "
            f"resolved {len(estimates)}"
        )

    value = lambda target: (
        estimates[target].shrunk_estimate
    )

    transition = FighterTransitionParameters(
        distance_retention=value(
            "transition.distance_retention"
        ),
        clinch_entry_tendency=value(
            "transition.clinch_entry_tendency"
        ),
        clinch_entry_resistance=value(
            "transition.clinch_entry_resistance"
        ),
        takedown_entry_tendency=value(
            "transition.takedown_entry_tendency"
        ),
        takedown_completion_ability=value(
            "transition.takedown_completion_ability"
        ),
        takedown_resistance=value(
            "transition.takedown_resistance"
        ),
        takedown_persistence=value(
            "transition.takedown_persistence"
        ),
        failed_takedown_persistence=value(
            "transition.failed_takedown_persistence"
        ),
        clinch_retention=value(
            "transition.clinch_retention"
        ),
        clinch_escape_ability=value(
            "transition.clinch_escape_ability"
        ),
        ground_retention=value(
            "transition.ground_retention"
        ),
        ground_escape_ability=value(
            "transition.ground_escape_ability"
        ),
        reversal_ability=value(
            "transition.reversal_ability"
        ),
        phase_imposition=value(
            "transition.phase_imposition"
        ),
        phase_resistance=value(
            "transition.phase_resistance"
        ),
    )

    phase = FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=value(
                "phase.distance.sig_strike_attempt_rate"
            ),
            sig_strike_accuracy=value(
                "phase.distance.sig_strike_accuracy"
            ),
            knockdown_probability_per_landed=value(
                "phase.distance.knockdown_probability_per_landed"
            ),
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=value(
                "phase.clinch.clinch_strike_attempt_rate"
            ),
            clinch_strike_accuracy=value(
                "phase.clinch.clinch_strike_accuracy"
            ),
            control_seconds_mean=value(
                "phase.clinch.control_seconds_mean"
            ),
            damaging_clinch_probability=value(
                "phase.clinch.damaging_clinch_probability"
            ),
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=value(
                "phase.ground_owner.ground_strike_attempt_rate"
            ),
            ground_strike_accuracy=value(
                "phase.ground_owner.ground_strike_accuracy"
            ),
            control_seconds_mean=value(
                "phase.ground_owner.control_seconds_mean"
            ),
            submission_attempt_rate=value(
                "phase.ground_owner.submission_attempt_rate"
            ),
            position_advancement_probability=value(
                "phase.ground_owner.position_advancement_probability"
            ),
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=value(
                "phase.ground_defender.escape_attempt_rate"
            ),
            reversal_attempt_rate=value(
                "phase.ground_defender.reversal_attempt_rate"
            ),
            scramble_attempt_rate=value(
                "phase.ground_defender.scramble_attempt_rate"
            ),
            submission_defense=value(
                "phase.ground_defender.submission_defense"
            ),
        ),
    )

    dynamic = FighterDynamicParameters(
        fatigue_accumulation_resistance=value(
            "dynamic.fatigue_accumulation_resistance"
        ),
        fatigue_performance_resilience=value(
            "dynamic.fatigue_performance_resilience"
        ),
        recovery_ability=value(
            "dynamic.recovery_ability"
        ),
        damage_resistance=value(
            "dynamic.damage_resistance"
        ),
        acute_stress_resistance=value(
            "dynamic.acute_stress_resistance"
        ),
        acute_stress_recovery=value(
            "dynamic.acute_stress_recovery"
        ),
    )

    return CalibratedFighterParameters(
        transition=transition,
        phase=phase,
        dynamic=dynamic,
        estimates=estimates,
    )
