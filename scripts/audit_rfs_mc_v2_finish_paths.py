"""Population audit for RFS Monte Carlo V2 finish-enabled paths.

This audit verifies:

- zero finish calibration preserves scheduled-distance paths
- zero finish calibration preserves the existing dynamic timeline
- seeded replay is deterministic
- stronger KO/TKO calibration increases finish rate
- stronger KO/TKO calibration produces earlier finishes
- sampled finish timing remains within legal segment bounds
- submission sampling matches deterministic probability
- submission finishes remain legal to the authoritative ground owner
- finished paths stop immediately
- unfinished paths contain every scheduled segment

These are structural and directional checks. They do not establish
production-calibrated UFC finish rates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from statistics import mean

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import FightPhase
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    SEGMENTS_PER_ROUND,
    FighterSide,
    SharedFightState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_calibration import (
    ActivityWorkloadCalibration,
    AdversityCalibration,
    DynamicStateCalibration,
    PhaseWorkloadCalibration,
    RecoveryCalibration,
    ResistanceScalingCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_effect_calibration import (
    DynamicEffectCalibration,
    StatePenaltyWeights,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_path_runner import (
    run_dynamic_activity_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FightDynamicState,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_transition_effect_calibration import (
    DynamicTransitionEffectCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_calibration import (
    FinishProbabilityCalibration,
    KnockoutFinishCalibration,
    SubmissionFinishCalibration,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import (
    FinishMethod,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_contracts import (
    FinishEnabledDynamicPath,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_runner import (
    run_finish_enabled_dynamic_path,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_probability import (
    FighterSegmentFinishProbabilities,
    SegmentFinishProbabilities,
    calculate_segment_finish_probabilities,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_sampler import (
    sample_segment_finish,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundFighterActivity,
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


@dataclass(frozen=True)
class AuditCheck:
    """One finish-system audit result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class FinishPopulationSummary:
    """Aggregated results from one full-path finish scenario."""

    path_count: int
    finish_count: int
    ko_tko_count: int
    submission_count: int
    scheduled_distance_count: int
    mean_finishing_segment: float | None
    method_counts: Counter[str]
    round_counts: Counter[int]
    structural_violations: int

    @property
    def finish_rate(self) -> float:
        """Return the fraction of paths ending by finish."""

        return self.finish_count / self.path_count


def zero_weights() -> StatePenaltyWeights:
    """Return a capability family with no dynamic penalty."""

    return StatePenaltyWeights(
        fatigue=0.0,
        damage=0.0,
        acute_stress=0.0,
    )


def distance_only_transition_parameters() -> FighterTransitionParameters:
    """Return transition parameters that preserve distance."""

    return FighterTransitionParameters(
        distance_retention=1.0,
        clinch_entry_tendency=0.0,
        clinch_entry_resistance=0.0,
        takedown_entry_tendency=0.0,
        takedown_completion_ability=0.0,
        takedown_resistance=0.0,
        takedown_persistence=0.0,
        failed_takedown_persistence=0.0,
        clinch_retention=0.0,
        clinch_escape_ability=0.0,
        ground_retention=0.0,
        ground_escape_ability=0.0,
        reversal_ability=0.0,
        phase_imposition=0.0,
        phase_resistance=0.0,
    )


def phase_parameters(
    *,
    distance_attempt_rate: float = 12.0,
    distance_accuracy: float = 0.65,
    distance_knockdown_probability: float = 0.10,
    submission_defense: float = 0.25,
) -> FighterPhaseParameters:
    """Return a controlled complete phase profile."""

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=distance_attempt_rate,
            sig_strike_accuracy=distance_accuracy,
            knockdown_probability_per_landed=(
                distance_knockdown_probability
            ),
        ),
        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=1.5,
            clinch_strike_accuracy=0.50,
            control_seconds_mean=8.0,
            damaging_clinch_probability=0.05,
        ),
        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=2.0,
            ground_strike_accuracy=0.50,
            control_seconds_mean=15.0,
            submission_attempt_rate=0.20,
            position_advancement_probability=0.25,
        ),
        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=0.20,
            reversal_attempt_rate=0.08,
            scramble_attempt_rate=0.15,
            submission_defense=submission_defense,
        ),
    )


def dynamic_parameters() -> FighterDynamicParameters:
    """Return a neutral dynamic-response profile."""

    return FighterDynamicParameters(
        fatigue_accumulation_resistance=0.0,
        fatigue_performance_resilience=0.0,
        recovery_ability=0.0,
        damage_resistance=0.0,
        acute_stress_resistance=0.0,
        acute_stress_recovery=0.0,
    )


def zero_state_calibration() -> DynamicStateCalibration:
    """Return state calibration with no accumulation or recovery."""

    return DynamicStateCalibration(
        phase_workload=PhaseWorkloadCalibration(
            distance=0.0,
            clinch_owner=0.0,
            clinch_defender=0.0,
            ground_owner=0.0,
            ground_defender=0.0,
        ),
        activity_workload=ActivityWorkloadCalibration(
            strike_attempt=0.0,
            control_second=0.0,
            submission_attempt=0.0,
            position_advancement=0.0,
            escape_attempt=0.0,
            reversal_attempt=0.0,
            scramble_attempt=0.0,
        ),
        adversity=AdversityCalibration(
            distance_landed_damage=0.0,
            clinch_landed_damage=0.0,
            damaging_clinch_bonus_damage=0.0,
            ground_landed_damage=0.0,
            knockdown_damage=0.0,
            distance_landed_stress=0.0,
            clinch_landed_stress=0.0,
            damaging_clinch_bonus_stress=0.0,
            ground_landed_stress=0.0,
            knockdown_stress=0.0,
            control_second_received_stress=0.0,
            submission_attempt_received_stress=0.0,
            position_advancement_received_stress=0.0,
        ),
        resistance_scaling=ResistanceScalingCalibration(
            minimum_fatigue_accumulation_multiplier=0.25,
            minimum_damage_accumulation_multiplier=0.20,
            minimum_acute_stress_accumulation_multiplier=0.15,
        ),
        recovery=RecoveryCalibration(
            low_workload_threshold=0.0,
            segment_fatigue_recovery=0.0,
            round_break_fatigue_recovery=0.0,
            segment_acute_stress_recovery=0.0,
            round_break_acute_stress_recovery=0.0,
        ),
    )


def zero_phase_effect_calibration() -> DynamicEffectCalibration:
    """Return phase effects that preserve baselines."""

    return DynamicEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_capability_multiplier=0.10,
        output=zero_weights(),
        accuracy=zero_weights(),
        power=zero_weights(),
        control=zero_weights(),
        grappling=zero_weights(),
        defense=zero_weights(),
    )


def zero_transition_effect_calibration(
) -> DynamicTransitionEffectCalibration:
    """Return transition effects that preserve baselines."""

    return DynamicTransitionEffectCalibration(
        minimum_fatigue_effect_multiplier=0.20,
        minimum_effective_transition_multiplier=0.10,
        entry=zero_weights(),
        completion=zero_weights(),
        retention=zero_weights(),
        escape=zero_weights(),
        reversal=zero_weights(),
        persistence=zero_weights(),
        imposition=zero_weights(),
        resistance=zero_weights(),
    )


def zero_submission_calibration() -> SubmissionFinishCalibration:
    """Return submission calibration producing zero probability."""

    return SubmissionFinishCalibration(
        base_probability_per_attempt=0.0,
        position_quality_amplifier=0.0,
        minimum_submission_defense_effect_multiplier=0.10,
        defender_fatigue_amplifier=0.0,
        defender_damage_amplifier=0.0,
        defender_acute_stress_amplifier=0.0,
        maximum_probability_per_attempt=1.0,
        maximum_segment_probability=1.0,
    )


def zero_finish_calibration() -> FinishProbabilityCalibration:
    """Return calibration producing no finishes."""

    return FinishProbabilityCalibration(
        knockout=KnockoutFinishCalibration(
            distance_landed_probability=0.0,
            distance_knockdown_probability=0.0,
            clinch_landed_probability=0.0,
            damaging_clinch_probability=0.0,
            ground_landed_probability=0.0,
            defender_fatigue_amplifier=0.0,
            defender_damage_amplifier=0.0,
            defender_acute_stress_amplifier=0.0,
            maximum_segment_probability=1.0,
        ),
        submission=zero_submission_calibration(),
    )


def knockout_finish_calibration(
    *,
    landed_probability: float,
    knockdown_probability: float,
) -> FinishProbabilityCalibration:
    """Return distance-only KO/TKO calibration."""

    return FinishProbabilityCalibration(
        knockout=KnockoutFinishCalibration(
            distance_landed_probability=landed_probability,
            distance_knockdown_probability=(
                knockdown_probability
            ),
            clinch_landed_probability=0.0,
            damaging_clinch_probability=0.0,
            ground_landed_probability=0.0,
            defender_fatigue_amplifier=0.0,
            defender_damage_amplifier=0.0,
            defender_acute_stress_amplifier=0.0,
            maximum_segment_probability=0.95,
        ),
        submission=zero_submission_calibration(),
    )


def submission_finish_calibration() -> FinishProbabilityCalibration:
    """Return controlled submission-only calibration."""

    return FinishProbabilityCalibration(
        knockout=KnockoutFinishCalibration(
            distance_landed_probability=0.0,
            distance_knockdown_probability=0.0,
            clinch_landed_probability=0.0,
            damaging_clinch_probability=0.0,
            ground_landed_probability=0.0,
            defender_fatigue_amplifier=0.0,
            defender_damage_amplifier=0.0,
            defender_acute_stress_amplifier=0.0,
            maximum_segment_probability=1.0,
        ),
        submission=SubmissionFinishCalibration(
            base_probability_per_attempt=0.35,
            position_quality_amplifier=0.40,
            minimum_submission_defense_effect_multiplier=0.10,
            defender_fatigue_amplifier=0.0,
            defender_damage_amplifier=0.0,
            defender_acute_stress_amplifier=0.0,
            maximum_probability_per_attempt=1.0,
            maximum_segment_probability=1.0,
        ),
    )


def run_path(
    *,
    seed: int,
    finish_calibration: FinishProbabilityCalibration,
) -> FinishEnabledDynamicPath:
    """Run one controlled distance-only finish-enabled path."""

    transition = distance_only_transition_parameters()
    phase = phase_parameters()
    dynamic = dynamic_parameters()

    return run_finish_enabled_dynamic_path(
        transition,
        transition,
        phase,
        phase,
        dynamic,
        dynamic,
        dynamic_state_calibration=zero_state_calibration(),
        phase_effect_calibration=zero_phase_effect_calibration(),
        transition_effect_calibration=(
            zero_transition_effect_calibration()
        ),
        finish_probability_calibration=finish_calibration,
        scheduled_rounds=3,
        seed=seed,
    )


def count_path_violations(
    path: FinishEnabledDynamicPath,
) -> int:
    """Count finish, timeline, and terminal-state violations."""

    violations = 0
    maximum_segments = (
        path.scheduled_rounds
        * SEGMENTS_PER_ROUND
    )

    if path.finish is None:
        if len(path.segments) != maximum_segments:
            violations += 1

        if not path.reached_scheduled_distance:
            violations += 1

        if any(
            record.finish is not None
            for record in path.segments
        ):
            violations += 1

    else:
        if path.reached_scheduled_distance:
            violations += 1

        if path.segments[-1].finish != path.finish:
            violations += 1

        if any(
            record.finish is not None
            for record in path.segments[:-1]
        ):
            violations += 1

        terminal = path.segments[-1]

        if terminal.transition is not None:
            violations += 1

        if terminal.red_effective_transition is not None:
            violations += 1

        if terminal.blue_effective_transition is not None:
            violations += 1

        if terminal.round_break_recovery_applied:
            violations += 1

        if (
            terminal.dynamic_state_after_segment
            != terminal.dynamic_state_after_activity
        ):
            violations += 1

        if not 1 <= path.finish.elapsed_seconds_in_segment <= 30:
            violations += 1

        if not 1 <= path.finish.elapsed_seconds_in_round <= 300:
            violations += 1

        if path.finish.method is FinishMethod.SUBMISSION:
            if path.finish.state.phase is not FightPhase.GROUND:
                violations += 1

            if (
                path.finish.state.phase_owner
                is not path.finish.winner
            ):
                violations += 1

    for index, record in enumerate(path.segments):
        if record.activity.state != record.state:
            violations += 1

        if record.exposure.state != record.state:
            violations += 1

        if record.finish_probabilities.state != record.state:
            violations += 1

        if index == len(path.segments) - 1:
            continue

        following = path.segments[index + 1]

        if record.finish is not None:
            violations += 1

        if (
            record.dynamic_state_after_segment
            != following.dynamic_state_before
        ):
            violations += 1

        if record.state.segment_number < SEGMENTS_PER_ROUND:
            if (
                record.transition is None
                or record.transition.next_state
                != following.state
            ):
                violations += 1

    return violations


def summarize_population(
    *,
    path_count: int,
    seed_start: int,
    calibration: FinishProbabilityCalibration,
) -> FinishPopulationSummary:
    """Run and summarize one finish-enabled path population."""

    finish_count = 0
    ko_tko_count = 0
    submission_count = 0
    scheduled_distance_count = 0
    finishing_segments: list[int] = []
    method_counts: Counter[str] = Counter()
    round_counts: Counter[int] = Counter()
    structural_violations = 0

    for path_index in range(path_count):
        path = run_path(
            seed=seed_start + path_index,
            finish_calibration=calibration,
        )

        structural_violations += count_path_violations(
            path
        )

        if path.finish is None:
            scheduled_distance_count += 1
            continue

        finish_count += 1
        method_counts[path.finish.method.value] += 1
        round_counts[path.finish.round_number] += 1

        finishing_segments.append(
            (
                path.finish.round_number - 1
            )
            * SEGMENTS_PER_ROUND
            + path.finish.segment_number
        )

        if path.finish.method is FinishMethod.KO_TKO:
            ko_tko_count += 1
        elif path.finish.method is FinishMethod.SUBMISSION:
            submission_count += 1

    return FinishPopulationSummary(
        path_count=path_count,
        finish_count=finish_count,
        ko_tko_count=ko_tko_count,
        submission_count=submission_count,
        scheduled_distance_count=scheduled_distance_count,
        mean_finishing_segment=(
            mean(finishing_segments)
            if finishing_segments
            else None
        ),
        method_counts=method_counts,
        round_counts=round_counts,
        structural_violations=structural_violations,
    )


def count_zero_finish_equivalence_violations(
    *,
    path_count: int,
    seed_start: int,
) -> int:
    """Compare zero-finish paths with the original dynamic runner."""

    violations = 0
    transition = distance_only_transition_parameters()
    phase = phase_parameters()
    dynamic = dynamic_parameters()
    state_calibration = zero_state_calibration()
    phase_effect = zero_phase_effect_calibration()
    transition_effect = zero_transition_effect_calibration()

    for path_index in range(path_count):
        seed = seed_start + path_index

        baseline = run_dynamic_activity_path(
            transition,
            transition,
            phase,
            phase,
            dynamic,
            dynamic,
            dynamic_state_calibration=state_calibration,
            phase_effect_calibration=phase_effect,
            transition_effect_calibration=transition_effect,
            scheduled_rounds=3,
            seed=seed,
        )

        finish_enabled = run_finish_enabled_dynamic_path(
            transition,
            transition,
            phase,
            phase,
            dynamic,
            dynamic,
            dynamic_state_calibration=state_calibration,
            phase_effect_calibration=phase_effect,
            transition_effect_calibration=transition_effect,
            finish_probability_calibration=(
                zero_finish_calibration()
            ),
            scheduled_rounds=3,
            seed=seed,
        )

        if len(baseline.segments) != len(
            finish_enabled.segments
        ):
            violations += 1
            continue

        for expected, actual in zip(
            baseline.segments,
            finish_enabled.segments,
            strict=True,
        ):
            comparable_actual = (
                actual.state,
                actual.dynamic_state_before,
                actual.red_effective_phase,
                actual.blue_effective_phase,
                actual.activity,
                actual.exposure,
                actual.dynamic_state_after_activity,
                actual.red_effective_transition,
                actual.blue_effective_transition,
                actual.transition,
                actual.round_break_recovery_applied,
                actual.dynamic_state_after_segment,
            )
            comparable_expected = (
                expected.state,
                expected.dynamic_state_before,
                expected.red_effective_phase,
                expected.blue_effective_phase,
                expected.activity,
                expected.exposure,
                expected.dynamic_state_after_activity,
                expected.red_effective_transition,
                expected.blue_effective_transition,
                expected.transition,
                expected.round_break_recovery_applied,
                expected.dynamic_state_after_segment,
            )

            if comparable_actual != comparable_expected:
                violations += 1

    return violations


def build_submission_activity() -> GroundSegmentActivity:
    """Build one controlled legal ground submission opportunity."""

    state = SharedFightState(
        phase=FightPhase.GROUND,
        phase_owner=FighterSide.RED,
        phase_age_segments=1,
        position_quality=0.50,
        round_number=1,
        segment_number=5,
    )

    return GroundSegmentActivity(
        state=state,
        red=GroundFighterActivity(
            ground_str_attempted=0,
            ground_str_landed=0,
            control_seconds=20,
            submission_attempts=1,
            position_advancements=0,
            escape_attempts=0,
            reversal_attempts=0,
            scramble_attempts=0,
        ),
        blue=GroundFighterActivity(
            ground_str_attempted=0,
            ground_str_landed=0,
            control_seconds=0,
            submission_attempts=0,
            position_advancements=0,
            escape_attempts=0,
            reversal_attempts=0,
            scramble_attempts=0,
        ),
    )


def audit_submission_sampling(
    *,
    sample_count: int,
    seed_start: int,
) -> tuple[float, float, int]:
    """Audit sampled submissions against deterministic probability."""

    activity = build_submission_activity()
    red_phase = phase_parameters(
        submission_defense=0.25,
    )
    blue_phase = phase_parameters(
        submission_defense=0.25,
    )
    calibration = submission_finish_calibration()

    probabilities = calculate_segment_finish_probabilities(
        activity,
        FightDynamicState.opening_state(),
        red_phase,
        blue_phase,
        calibration,
    )

    expected_probability = (
        probabilities.red.submission_probability
    )
    sampled_finishes = 0
    violations = 0

    for sample_index in range(sample_count):
        finish = sample_segment_finish(
            probabilities,
            np.random.default_rng(
                seed_start + sample_index
            ),
        )

        if finish is None:
            continue

        sampled_finishes += 1

        if finish.method is not FinishMethod.SUBMISSION:
            violations += 1

        if finish.winner is not FighterSide.RED:
            violations += 1

        if finish.state.phase is not FightPhase.GROUND:
            violations += 1

        if finish.state.phase_owner is not FighterSide.RED:
            violations += 1

    observed_probability = (
        sampled_finishes / sample_count
    )

    return (
        expected_probability,
        observed_probability,
        violations,
    )


def audit_finish_timing(
    *,
    sample_count: int,
    seed_start: int,
) -> tuple[float, int, int, int]:
    """Audit legal and approximately uniform finish-second sampling."""

    probabilities = SegmentFinishProbabilities(
        state=SharedFightState.opening_state(
            round_number=1,
        ),
        red=FighterSegmentFinishProbabilities(
            ko_tko_probability=1.0,
            submission_probability=0.0,
        ),
        blue=FighterSegmentFinishProbabilities(
            ko_tko_probability=0.0,
            submission_probability=0.0,
        ),
    )

    sampled_seconds: list[int] = []
    violations = 0

    for sample_index in range(sample_count):
        finish = sample_segment_finish(
            probabilities,
            np.random.default_rng(
                seed_start + sample_index
            ),
        )

        if finish is None:
            violations += 1
            continue

        sampled_seconds.append(
            finish.elapsed_seconds_in_segment
        )

        if not 1 <= finish.elapsed_seconds_in_segment <= 30:
            violations += 1

    return (
        mean(sampled_seconds),
        min(sampled_seconds),
        max(sampled_seconds),
        violations,
    )


def run_audit(
    *,
    path_count: int,
    seed_start: int,
) -> int:
    """Run all finish population audit scenarios."""

    if path_count <= 0:
        raise ValueError(
            "path_count must be positive"
        )

    replay_path_count = min(
        path_count,
        100,
    )
    sampler_count = max(
        path_count * 5,
        5_000,
    )

    zero_summary = summarize_population(
        path_count=path_count,
        seed_start=seed_start,
        calibration=zero_finish_calibration(),
    )

    low_ko_summary = summarize_population(
        path_count=path_count,
        seed_start=seed_start + 100_000,
        calibration=knockout_finish_calibration(
            landed_probability=0.0005,
            knockdown_probability=0.0,
        ),
    )

    high_ko_summary = summarize_population(
        path_count=path_count,
        seed_start=seed_start + 200_000,
        calibration=knockout_finish_calibration(
            landed_probability=0.15,
            knockdown_probability=0.60,
        ),
    )

    equivalence_violations = (
        count_zero_finish_equivalence_violations(
            path_count=replay_path_count,
            seed_start=seed_start + 300_000,
        )
    )

    deterministic_replay_violations = 0

    for path_index in range(replay_path_count):
        seed = seed_start + 400_000 + path_index

        first = run_path(
            seed=seed,
            finish_calibration=(
                knockout_finish_calibration(
                    landed_probability=0.05,
                    knockdown_probability=0.30,
                )
            ),
        )
        second = run_path(
            seed=seed,
            finish_calibration=(
                knockout_finish_calibration(
                    landed_probability=0.05,
                    knockdown_probability=0.30,
                )
            ),
        )

        if first != second:
            deterministic_replay_violations += 1

    (
        expected_submission_probability,
        observed_submission_probability,
        submission_violations,
    ) = audit_submission_sampling(
        sample_count=sampler_count,
        seed_start=seed_start + 500_000,
    )

    (
        mean_finish_second,
        minimum_finish_second,
        maximum_finish_second,
        timing_violations,
    ) = audit_finish_timing(
        sample_count=sampler_count,
        seed_start=seed_start + 600_000,
    )

    high_mean = high_ko_summary.mean_finishing_segment
    low_mean = low_ko_summary.mean_finishing_segment

    checks = [
        AuditCheck(
            name="zero calibration produces no finishes",
            passed=zero_summary.finish_count == 0,
            detail=(
                f"finishes={zero_summary.finish_count}, "
                f"scheduled distance="
                f"{zero_summary.scheduled_distance_count}"
            ),
        ),
        AuditCheck(
            name="zero calibration preserves structural legality",
            passed=zero_summary.structural_violations == 0,
            detail=(
                f"violations="
                f"{zero_summary.structural_violations}"
            ),
        ),
        AuditCheck(
            name="zero calibration preserves dynamic timeline",
            passed=equivalence_violations == 0,
            detail=(
                f"compared paths={replay_path_count}, "
                f"violations={equivalence_violations}"
            ),
        ),
        AuditCheck(
            name="seeded full-path replay is deterministic",
            passed=deterministic_replay_violations == 0,
            detail=(
                f"replayed paths={replay_path_count}, "
                f"violations="
                f"{deterministic_replay_violations}"
            ),
        ),
        AuditCheck(
            name="stronger KO calibration increases finish rate",
            passed=(
                high_ko_summary.finish_rate
                > low_ko_summary.finish_rate + 0.50
            ),
            detail=(
                f"low={low_ko_summary.finish_rate:.2%}, "
                f"high={high_ko_summary.finish_rate:.2%}"
            ),
        ),
        AuditCheck(
            name="high KO calibration produces only KO/TKO finishes",
            passed=(
                high_ko_summary.finish_count > 0
                and high_ko_summary.ko_tko_count
                == high_ko_summary.finish_count
                and high_ko_summary.submission_count == 0
            ),
            detail=(
                f"KO/TKO={high_ko_summary.ko_tko_count}, "
                f"submission="
                f"{high_ko_summary.submission_count}"
            ),
        ),
        AuditCheck(
            name="stronger KO calibration produces earlier finishes",
            passed=(
                high_mean is not None
                and low_mean is not None
                and high_mean < low_mean
            ),
            detail=(
                f"low mean segment={low_mean:.3f}, "
                f"high mean segment={high_mean:.3f}"
                if low_mean is not None and high_mean is not None
                else "one population produced no finishes"
            ),
        ),
        AuditCheck(
            name="KO finish paths preserve terminal legality",
            passed=(
                low_ko_summary.structural_violations == 0
                and high_ko_summary.structural_violations == 0
            ),
            detail=(
                f"low violations="
                f"{low_ko_summary.structural_violations}, "
                f"high violations="
                f"{high_ko_summary.structural_violations}"
            ),
        ),
        AuditCheck(
            name="submission sampling matches deterministic probability",
            passed=(
                abs(
                    observed_submission_probability
                    - expected_submission_probability
                )
                <= 0.03
            ),
            detail=(
                f"expected="
                f"{expected_submission_probability:.4f}, "
                f"observed="
                f"{observed_submission_probability:.4f}"
            ),
        ),
        AuditCheck(
            name="sampled submissions preserve ground-owner legality",
            passed=submission_violations == 0,
            detail=(
                f"violations={submission_violations}"
            ),
        ),
        AuditCheck(
            name="finish timing remains within segment bounds",
            passed=(
                timing_violations == 0
                and minimum_finish_second == 1
                and maximum_finish_second == 30
            ),
            detail=(
                f"minimum={minimum_finish_second}, "
                f"maximum={maximum_finish_second}, "
                f"violations={timing_violations}"
            ),
        ),
        AuditCheck(
            name="finish timing remains centered in segment",
            passed=14.5 <= mean_finish_second <= 16.5,
            detail=(
                f"mean finish second="
                f"{mean_finish_second:.4f}"
            ),
        ),
    ]

    print("=" * 80)
    print("RFS MONTE CARLO V2 FINISH PATH AUDIT")
    print("=" * 80)
    print(f"Paths per full-path scenario: {path_count:,}")
    print(f"Samples per segment scenario: {sampler_count:,}")
    print(f"Seed start:                   {seed_start:,}")
    print()

    print("KO/TKO POPULATION SUMMARY")
    print("-" * 80)
    print(
        f"Low calibration:  "
        f"{low_ko_summary.finish_rate:.2%} finishes, "
        f"mean segment="
        f"{low_ko_summary.mean_finishing_segment}"
    )
    print(
        f"High calibration: "
        f"{high_ko_summary.finish_rate:.2%} finishes, "
        f"mean segment="
        f"{high_ko_summary.mean_finishing_segment}"
    )
    print(
        "High-calibration finish rounds: "
        f"{dict(sorted(high_ko_summary.round_counts.items()))}"
    )
    print()

    all_passed = True

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        all_passed = all_passed and check.passed

        print(f"[{status}] {check.name}")
        print(f"       {check.detail}")

    print()
    print("=" * 80)
    print(
        "AUDIT PASS"
        if all_passed
        else "AUDIT FAIL"
    )
    print("=" * 80)

    return 0 if all_passed else 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit RFS Monte Carlo V2 finish-enabled paths."
        )
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=1_000,
        help="Number of paths per full-path scenario.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First deterministic seed.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the command-line audit."""

    args = parse_args()

    return run_audit(
        path_count=args.paths,
        seed_start=args.seed_start,
    )


if __name__ == "__main__":
    raise SystemExit(main())
