"""Shadow-only activity runner for RFS Monte Carlo V1.

This module currently runs activity-only simulation paths.

Implemented:
- ten 30-second segments per round
- three- or five-round schedules
- explicit seeded NumPy generators
- deterministic path reproduction
- red and blue fight-level activity totals

Not implemented:
- dynamic fatigue or damage
- KO/TKO or submission finishes
- scoring or decision outcomes
- production artifact writes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from pipeline.simulation.rfs_mc_v1.contracts import (
    MatchupSimulationRequest,
)
from pipeline.simulation.rfs_mc_v1.segment_engine import (
    SEGMENTS_PER_ROUND,
    SegmentMatchupActivity,
    aggregate_segment_activity,
    generate_matchup_segment,
)


@dataclass(frozen=True)
class ActivityPathResult:
    """Activity-only output from one simulated fight path."""

    path_index: int
    seed: int
    scheduled_rounds: int

    segments: tuple[SegmentMatchupActivity, ...]

    red_totals: Mapping[str, int]
    blue_totals: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.path_index < 0:
            raise ValueError("path_index cannot be negative")

        if self.scheduled_rounds not in {3, 5}:
            raise ValueError("scheduled_rounds must be 3 or 5")

        expected_segments = (
            self.scheduled_rounds * SEGMENTS_PER_ROUND
        )

        if len(self.segments) != expected_segments:
            raise ValueError(
                f"Expected {expected_segments} segments, "
                f"received {len(self.segments)}"
            )


@dataclass(frozen=True)
class ActivitySimulationResult:
    """Collection of activity-only Monte Carlo paths."""

    simulator_version: str
    calibration_version: str
    seed: int
    path_count: int

    red_fighter_id: str
    blue_fighter_id: str
    scheduled_rounds: int

    paths: tuple[ActivityPathResult, ...]

    def __post_init__(self) -> None:
        if self.path_count <= 0:
            raise ValueError("path_count must be positive")

        if len(self.paths) != self.path_count:
            raise ValueError(
                "paths length must equal path_count"
            )


def _aggregate_matchup_side(
    segments: list[SegmentMatchupActivity],
    *,
    side: str,
) -> Mapping[str, int]:
    """Aggregate red or blue activity across a complete path."""

    if side not in {"red", "blue"}:
        raise ValueError("side must be 'red' or 'blue'")

    fighter_segments = [
        getattr(segment, side)
        for segment in segments
    ]

    return aggregate_segment_activity(fighter_segments)


def simulate_activity_path(
    request: MatchupSimulationRequest,
    *,
    path_index: int,
    seed: int,
) -> ActivityPathResult:
    """Simulate one complete activity-only fight path."""

    if path_index < 0:
        raise ValueError("path_index cannot be negative")

    rng = np.random.default_rng(seed)

    scheduled_rounds = (
        request.red_profile.scheduled_rounds
    )

    segments: list[SegmentMatchupActivity] = []

    for round_number in range(1, scheduled_rounds + 1):
        for segment_number in range(
            1,
            SEGMENTS_PER_ROUND + 1,
        ):
            segments.append(
                generate_matchup_segment(
                    red_profile=request.red_profile,
                    blue_profile=request.blue_profile,
                    round_number=round_number,
                    segment_number=segment_number,
                    rng=rng,
                )
            )

    return ActivityPathResult(
        path_index=path_index,
        seed=seed,
        scheduled_rounds=scheduled_rounds,
        segments=tuple(segments),
        red_totals=_aggregate_matchup_side(
            segments,
            side="red",
        ),
        blue_totals=_aggregate_matchup_side(
            segments,
            side="blue",
        ),
    )


def simulate_activity_paths(
    request: MatchupSimulationRequest,
) -> ActivitySimulationResult:
    """Run deterministic activity-only Monte Carlo paths.

    A root seed sequence spawns one independent child seed per path.
    Identical requests reproduce identical path results.
    """

    root_seed = np.random.SeedSequence(request.seed)
    child_sequences = root_seed.spawn(request.path_count)

    paths: list[ActivityPathResult] = []

    for path_index, child_sequence in enumerate(
        child_sequences
    ):
        child_seed = int(
            child_sequence.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )

        paths.append(
            simulate_activity_path(
                request,
                path_index=path_index,
                seed=child_seed,
            )
        )

    return ActivitySimulationResult(
        simulator_version=request.simulator_version,
        calibration_version=request.calibration_version,
        seed=request.seed,
        path_count=request.path_count,
        red_fighter_id=request.red_profile.fighter_id,
        blue_fighter_id=request.blue_profile.fighter_id,
        scheduled_rounds=(
            request.red_profile.scheduled_rounds
        ),
        paths=tuple(paths),
    )


def summarize_activity_simulation(
    result: ActivitySimulationResult,
) -> dict[str, object]:
    """Summarize activity distributions across simulation paths."""

    metrics = (
        "sig_str_attempted",
        "sig_str_landed",
        "td_attempted",
        "td_landed",
        "control_seconds",
        "ground_str_attempted",
        "ground_str_landed",
        "submission_attempts",
        "knockdowns",
    )

    summary: dict[str, object] = {
        "simulator_version": result.simulator_version,
        "calibration_version": result.calibration_version,
        "seed": result.seed,
        "path_count": result.path_count,
        "red_fighter_id": result.red_fighter_id,
        "blue_fighter_id": result.blue_fighter_id,
        "scheduled_rounds": result.scheduled_rounds,
        "red": {},
        "blue": {},
    }

    for side in ("red", "blue"):
        side_summary: dict[str, dict[str, float]] = {}

        for metric in metrics:
            values = np.array(
                [
                    getattr(path, f"{side}_totals")[metric]
                    for path in result.paths
                ],
                dtype=float,
            )

            side_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
            }

        summary[side] = side_summary

    return summary


@dataclass(frozen=True)
class StatefulSegmentTrace:
    """One segment plus fighter states immediately after that segment."""

    activity: SegmentMatchupActivity
    red_state: object
    blue_state: object


@dataclass(frozen=True)
class StatefulActivityPathResult:
    """One complete activity path with evolving fighter states."""

    path_index: int
    seed: int
    scheduled_rounds: int

    traces: tuple[StatefulSegmentTrace, ...]

    red_totals: Mapping[str, int]
    blue_totals: Mapping[str, int]

    final_red_state: object
    final_blue_state: object

    def __post_init__(self) -> None:
        if self.path_index < 0:
            raise ValueError("path_index cannot be negative")

        expected_segments = (
            self.scheduled_rounds * SEGMENTS_PER_ROUND
        )

        if len(self.traces) != expected_segments:
            raise ValueError(
                f"Expected {expected_segments} traces, "
                f"received {len(self.traces)}"
            )


def simulate_stateful_activity_path(
    request: MatchupSimulationRequest,
    *,
    path_index: int,
    seed: int,
) -> StatefulActivityPathResult:
    """Simulate one full path while evolving independent fighter states."""

    from copy import deepcopy

    from pipeline.simulation.rfs_mc_v1.dynamic_state import (
        apply_between_round_recovery,
        initialize_dynamic_state,
        update_dynamic_state,
    )

    if path_index < 0:
        raise ValueError("path_index cannot be negative")

    rng = np.random.default_rng(seed)

    red_state = initialize_dynamic_state(request.red_profile)
    blue_state = initialize_dynamic_state(request.blue_profile)

    activities: list[SegmentMatchupActivity] = []
    traces: list[StatefulSegmentTrace] = []

    scheduled_rounds = request.red_profile.scheduled_rounds

    for round_number in range(1, scheduled_rounds + 1):
        for segment_number in range(
            1,
            SEGMENTS_PER_ROUND + 1,
        ):
            activity = generate_matchup_segment(
                red_profile=request.red_profile,
                blue_profile=request.blue_profile,
                round_number=round_number,
                segment_number=segment_number,
                rng=rng,
                red_energy=red_state.energy,
                blue_energy=blue_state.energy,
            )

            red_state = update_dynamic_state(
                state=red_state,
                own_activity=activity.red,
                opponent_activity=activity.blue,
                profile=request.red_profile,
            )

            blue_state = update_dynamic_state(
                state=blue_state,
                own_activity=activity.blue,
                opponent_activity=activity.red,
                profile=request.blue_profile,
            )

            activities.append(activity)

            traces.append(
                StatefulSegmentTrace(
                    activity=activity,
                    red_state=deepcopy(red_state),
                    blue_state=deepcopy(blue_state),
                )
            )

        if round_number < scheduled_rounds:
            red_state = apply_between_round_recovery(red_state)
            blue_state = apply_between_round_recovery(blue_state)

    return StatefulActivityPathResult(
        path_index=path_index,
        seed=seed,
        scheduled_rounds=scheduled_rounds,
        traces=tuple(traces),
        red_totals=_aggregate_matchup_side(
            activities,
            side="red",
        ),
        blue_totals=_aggregate_matchup_side(
            activities,
            side="blue",
        ),
        final_red_state=deepcopy(red_state),
        final_blue_state=deepcopy(blue_state),
    )


def simulate_stateful_activity_paths(
    request: MatchupSimulationRequest,
) -> tuple[StatefulActivityPathResult, ...]:
    """Run reproducible stateful activity paths."""

    root_seed = np.random.SeedSequence(request.seed)
    child_sequences = root_seed.spawn(request.path_count)

    paths: list[StatefulActivityPathResult] = []

    for path_index, child_sequence in enumerate(
        child_sequences
    ):
        child_seed = int(
            child_sequence.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )

        paths.append(
            simulate_stateful_activity_path(
                request,
                path_index=path_index,
                seed=child_seed,
            )
        )

    return tuple(paths)


@dataclass(frozen=True)
class FightPathOutcome:
    """Terminal outcome for one simulated fight path."""

    winner: str | None
    loser: str | None
    method: str
    finish_round: int | None
    finish_segment: int | None
    elapsed_seconds: int

    def __post_init__(self) -> None:
        if self.method not in {
            "ko_tko",
            "submission",
            "decision",
        }:
            raise ValueError(f"Unsupported outcome method: {self.method}")

        if self.method == "decision":
            if self.finish_round is not None:
                raise ValueError(
                    "Decision outcome cannot have finish_round"
                )
            if self.finish_segment is not None:
                raise ValueError(
                    "Decision outcome cannot have finish_segment"
                )
        else:
            if self.winner not in {"red", "blue"}:
                raise ValueError(
                    "Finish outcome requires a winner"
                )
            if self.loser not in {"red", "blue"}:
                raise ValueError(
                    "Finish outcome requires a loser"
                )
            if self.winner == self.loser:
                raise ValueError("winner and loser must differ")
            if self.finish_round is None:
                raise ValueError(
                    "Finish outcome requires finish_round"
                )
            if self.finish_segment is None:
                raise ValueError(
                    "Finish outcome requires finish_segment"
                )

        if self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")


@dataclass(frozen=True)
class FinishAwareSegmentTrace:
    """Activity, state, and finish evaluation for one segment."""

    activity: SegmentMatchupActivity
    red_state: object
    blue_state: object
    finish_result: object


@dataclass(frozen=True)
class FinishAwarePathResult:
    """Complete fight path that may terminate before the scheduled distance."""

    path_index: int
    seed: int
    scheduled_rounds: int

    traces: tuple[FinishAwareSegmentTrace, ...]
    outcome: FightPathOutcome

    red_totals: Mapping[str, int]
    blue_totals: Mapping[str, int]

    final_red_state: object
    final_blue_state: object

    def __post_init__(self) -> None:
        if self.path_index < 0:
            raise ValueError("path_index cannot be negative")

        maximum_segments = (
            self.scheduled_rounds * SEGMENTS_PER_ROUND
        )

        if not 1 <= len(self.traces) <= maximum_segments:
            raise ValueError(
                "Finish-aware path must contain between one and "
                f"{maximum_segments} traces"
            )

        if self.outcome.method == "decision":
            if len(self.traces) != maximum_segments:
                raise ValueError(
                    "Decision path must complete all scheduled segments"
                )
        elif len(self.traces) >= maximum_segments:
            # A finish may occur in the final scheduled segment.
            expected_final_round = self.scheduled_rounds
            expected_final_segment = SEGMENTS_PER_ROUND

            if (
                self.outcome.finish_round != expected_final_round
                or self.outcome.finish_segment
                != expected_final_segment
            ):
                raise ValueError(
                    "Full-length finish must occur in the final segment"
                )


def simulate_finish_aware_path(
    request: MatchupSimulationRequest,
    *,
    path_index: int,
    seed: int,
) -> FinishAwarePathResult:
    """Simulate one stateful fight path with competing finish hazards."""

    from copy import deepcopy

    from pipeline.simulation.rfs_mc_v1.dynamic_state import (
        apply_between_round_recovery,
        initialize_dynamic_state,
        update_dynamic_state,
    )
    from pipeline.simulation.rfs_mc_v1.finish_engine import (
        sample_competing_finish,
    )

    if path_index < 0:
        raise ValueError("path_index cannot be negative")

    rng = np.random.default_rng(seed)

    red_state = initialize_dynamic_state(request.red_profile)
    blue_state = initialize_dynamic_state(request.blue_profile)

    activities: list[SegmentMatchupActivity] = []
    traces: list[FinishAwareSegmentTrace] = []

    scheduled_rounds = request.red_profile.scheduled_rounds
    outcome: FightPathOutcome | None = None

    for round_number in range(1, scheduled_rounds + 1):
        for segment_number in range(
            1,
            SEGMENTS_PER_ROUND + 1,
        ):
            activity = generate_matchup_segment(
                red_profile=request.red_profile,
                blue_profile=request.blue_profile,
                round_number=round_number,
                segment_number=segment_number,
                rng=rng,
                red_energy=red_state.energy,
                blue_energy=blue_state.energy,
            )

            red_state = update_dynamic_state(
                state=red_state,
                own_activity=activity.red,
                opponent_activity=activity.blue,
                profile=request.red_profile,
            )
            blue_state = update_dynamic_state(
                state=blue_state,
                own_activity=activity.blue,
                opponent_activity=activity.red,
                profile=request.blue_profile,
            )

            finish_result = sample_competing_finish(
                red_state=red_state,
                blue_state=blue_state,
                red_activity=activity.red,
                blue_activity=activity.blue,
                red_profile=request.red_profile,
                blue_profile=request.blue_profile,
                rng=rng,
            )

            activities.append(activity)
            traces.append(
                FinishAwareSegmentTrace(
                    activity=activity,
                    red_state=deepcopy(red_state),
                    blue_state=deepcopy(blue_state),
                    finish_result=finish_result,
                )
            )

            if finish_result.finished:
                elapsed_seconds = (
                    (round_number - 1)
                    * SEGMENTS_PER_ROUND
                    * 30
                    + segment_number * 30
                )

                outcome = FightPathOutcome(
                    winner=finish_result.winner,
                    loser=finish_result.loser,
                    method=finish_result.method.value,
                    finish_round=round_number,
                    finish_segment=segment_number,
                    elapsed_seconds=elapsed_seconds,
                )
                break

        if outcome is not None:
            break

        if round_number < scheduled_rounds:
            red_state = apply_between_round_recovery(red_state)
            blue_state = apply_between_round_recovery(blue_state)

    if outcome is None:
        outcome = FightPathOutcome(
            winner=None,
            loser=None,
            method="decision",
            finish_round=None,
            finish_segment=None,
            elapsed_seconds=(
                scheduled_rounds
                * SEGMENTS_PER_ROUND
                * 30
            ),
        )

    return FinishAwarePathResult(
        path_index=path_index,
        seed=seed,
        scheduled_rounds=scheduled_rounds,
        traces=tuple(traces),
        outcome=outcome,
        red_totals=_aggregate_matchup_side(
            activities,
            side="red",
        ),
        blue_totals=_aggregate_matchup_side(
            activities,
            side="blue",
        ),
        final_red_state=deepcopy(red_state),
        final_blue_state=deepcopy(blue_state),
    )


def simulate_finish_aware_paths(
    request: MatchupSimulationRequest,
) -> tuple[FinishAwarePathResult, ...]:
    """Run deterministic finish-aware Monte Carlo fight paths."""

    root_seed = np.random.SeedSequence(request.seed)
    child_sequences = root_seed.spawn(request.path_count)

    paths: list[FinishAwarePathResult] = []

    for path_index, child_sequence in enumerate(
        child_sequences
    ):
        child_seed = int(
            child_sequence.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )

        paths.append(
            simulate_finish_aware_path(
                request,
                path_index=path_index,
                seed=child_seed,
            )
        )

    return tuple(paths)


@dataclass(frozen=True)
class ScoredFightPathResult:
    """Finish-aware path with an optional completed scorecard."""

    path: FinishAwarePathResult
    decision: object | None

    def __post_init__(self) -> None:
        if self.path.outcome.method == "decision":
            if self.decision is None:
                raise ValueError(
                    "Decision path requires a scorecard"
                )
        elif self.decision is not None:
            raise ValueError(
                "Finished path cannot have a decision scorecard"
            )


def score_finish_aware_path(
    path: FinishAwarePathResult,
) -> ScoredFightPathResult:
    """Attach a decision scorecard when a path reaches distance."""

    if path.outcome.method != "decision":
        return ScoredFightPathResult(
            path=path,
            decision=None,
        )

    from pipeline.simulation.rfs_mc_v1.scoring import (
        score_decision,
    )

    decision = score_decision(
        (
            trace.activity
            for trace in path.traces
        ),
        scheduled_rounds=path.scheduled_rounds,
    )

    scored_outcome = FightPathOutcome(
        winner=decision.winner,
        loser=decision.loser,
        method="decision",
        finish_round=None,
        finish_segment=None,
        elapsed_seconds=path.outcome.elapsed_seconds,
    )

    scored_path = FinishAwarePathResult(
        path_index=path.path_index,
        seed=path.seed,
        scheduled_rounds=path.scheduled_rounds,
        traces=path.traces,
        outcome=scored_outcome,
        red_totals=path.red_totals,
        blue_totals=path.blue_totals,
        final_red_state=path.final_red_state,
        final_blue_state=path.final_blue_state,
    )

    return ScoredFightPathResult(
        path=scored_path,
        decision=decision,
    )


def simulate_scored_paths(
    request: MatchupSimulationRequest,
) -> tuple[ScoredFightPathResult, ...]:
    """Run finish-aware paths and score all distance outcomes."""

    return tuple(
        score_finish_aware_path(path)
        for path in simulate_finish_aware_paths(request)
    )


def summarize_scored_paths(
    results: tuple[ScoredFightPathResult, ...],
) -> dict[str, object]:
    """Aggregate scored paths into matchup-level simulation probabilities."""

    if not results:
        raise ValueError("At least one scored path is required")

    path_count = len(results)

    outcome_counts = {
        "red": 0,
        "blue": 0,
        "draw": 0,
    }
    method_counts = {
        "red_ko_tko": 0,
        "red_submission": 0,
        "red_decision": 0,
        "blue_ko_tko": 0,
        "blue_submission": 0,
        "blue_decision": 0,
        "draw": 0,
    }

    finish_rounds: list[int] = []
    finish_seconds: list[int] = []

    red_score_totals: list[int] = []
    blue_score_totals: list[int] = []

    # Temporary scoring diagnostics used to identify why simulated
    # scorecards end in draws.
    draw_scorecard_counts: dict[str, int] = {}
    draw_round_pattern_counts: dict[str, int] = {}
    draws_with_ten_eight = 0
    draws_with_ten_ten = 0

    for result in results:
        outcome = result.path.outcome

        if outcome.winner == "red":
            outcome_counts["red"] += 1
        elif outcome.winner == "blue":
            outcome_counts["blue"] += 1
        else:
            outcome_counts["draw"] += 1

        if outcome.method == "decision":
            if outcome.winner == "red":
                method_counts["red_decision"] += 1
            elif outcome.winner == "blue":
                method_counts["blue_decision"] += 1
            else:
                method_counts["draw"] += 1

            if result.decision is not None:
                decision = result.decision

                red_score_totals.append(decision.red_total)
                blue_score_totals.append(decision.blue_total)

                if outcome.winner is None:
                    scorecard_key = (
                        f"{decision.red_total}-{decision.blue_total}"
                    )
                    draw_scorecard_counts[scorecard_key] = (
                        draw_scorecard_counts.get(scorecard_key, 0) + 1
                    )

                    round_pattern = "/".join(
                        f"{round_score.red_points}-"
                        f"{round_score.blue_points}"
                        for round_score in decision.round_scores
                    )
                    draw_round_pattern_counts[round_pattern] = (
                        draw_round_pattern_counts.get(
                            round_pattern,
                            0,
                        )
                        + 1
                    )

                    if any(
                        8
                        in {
                            round_score.red_points,
                            round_score.blue_points,
                        }
                        for round_score in decision.round_scores
                    ):
                        draws_with_ten_eight += 1

                    if any(
                        round_score.red_points == 10
                        and round_score.blue_points == 10
                        for round_score in decision.round_scores
                    ):
                        draws_with_ten_ten += 1
        else:
            method_key = f"{outcome.winner}_{outcome.method}"
            method_counts[method_key] += 1

            if outcome.finish_round is not None:
                finish_rounds.append(outcome.finish_round)

            finish_seconds.append(outcome.elapsed_seconds)

    def probability(count: int) -> float:
        return float(count / path_count)

    summary: dict[str, object] = {
        "path_count": path_count,
        "red_win_probability": probability(
            outcome_counts["red"]
        ),
        "blue_win_probability": probability(
            outcome_counts["blue"]
        ),
        "draw_probability": probability(
            outcome_counts["draw"]
        ),
        "method_probabilities": {
            key: probability(value)
            for key, value in method_counts.items()
        },
        "distance_probability": probability(
            sum(
                outcome.method == "decision"
                for outcome in (
                    result.path.outcome
                    for result in results
                )
            )
        ),
        "finish_probability": probability(
            sum(
                outcome.method != "decision"
                for outcome in (
                    result.path.outcome
                    for result in results
                )
            )
        ),
    }

    summary["draw_diagnostics"] = {
        "draw_count": outcome_counts["draw"],
        "draws_with_ten_eight": draws_with_ten_eight,
        "draws_with_ten_ten": draws_with_ten_ten,
        "scorecard_counts": dict(
            sorted(draw_scorecard_counts.items())
        ),
        "round_pattern_counts": dict(
            sorted(
                draw_round_pattern_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    }

    if finish_rounds:
        finish_round_values = np.array(
            finish_rounds,
            dtype=float,
        )
        finish_time_values = np.array(
            finish_seconds,
            dtype=float,
        )

        summary["finish_distribution"] = {
            "mean_round": float(
                finish_round_values.mean()
            ),
            "median_round": float(
                np.median(finish_round_values)
            ),
            "mean_elapsed_seconds": float(
                finish_time_values.mean()
            ),
            "median_elapsed_seconds": float(
                np.median(finish_time_values)
            ),
        }
    else:
        summary["finish_distribution"] = None

    if red_score_totals:
        summary["decision_score_distribution"] = {
            "mean_red_score": float(
                np.mean(red_score_totals)
            ),
            "mean_blue_score": float(
                np.mean(blue_score_totals)
            ),
        }
    else:
        summary["decision_score_distribution"] = None

    return summary
