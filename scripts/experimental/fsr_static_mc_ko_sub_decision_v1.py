"""Shadow full-fight KO/TKO + SUB + decision Monte Carlo.

This module preserves the current KO/damage/stamina/phase engine, applies the
selected 34% neutral submission-conversion candidate, and adds a scheduled-
distance decision layer derived from the previous RFS MC V1 scorer.

Decision scoring hierarchy:
1. effective offense: landed significant strikes, ground strikes, knockdowns,
   takedowns, submission attempts;
2. aggression: strike attempts and takedown attempts;
3. control as a secondary separator.

Every completed round is scored 10-9. Fight draws are disabled. Exact round ties
are resolved with a seeded unbiased coin flip after all deterministic scoring
separators are exhausted.

Research-only. Judge-specific disagreement is intentionally deferred until the
real historical judge-scorecard dataset is wired in for calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_ko_sub_v1 as combined
from scripts.experimental import fsr_static_mc_v0 as base


CALIBRATED_SUBMISSION_NEUTRAL_RATE = 0.34

SIG_LANDED_WEIGHT = 1.00
GROUND_LANDED_BONUS_WEIGHT = 0.20
KNOCKDOWN_WEIGHT = 6.00
TD_LANDED_WEIGHT = 1.50
SUB_ATTEMPT_WEIGHT = 2.25
SIG_ATTEMPT_WEIGHT = 0.025
TD_ATTEMPT_WEIGHT = 0.10
CONTROL_SECOND_WEIGHT = 0.015


def configure_current_full_fight_candidate() -> None:
    """Configure the current KO locks and selected submission neutral rate."""
    combined.configure_current_finish_candidate()
    combined.SUBMISSION_NEUTRAL_FINISH_PROBABILITY_PER_ATTEMPT = (
        CALIBRATED_SUBMISSION_NEUTRAL_RATE
    )


@dataclass(frozen=True)
class FighterRoundMetrics:
    sig_att: int
    sig_landed: int
    ground_sig_landed: int
    knockdowns: int
    td_att: int
    td_landed: int
    sub_att: int
    control_seconds: int
    effective_score: float
    aggression_score: float
    control_score: float
    total_score: float


@dataclass(frozen=True)
class RoundScore:
    round_number: int
    winner: int
    loser: int
    red_points: int
    blue_points: int
    red: FighterRoundMetrics
    blue: FighterRoundMetrics
    margin: float
    tie_break_used: bool


@dataclass(frozen=True)
class DecisionResult:
    winner: int
    loser: int
    method: str
    red_total: int
    blue_total: int
    red_rounds_won: int
    blue_rounds_won: int
    round_scores: tuple[RoundScore, ...]


@dataclass
class FullFightPath:
    events: list[dict[str, Any]]
    stats: list[Any]
    finish: Any | None
    decision: DecisionResult | None

    @property
    def winner(self) -> int:
        if self.finish is not None:
            return int(self.finish.winner)
        if self.decision is None:
            raise RuntimeError("full fight path has neither finish nor decision")
        return int(self.decision.winner)

    @property
    def method(self) -> str:
        if self.finish is not None:
            return str(self.finish.method)
        if self.decision is None:
            raise RuntimeError("full fight path has neither finish nor decision")
        return str(self.decision.method)


class StaticFSRMCFullFightV1(combined.StaticFSRMCKOSUBV1):
    """Current finish engine plus deterministic no-draw scheduled-distance scoring."""

    def __init__(self, *args, **kwargs) -> None:
        configure_current_full_fight_candidate()
        super().__init__(*args, **kwargs)
        if self.rounds not in {3, 5}:
            raise ValueError("decision layer supports scheduled 3- or 5-round fights")
        self._round_metrics: dict[int, list[dict[str, int]]] = {
            r: [self._empty_round_counts(), self._empty_round_counts()]
            for r in range(1, self.rounds + 1)
        }
        self.decision: DecisionResult | None = None

    @staticmethod
    def _empty_round_counts() -> dict[str, int]:
        return {
            "sig_att": 0,
            "sig_landed": 0,
            "ground_sig_landed": 0,
            "knockdowns": 0,
            "td_att": 0,
            "td_landed": 0,
            "sub_att": 0,
            "control_seconds": 0,
        }

    @staticmethod
    def _stats_snapshot(stats: Any) -> dict[str, int]:
        return {
            "sig_att": int(stats.sig_att),
            "sig_landed": int(stats.sig_landed),
            "knockdowns": int(getattr(stats, "knockdowns_scored", 0)),
            "td_att": int(stats.td_att),
            "td_landed": int(stats.td_landed),
            "sub_att": int(stats.sub_att),
            "control_seconds": int(stats.control_seconds),
        }

    def _capture_segment_delta(
        self,
        round_no: int,
        phase_start: str,
        before: list[dict[str, int]],
    ) -> None:
        for fighter in (0, 1):
            after = self._stats_snapshot(self.stats[fighter])
            target = self._round_metrics[round_no][fighter]
            for key in (
                "sig_att",
                "sig_landed",
                "knockdowns",
                "td_att",
                "td_landed",
                "sub_att",
                "control_seconds",
            ):
                delta = max(0, after[key] - before[fighter][key])
                target[key] += delta
                if key == "sig_landed" and phase_start == "GROUND":
                    target["ground_sig_landed"] += delta

    @staticmethod
    def _metric_score(counts: dict[str, int]) -> FighterRoundMetrics:
        effective = (
            counts["sig_landed"] * SIG_LANDED_WEIGHT
            + counts["ground_sig_landed"] * GROUND_LANDED_BONUS_WEIGHT
            + counts["knockdowns"] * KNOCKDOWN_WEIGHT
            + counts["td_landed"] * TD_LANDED_WEIGHT
            + counts["sub_att"] * SUB_ATTEMPT_WEIGHT
        )
        aggression = (
            counts["sig_att"] * SIG_ATTEMPT_WEIGHT
            + counts["td_att"] * TD_ATTEMPT_WEIGHT
        )
        control = counts["control_seconds"] * CONTROL_SECOND_WEIGHT
        total = effective + aggression + control
        return FighterRoundMetrics(
            sig_att=counts["sig_att"],
            sig_landed=counts["sig_landed"],
            ground_sig_landed=counts["ground_sig_landed"],
            knockdowns=counts["knockdowns"],
            td_att=counts["td_att"],
            td_landed=counts["td_landed"],
            sub_att=counts["sub_att"],
            control_seconds=counts["control_seconds"],
            effective_score=float(effective),
            aggression_score=float(aggression),
            control_score=float(control),
            total_score=float(total),
        )

    def _tie_break_round(self, red: FighterRoundMetrics, blue: FighterRoundMetrics) -> int:
        """Resolve an exact arithmetic tie without red/blue corner bias."""
        separators = (
            (red.knockdowns, blue.knockdowns),
            (red.sub_att, blue.sub_att),
            (red.sig_landed, blue.sig_landed),
            (red.ground_sig_landed, blue.ground_sig_landed),
            (red.td_landed, blue.td_landed),
            (red.control_seconds, blue.control_seconds),
            (red.sig_att, blue.sig_att),
            (red.td_att, blue.td_att),
        )
        for rv, bv in separators:
            if rv > bv:
                return 0
            if bv > rv:
                return 1
        return int(self.rng.integers(0, 2))

    def _score_round(self, round_no: int) -> RoundScore:
        red = self._metric_score(self._round_metrics[round_no][0])
        blue = self._metric_score(self._round_metrics[round_no][1])
        margin = red.total_score - blue.total_score
        tie_break_used = False
        if margin > 1e-12:
            winner = 0
        elif margin < -1e-12:
            winner = 1
        else:
            winner = self._tie_break_round(red, blue)
            tie_break_used = True
        loser = 1 - winner

        # Decision-layer lock: every completed round is 10-9. No 10-8 or 10-10
        # scores are emitted in this shadow candidate.
        if winner == 0:
            red_points, blue_points = 10, 9
        else:
            red_points, blue_points = 9, 10

        return RoundScore(
            round_number=round_no,
            winner=winner,
            loser=loser,
            red_points=red_points,
            blue_points=blue_points,
            red=red,
            blue=blue,
            margin=float(margin),
            tie_break_used=tie_break_used,
        )

    def _score_decision(self) -> DecisionResult:
        round_scores = tuple(self._score_round(r) for r in range(1, self.rounds + 1))
        red_rounds = sum(score.winner == 0 for score in round_scores)
        blue_rounds = sum(score.winner == 1 for score in round_scores)
        if red_rounds > blue_rounds:
            winner = 0
        elif blue_rounds > red_rounds:
            winner = 1
        else:
            winner = int(self.rng.integers(0, 2))
        return DecisionResult(
            winner=winner,
            loser=1 - winner,
            method="DEC",
            red_total=sum(score.red_points for score in round_scores),
            blue_total=sum(score.blue_points for score in round_scores),
            red_rounds_won=red_rounds,
            blue_rounds_won=blue_rounds,
            round_scores=round_scores,
        )

    def run(self) -> FullFightPath:
        events: list[dict[str, Any]] = []
        for round_no in range(1, self.rounds + 1):
            self.phase = "DISTANCE"
            self.ground_controller = None
            self.clinch_controller = None
            self.clinch_initiator = None

            for segment_no in range(1, base.SEGMENTS_PER_ROUND + 1):
                self._refresh_effective_fighters(round_no, segment_no)
                self.pending_stamina_costs = [[], []]

                phase_start = self.phase
                ground_controller_start = self.ground_controller
                clinch_controller_start = self.clinch_controller
                before = [self._stats_snapshot(self.stats[0]), self._stats_snapshot(self.stats[1])]

                for stats in self.stats:
                    stats.phase_segments[phase_start] += 1

                strike_notes = self._generate_striking(phase_start)
                if self.finish is not None:
                    transition_note = (
                        f"fight stopped: {self.names[self.finish.winner]} "
                        f"{self.finish.method} {self.names[self.finish.loser]}"
                    )
                elif phase_start == "DISTANCE":
                    transition_note = self._distance_transition()
                elif phase_start == "CLINCH":
                    transition_note = self._clinch_transition()
                else:
                    transition_note = self._ground_transition()

                if self.finish is not None and self.finish.round is None:
                    self.finish.round = round_no
                    self.finish.segment = segment_no
                    self.finish.clock_start = self._clock_start(segment_no)

                self._capture_segment_delta(round_no, phase_start, before)
                self._flush_pending_stamina_costs()

                events.append({
                    "round": round_no,
                    "segment": segment_no,
                    "clock_start": self._clock_start(segment_no),
                    "phase_start": phase_start,
                    "phase_end": self.phase,
                    "top_start": self.names[ground_controller_start] if ground_controller_start is not None else None,
                    "top_end": self.names[self.ground_controller] if self.ground_controller is not None else None,
                    "clinch_controller_start": self.names[clinch_controller_start] if clinch_controller_start is not None else None,
                    "clinch_controller_end": self.names[self.clinch_controller] if self.clinch_controller is not None else None,
                    "striking": "; ".join(strike_notes) if strike_notes else "no sig attempts",
                    "transition": transition_note,
                    "finish": self.finish is not None,
                    "finish_method": self.finish.method if self.finish is not None else "",
                    "red_stamina_after": self.stamina_state[0].fraction,
                    "blue_stamina_after": self.stamina_state[1].fraction,
                })

                if self.finish is not None:
                    return FullFightPath(events=events, stats=self.stats, finish=self.finish, decision=None)

            if round_no < self.rounds:
                self._apply_between_round_recovery(round_no)

        self.decision = self._score_decision()
        return FullFightPath(events=events, stats=self.stats, finish=None, decision=self.decision)
