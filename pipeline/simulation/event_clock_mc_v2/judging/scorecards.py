"""Round-local statistics and independent three-judge scorecards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from .model import Event2JudgeModel, JudgeFeatures

SIGNIFICANT = frozenset({
    ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER, ActionFamily.CLINCH_STRIKE,
    ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE,
})
TAKEDOWNS = frozenset({ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN})


@dataclass(frozen=True)
class RoundCard:
    round_number: int
    red_score: int
    blue_score: int
    winner: Side
    p_red_round: float
    features: JudgeFeatures


@dataclass(frozen=True)
class JudgeScorecard:
    judge_number: int
    rounds: tuple[RoundCard, ...]
    red_total: int
    blue_total: int
    winner: Side


@dataclass(frozen=True)
class DecisionResult:
    scorecards: tuple[JudgeScorecard, ...]
    winner: Side
    classification: str
    round_probabilities: tuple[float, ...]


def round_features(events, timeline_segments, round_number: int, round_length: float) -> JudgeFeatures:
    start, end = (round_number - 1) * round_length, round_number * round_length
    stats = {side: dict(sig=0, kd=0, td=0, sub=0, ctrl=0.0) for side in Side}
    for event in events:
        if not start <= event.timestamp_seconds < end:
            continue
        row = stats[event.actor]
        if event.selected_action in SIGNIFICANT and event.outcome.value == "landed": row["sig"] += 1
        row["kd"] += int(event.knockdown)
        if event.selected_action in TAKEDOWNS and event.transition_kind is not None: row["td"] += 1
        if event.selected_action is ActionFamily.SUBMISSION_ATTACK: row["sub"] += 1
    for segment in timeline_segments:
        overlap = max(0.0, min(segment.end_time, end) - max(segment.start_time, start))
        if overlap and segment.controller is not None:
            stats[segment.controller]["ctrl"] += overlap
    red, blue = stats[Side.RED], stats[Side.BLUE]
    return JudgeFeatures(*(red[k] - blue[k] for k in ("sig", "kd", "td", "sub", "ctrl")))


def score_decision(events, timeline_segments, *, rounds: int, round_length: float,
                   model: Event2JudgeModel, rng: np.random.Generator) -> DecisionResult:
    features = tuple(round_features(events, timeline_segments, r, round_length) for r in range(1, rounds + 1))
    probabilities = tuple(model.probability(row) for row in features)
    cards = []
    for judge in range(1, 4):
        round_cards = []
        for number, (row, probability) in enumerate(zip(features, probabilities), 1):
            winner = Side.RED if rng.random() < probability else Side.BLUE
            round_cards.append(RoundCard(number, 10 if winner is Side.RED else 9,
                                         10 if winner is Side.BLUE else 9, winner, probability, row))
        red_total = sum(x.red_score for x in round_cards)
        blue_total = sum(x.blue_score for x in round_cards)
        winner = Side.RED if red_total > blue_total else Side.BLUE
        cards.append(JudgeScorecard(judge, tuple(round_cards), red_total, blue_total, winner))
    red_cards = sum(card.winner is Side.RED for card in cards)
    winner = Side.RED if red_cards >= 2 else Side.BLUE
    unanimous = red_cards in (0, 3)
    return DecisionResult(tuple(cards), winner, "unanimous_decision" if unanimous else "split_decision", probabilities)
