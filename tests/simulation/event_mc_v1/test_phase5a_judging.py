import numpy as np

from pipeline.simulation.event_mc_v1.components.actions import ActionAttempt, ActionOutcome
from pipeline.simulation.event_mc_v1.components.profiles import Side
from pipeline.simulation.event_mc_v1.events import ConsequenceEvent, PrimaryEvent
from pipeline.simulation.event_mc_v1.judging import DeterministicJudgingModel, RoundEvidence, RoundScore


def judge(evidence, seed=1):
    subject = DeterministicJudgingModel()
    subject.evidence = evidence
    return subject.score_round(1, np.random.default_rng(seed))


def evidence(striking=(0, 0), grappling=(0, 0), aggression=(0, 0), control=(0, 0)):
    item = RoundEvidence()
    for values, target in ((striking, item.striking), (grappling, item.grappling), (aggression, item.aggression), (control, item.control)):
        target["red"], target["blue"] = values
    return item


def test_primary_score_combines_striking_grappling_and_control():
    striking = judge(evidence(striking=(5, 1)))
    grappling = judge(evidence(grappling=(2, 0), striking=(0, 1)))
    control = judge(evidence(striking=(5, 1), control=(0, 300)))

    assert striking.winner is Side.RED and striking.criterion == "PRIMARY"
    assert grappling.winner is Side.RED and grappling.criterion == "PRIMARY"

    # 300 seconds of control is worth 300 * 0.048904 = 14.6712
    # calibrated significant-strike equivalents, enough to overcome 5-1.
    assert control.winner is Side.BLUE and control.criterion == "PRIMARY"


def test_aggression_is_only_used_when_calibrated_primary_score_is_exactly_tied():
    aggression = judge(
        evidence(
            striking=(1, 1),
            aggression=(5, 2),
            control=(0, 0),
        )
    )
    assert aggression.winner is Side.RED
    assert aggression.criterion == "AGGRESSION"


def test_exact_tie_is_reproducible_and_always_ten_nine():
    first = judge(evidence(), 42)
    second = judge(evidence(), 42)
    assert first == second
    assert {first.red_score, first.blue_score} == {9, 10}
    assert first.criterion == "FINAL_TIEBREAKER"


def card(round_number, winner):
    return RoundScore(round_number, winner, 0, 0, 0, 0, 0, "PRIMARY", 10 if winner is Side.RED else 9, 10 if winner is Side.BLUE else 9)


def test_three_and_five_round_majorities_produce_decision_without_draw():
    three = DeterministicJudgingModel(); three.cards = [card(1, Side.RED), card(2, Side.BLUE), card(3, Side.RED)]
    five = DeterministicJudgingModel(); five.cards = [card(1, Side.BLUE), card(2, Side.RED), card(3, Side.BLUE), card(4, Side.RED), card(5, Side.BLUE)]
    assert three.decision_delta().winner == "red" and three.decision_delta().finish_method == "DEC"
    assert five.decision_delta().winner == "blue" and five.decision_delta().finish_method == "DEC"


def test_aggression_counts_only_offensive_initiative_and_reversal_keeps_grappling():
    subject = DeterministicJudgingModel()
    snapshots = type("Snapshot", (), {})()
    offensive = (
        "strike", "takedown", "clinch_entry", "clinch_strike",
        "clinch_takedown", "ground_strike", "submission_attempt",
    )
    excluded = ("ground_escape", "clinch_separation", "ground_reversal")
    for family in offensive + excluded:
        subject.on_event(
            PrimaryEvent(1.0, f"red_{family}", ActionAttempt(Side.RED, family)),
            snapshots,
            snapshots,
        )
    assert subject.evidence.aggression == {"red": 7.0, "blue": 0.0}

    subject.on_event(
        ConsequenceEvent(
            1.0,
            "ActionOutcome",
            ActionOutcome(Side.RED, "ground_reversal", "reversed"),
        ),
        snapshots,
        snapshots,
    )
    assert subject.evidence.aggression["red"] == 7.0
    assert subject.evidence.grappling["red"] == 2.854417
