import numpy as np

from pipeline.simulation.event_mc_v1.components.profiles import Side
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


def test_primary_striking_and_grappling_win_without_control_points():
    striking = judge(evidence(striking=(5, 1), control=(0, 300)))
    grappling = judge(evidence(grappling=(2, 0), striking=(0, 1)))
    assert striking.winner is Side.RED and striking.criterion == "PRIMARY"
    assert grappling.winner is Side.RED and grappling.criterion == "PRIMARY"


def test_hierarchy_uses_aggression_then_control_only_for_close_primary():
    aggression = judge(evidence(striking=(1, 1.1), aggression=(5, 2), control=(0, 100)))
    control = judge(evidence(striking=(1, 1.1), aggression=(2, 2), control=(90, 20)))
    assert aggression.winner is Side.RED and aggression.criterion == "AGGRESSION"
    assert control.winner is Side.RED and control.criterion == "CONTROL"


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
