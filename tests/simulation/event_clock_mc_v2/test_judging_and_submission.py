from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.causal.timeline import PhaseSegment
from pipeline.simulation.event_clock_mc_v2.judging.model import (
    EVENT2_TOTAL_JUDGE_ROUND_TRANSFER, JudgeFeatures,
)
from pipeline.simulation.event_clock_mc_v2.judging.scorecards import round_features, score_decision
from pipeline.simulation.event_clock_mc_v2.engine.causal_engine import EngineRNGs
from pipeline.simulation.event_clock_mc_v2.mechanics.submission import (
    stage11c_matchup_probability, submission_conversion_probability,
)


def event(time, actor, action, *, landed=True, kd=False, td=False):
    return SimpleNamespace(
        timestamp_seconds=time, actor=actor, selected_action=action,
        outcome=SimpleNamespace(value="landed" if landed else "failure"),
        knockdown=kd, transition_kind=("td" if td else None),
    )


def segment(start, end, controller=None):
    return PhaseSegment(start, end, Phase.STANDING if controller is None else Phase.GROUND,
                        controller, "test", "test")


def test_round_features_are_local_signed_and_include_failed_submission():
    events = [
        event(10, Side.RED, ActionFamily.STAND_ATTACK),
        event(20, Side.BLUE, ActionFamily.STAND_ATTACK, kd=True),
        event(30, Side.RED, ActionFamily.TAKEDOWN_ENTRY, landed=False, td=True),
        event(40, Side.BLUE, ActionFamily.SUBMISSION_ATTACK, landed=False),
        event(310, Side.RED, ActionFamily.STAND_ATTACK),
    ]
    features = round_features(events, [segment(0, 100, Side.RED), segment(100, 600)], 1, 300)
    assert features == JudgeFeatures(sig_diff=0, kd_diff=-1, td_diff=1, sub_diff=-1, ctrl_diff=100)
    # A future-round event cannot alter the frozen first-round vector.
    assert features == round_features(events[:-1], [segment(0, 100, Side.RED), segment(100, 300)], 1, 300)


def test_each_feature_has_source_learned_red_direction_and_blue_symmetry():
    model = EVENT2_TOTAL_JUDGE_ROUND_TRANSFER
    neutral = model.probability(JudgeFeatures())
    for name in ("sig_diff", "kd_diff", "td_diff", "sub_diff", "ctrl_diff"):
        red = model.probability(JudgeFeatures(**{name: 1}))
        blue = model.probability(JudgeFeatures(**{name: -1}))
        assert red > neutral > blue


class ScriptRNG:
    def __init__(self, values): self.values = iter(values)
    def random(self): return next(self.values)


@pytest.mark.parametrize("rounds", [3, 5])
def test_three_judges_score_each_completed_round_once(rounds):
    result = score_decision([], [segment(0, rounds * 300)], rounds=rounds, round_length=300,
                            model=EVENT2_TOTAL_JUDGE_ROUND_TRANSFER,
                            rng=ScriptRNG([0.0] * (rounds * 3)))
    assert len(result.scorecards) == 3
    assert all(len(card.rounds) == rounds for card in result.scorecards)
    assert result.winner is Side.RED and result.classification == "unanimous_decision"


def test_independent_three_judge_draws_make_split_majority():
    # Judge 1 red all rounds; judges 2/3 blue all rounds.
    result = score_decision([], [segment(0, 900)], rounds=3, round_length=300,
                            model=EVENT2_TOTAL_JUDGE_ROUND_TRANSFER,
                            rng=ScriptRNG([0, 0, 0, 1, 1, 1, 1, 1, 1]))
    assert [card.winner for card in result.scorecards] == [Side.RED, Side.BLUE, Side.BLUE]
    assert result.winner is Side.BLUE and result.classification == "split_decision"


def test_committed_stage10d_probabilities_have_total_fight_parity():
    frame = pd.read_csv("data/diagnostics/event_clock_mc_v1/stage10d_total_fight_judge.csv")
    for row in frame.iloc[[0, 40, 100, 180, 248]].itertuples():
        features = JudgeFeatures(row.sig_diff, row.kd_diff, row.td_diff, row.sub_diff, row.ctrl_diff)
        assert EVENT2_TOTAL_JUDGE_ROUND_TRANSFER.probability(features) == pytest.approx(row.full_total_p_red, abs=2e-12)


@pytest.mark.parametrize("baseline,offense,defense", [(.1,0,0),(.1,1,0),(.1,-1,0),(.1,0,1),(.1,0,-1),(.01,.2,-.1),(.5,.2,-.1)])
def test_stage11c_formula_parity(baseline, offense, defense):
    expected = 1 / (1 + np.exp(-(np.log(baseline / (1-baseline)) + offense - defense)))
    assert stage11c_matchup_probability(baseline, offense, defense) == pytest.approx(expected, abs=1e-15)


def test_integrated_replay_offset_is_exact_and_has_no_unapproved_predictors():
    base = submission_conversion_probability(.1, .25)
    expected = 1 / (1 + np.exp(-(np.log(.1/.9) + .25)))
    assert base == pytest.approx(expected, abs=1e-15)
    # Age, stamina, trauma and KO/KD state are absent from the function contract.
    assert submission_conversion_probability(.1, .25) == base


def test_submission_and_judge_streams_preserve_existing_six_identities():
    expected = [np.random.default_rng(s).random(8) for s in np.random.SeedSequence(91).spawn(6)]
    rngs = EngineRNGs.from_seed(91)
    actual = [rngs.red_timing, rngs.blue_timing, rngs.red_selection,
              rngs.blue_selection, rngs.mechanics, rngs.ko_kd]
    for old, current in zip(expected, actual):
        assert np.array_equal(old, current.random(8))
