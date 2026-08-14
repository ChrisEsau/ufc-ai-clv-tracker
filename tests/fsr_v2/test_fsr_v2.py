from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from pipeline.fsr_v2.config import FSRV2Config
from pipeline.fsr_v2.publish.snapshots import assemble_latest, assemble_prefight
from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v2.traits.registry import GROUPS, resolve_groups


def _sources():
    rows = []
    for date, fight, attempts_a, landed_a in [
        ("2020-01-01", "f1", 10, 5), ("2020-01-01", "f2", 4, 1),
        ("2020-02-01", "f3", 10, 8),
    ]:
        for fighter, opponent, corner in [("a", "b", "red"), ("b", "a", "blue")]:
            attempts = attempts_a if fighter == "a" else 6
            landed = landed_a if fighter == "a" else 3
            rows.append({
                "event_id": date, "event_date": pd.Timestamp(date), "fight_id": fight,
                "fighter_id": fighter, "fighter_name": fighter.upper(), "opponent_id": opponent,
                "opponent_name": opponent.upper(), "round": 1, "sig_str_landed": landed,
                "sig_str_attempted": attempts, "td_landed": int(fighter == "a"), "td_attempted": 2,
                "sub_att": 0, "rev": 0, "ctrl_sec": 20 if fighter == "a" else 10,
                "head_landed": landed, "head_attempted": attempts, "body_landed": 0,
                "body_attempted": 0, "leg_landed": 0, "leg_attempted": 0,
                "distance_landed": landed, "distance_attempted": attempts, "clinch_landed": 0,
                "clinch_attempted": 0, "ground_landed": 0, "ground_attempted": 0,
            })
    master = pd.DataFrame([
        {"fight_id": f, "finish_round": 1, "match_time_sec": 300, "method": "Decision", "winner_id": "a"}
        for f in ["f1", "f2", "f3"]
    ])
    return pd.DataFrame(rows), master


def test_pairing_exposure_control_cap_and_elapsed_normalization():
    rounds, master = _sources()
    master.loc[0, ["finish_round", "match_time_sec"]] = [3, 120]
    rounds.loc[rounds.fight_id == "f1", "round"] = 3
    paired = build_paired_rounds(rounds, master)
    assert paired.loc[paired.fight_id == "f1", "round_elapsed_seconds"].eq(120).all()
    assert paired["ground_exposure_seconds"].eq(30).all()
    assert paired["standing_exposure_seconds"].ge(0).all()


def test_same_date_isolation_and_update_direction():
    fights = aggregate_fights(build_paired_rounds(*_sources()))
    result = ReplayEngine().replay(GROUPS["standing_striking_effectiveness"], fights).history
    same_date = result[(result.event_date == pd.Timestamp("2020-01-01")) & (result.trait == "standing_striking_offense")]
    assert same_date.pre_rating.eq(0).all()
    a_later = result[(result.fight_id == "f3") & (result.fighter_id == "a") & (result.trait == "standing_striking_offense")].iloc[0]
    assert a_later.pre_rating != 0
    assert a_later["update"] > 0


def test_zero_opportunity_does_not_update_behavior():
    fights = aggregate_fights(build_paired_rounds(*_sources()))
    fights["standing_exposure_seconds"] = 0
    history = ReplayEngine().replay(GROUPS["takedown_tendency"], fights).history
    assert history.observed.isna().all()
    assert history.pre_rating.eq(history.post_rating).all()


def test_suppression_positive_when_opponent_below_prior_rate():
    fights = aggregate_fights(build_paired_rounds(*_sources()))
    history = ReplayEngine().replay(GROUPS["standing_striking_suppression"], fights).history
    later_b = history[(history.fight_id == "f3") & (history.fighter_id == "b")].iloc[0]
    assert later_b.opponent_expected_rate < later_b.opponent_actual_rate
    assert later_b.observed < 0


def test_replay_determinism_and_publish_assembly():
    fights = aggregate_fights(build_paired_rounds(*_sources()))
    group = GROUPS["takedown_effectiveness"]
    first = ReplayEngine().replay(group, fights).history
    second = ReplayEngine().replay(group, fights).history
    pdt.assert_frame_equal(first, second)
    prefight = assemble_prefight(first)
    latest = assemble_latest(first)
    assert len(prefight) == 6
    assert {"takedown_offense", "takedown_defense"}.issubset(latest.columns)


def test_selective_registry_and_experimental_isolation():
    assert [g.name for g in resolve_groups(["takedowns"])] == [
        "takedown_tendency", "takedown_suppression", "takedown_effectiveness"
    ]
    assert not any(group.experimental for group in resolve_groups(None))
    assert resolve_groups(["reversal_tendency"], include_experimental=True)[0].experimental
