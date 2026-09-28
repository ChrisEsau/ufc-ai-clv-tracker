from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from pipeline.fsr_v2.config import FSRV2Config
from pipeline.fsr_v2.publish.snapshots import assemble_latest, assemble_prefight
from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v2.traits.registry import GROUPS, resolve_groups
from pipeline.fsr_v2.replay.engine import _logit


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
    fights["td_tendency_exposure_seconds"] = 0
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


def test_final_constants_and_target_composition_contract():
    config = FSRV2Config()
    assert config.escape_prior_entries == 5
    assert config.target_composition_prior_attempts == 200
    assert config.takedown_effectiveness_prior_attempts == 10
    fights = aggregate_fights(build_paired_rounds(*_sources()))
    fights.loc[:, ["head_attempted", "body_attempted", "leg_attempted"]] = [10, 5, 5]
    fights["target_attempted"] = 20
    result = ReplayEngine(config).replay(GROUPS["leg_strike_tendency"], fights).history
    # First-date population is neutral zero; denominator is H+B+L plus 200,
    # never the unrelated distance-attempt count.
    assert result.iloc[0].raw_denominator == 20
    assert result.iloc[0].post_rating == pytest.approx(5 / 220)


def test_final_td_exposures_and_cumulative_logit_model():
    paired = build_paired_rounds(*_sources())
    sample = paired.iloc[0]
    assert sample.td_tendency_exposure_seconds == sample.round_elapsed_seconds - sample.opponent_ctrl_sec
    assert sample.td_suppression_exposure_seconds == sample.round_elapsed_seconds - sample.ctrl_sec
    fights = aggregate_fights(paired)
    history = ReplayEngine().replay(GROUPS["takedown_effectiveness"], fights).history
    later = history[(history.fight_id == "f3") & (history.fighter_id == "a")
                    & (history.trait == "takedown_offense")].iloc[0]
    baseline = later.population_baseline
    # A had two prior same-date fights with 2/4 takedowns. This is the locked
    # 10-attempt cumulative logit state, not an Elo carry-forward.
    expected = _logit((2 + baseline * 10) / 14) - _logit(baseline)
    assert later.pre_rating == pytest.approx(expected)


def test_ground_qualification_excludes_clinch_and_requires_evidence():
    rounds, master = _sources()
    rounds.loc[:, ["td_landed", "ground_attempted", "sub_att", "rev"]] = 0
    rounds.loc[:, "clinch_attempted"] = 20
    paired = build_paired_rounds(rounds, master)
    assert not paired.explicit_true_ground_evidence.any()
    assert paired.modeled_ground_exposure_seconds.eq(0).all()
    assert paired.ground_entries.eq(0).all()
    fights = aggregate_fights(paired)
    assert fights.ground_attempted.eq(0).all()


def test_submission_finish_creates_effective_attempt_and_escape_is_duration_rating():
    rounds, master = _sources()
    master.loc[master.fight_id == "f1", ["method", "winner_id"]] = ["Submission", "a"]
    rounds.loc[rounds.fight_id == "f1", "sub_att"] = 0
    fights = aggregate_fights(build_paired_rounds(rounds, master))
    winner = fights[(fights.fight_id == "f1") & (fights.fighter_id == "a")].iloc[0]
    assert winner.effective_submission_attempts == 1
    escape = ReplayEngine().replay(GROUPS["escape_effectiveness"], fights).history
    assert set(escape.trait) == {"escape_offense", "escape_defense"}
    assert "update" not in escape.columns  # direct log-duration state, not Elo


def test_latest_behavior_recenter_uses_final_population_baseline():
    fights = aggregate_fights(build_paired_rounds(*_sources()))
    result = ReplayEngine().replay(GROUPS["standing_striking_tendency"], fights)
    latest = assemble_latest(result.history)
    state_n, state_d = result.state["a"]
    final_population = result.population["numerator"] / result.population["denominator"]
    expected = (state_n + final_population * 900) / (state_d + 900)
    actual = latest.loc[latest.fighter_id.eq("a"), "standing_striking_tendency"].iloc[0]
    assert actual == pytest.approx(expected)


def test_takedown_tendency_endpoint2_prior_rule():
    fights = aggregate_fights(
        build_paired_rounds(*_sources())
    )

    # --------------------------------------------------------
    # ONE prior UFC fight:
    # retain the standard 900-second population prior.
    # --------------------------------------------------------

    one_prior = fights[
        ~fights["fight_id"].eq("f2")
    ].copy()

    history = ReplayEngine().replay(
        GROUPS["takedown_tendency"],
        one_prior,
    ).history

    later = history[
        history["fight_id"].eq("f3")
        & history["fighter_id"].eq("a")
    ].iloc[0]

    first_fight = one_prior[
        one_prior["fight_id"].eq("f1")
    ]

    global_rate = (
        first_fight["td_attempted"].sum()
        / first_fight[
            "td_tendency_exposure_seconds"
        ].sum()
    )

    a_first = first_fight[
        first_fight["fighter_id"].eq("a")
    ].iloc[0]

    expected_shrunk = (
        a_first["td_attempted"]
        + global_rate * 900.0
    ) / (
        a_first[
            "td_tendency_exposure_seconds"
        ]
        + 900.0
    )

    assert later["prior_ufc_fights"] == 1
    assert later["population_prior_seconds"] == 900.0
    assert later["pre_rating"] == pytest.approx(
        expected_shrunk
    )

    # --------------------------------------------------------
    # TWO prior UFC fights:
    # remove the population prior and use raw observed rate.
    # --------------------------------------------------------

    full_history = ReplayEngine().replay(
        GROUPS["takedown_tendency"],
        fights,
    ).history

    later = full_history[
        full_history["fight_id"].eq("f3")
        & full_history["fighter_id"].eq("a")
    ].iloc[0]

    a_prior = fights[
        fights["fight_id"].isin(["f1", "f2"])
        & fights["fighter_id"].eq("a")
    ]

    expected_raw = (
        a_prior["td_attempted"].sum()
        / a_prior[
            "td_tendency_exposure_seconds"
        ].sum()
    )

    assert later["prior_ufc_fights"] == 2
    assert later["population_prior_seconds"] == 0.0
    assert later["pre_rating"] == pytest.approx(
        expected_raw
    )

