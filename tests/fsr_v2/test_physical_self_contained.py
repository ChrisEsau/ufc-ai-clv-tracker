from math import exp, isclose
from pathlib import Path
import inspect

import pandas as pd
import pytest

from pipeline.fsr_v2 import physical


def _round(fight, date, fighter, opponent, rnd, **changes):
    row = dict(event_date=pd.Timestamp(date), fight_id=fight, fighter_id=fighter,
        fighter_name=fighter.upper(), opponent_id=opponent, round=rnd, kd=0,
        sig_str_landed=5, sig_str_attempted=10, total_str_landed=10,
        total_str_attempted=20, td_landed=0, td_attempted=0, sub_att=0,
        rev=0, ctrl_sec=10, head_landed=3, ground_landed=0)
    row.update(changes)
    return row


def _sources():
    rounds=[]
    for fight,date,a,b in (("f1","2020-01-01","a","b"),("f2","2020-02-01","a","b")):
        for fighter,opponent in ((a,b),(b,a)):
            for rnd in (1,2,3):
                rounds.append(_round(fight,date,fighter,opponent,rnd,
                    sig_str_landed=5*rnd,sig_str_attempted=10*rnd,
                    total_str_landed=8*rnd,total_str_attempted=20*rnd,
                    td_attempted=rnd-1,ctrl_sec=10*rnd,
                    kd=int(fighter=="a" and rnd==1 and fight=="f2")))
    master=pd.DataFrame([
        dict(fight_id="f1",date="2020-01-01",method="Decision",finish_round=3,match_time_sec=900,winner_id="a"),
        dict(fight_id="f2",date="2020-02-01",method="KO/TKO",finish_round=2,match_time_sec=500,winner_id="a"),
    ])
    return pd.DataFrame(rounds),master


def test_production_physical_module_has_no_legacy_runtime_imports():
    source=inspect.getsource(physical)
    assert "scripts.experimental" not in source
    assert "ROUND_FIGHTER_STATE_HISTORY_PATH" not in source


def test_raw_stamina_observations_reproduce_frozen_round_math():
    rounds,master=_sources()
    obs=physical.build_physical_observations(rounds,master)
    row=obs[(obs.fight_id=="f1")&(obs.fighter_id=="a")].iloc[0]
    assert row.sig_attempt_first_last_ratio==3
    assert row.total_attempt_first_last_ratio==3
    assert row.sig_attempt_slope==pytest.approx(10)
    assert row.total_attempt_slope==pytest.approx(20)
    assert row.td_attempt_slope==pytest.approx(1)
    assert row.control_slope==pytest.approx(10)
    assert row.late_early_workload_ratio==pytest.approx(3)
    assert row.late_early_output_ratio==pytest.approx(3)
    assert row.sig_accuracy_change==pytest.approx(0)
    assert row.total_accuracy_change==pytest.approx(0)
    units=row.sig_attempts/60+row.td_attempts/4+row.control_seconds/180
    expected=(1-exp(-(3-1)/2))*(1-exp(-units))
    assert physical.stamina_quality(row)==pytest.approx(expected)


def test_stamina_constants_expected_update_and_same_date_isolation():
    assert physical.stamina_k(0)==7
    assert physical.stamina_k(6)==pytest.approx(7/(2**.5))
    rounds,master=_sources()
    snapshots=physical.build_physical_snapshots(rounds=rounds,master=master)
    first=snapshots.prefight[snapshots.prefight.fight_id=="f1"]
    assert first.stamina_depletion_resistance.eq(50).all()
    assert first.stamina_performance_resilience.eq(50).all()
    # Latest is a true post-f2 state rather than f2's prefight value.
    a_pre=snapshots.prefight.query("fight_id == 'f2' and fighter_id == 'a'").iloc[0]
    a_latest=snapshots.latest.query("fighter_id == 'a'").iloc[0]
    assert (a_latest.stamina_depletion_resistance,a_latest.stamina_performance_resilience)!=(
        a_pre.stamina_depletion_resistance,a_pre.stamina_performance_resilience)

    # A second bout copied onto the same first date sees the identical prefight
    # state; neither observation is applied until every date snapshot exists.
    extra_rounds=rounds[rounds.fight_id.eq("f1")].copy()
    extra_rounds["fight_id"]="f1b"
    extra_master=master[master.fight_id.eq("f1")].copy();extra_master["fight_id"]="f1b"
    same_date=physical.build_physical_snapshots(
        rounds=pd.concat([rounds,extra_rounds],ignore_index=True),
        master=pd.concat([master,extra_master],ignore_index=True),
    ).prefight
    a_same=same_date.query("fighter_id == 'a' and fight_id in ['f1','f1b']")
    assert a_same.stamina_depletion_resistance.nunique()==1
    assert a_same.stamina_performance_resilience.nunique()==1
    assert a_same.striking_power.nunique()==1


def test_paired_power_observations_and_direct_rating_mapping():
    rounds,master=_sources();obs=physical.build_physical_observations(rounds,master)
    ko=obs.query("fight_id == 'f2' and fighter_id == 'a'").iloc[0]

    # Paired power consumes full-fight damaging events and full-fight
    # significant-strike landing opportunity.
    assert ko.opponent_id=="b"
    assert ko.kd_scored==pytest.approx(1.0)
    assert ko.sig_landed==pytest.approx(30.0)
    assert ko.ko_win==pytest.approx(1.0)

    # Public 35-90 power is a direct monotonic translation of raw paired state.
    assert physical._power_rating(0.0)==pytest.approx(50.0)
    assert physical._power_rating(0.10)==pytest.approx(70.0)
    assert physical._power_rating(-0.05)==pytest.approx(40.0)
    assert physical._power_rating(0.30)==pytest.approx(90.0)
    assert physical._power_rating(-0.20)==pytest.approx(35.0)


def test_resistance_component_and_rating_equations():
    s=physical._new_resistance_state();s.update(fights=4,kd_absorbed=1,sig_absorbed=100,
        kd_free_fights=3,kd_high_exposure_fights=2,kd_free_high_exposure=1)
    avoidance,free,high=physical._kd_components(s)
    assert avoidance==pytest.approx(-(1.5/150));assert free==.75;assert high==.5
    expected_confidence=1-exp(-4/3)
    assert physical._rating(.75,4)==pytest.approx(10+80*(.5+expected_confidence*.25))
    row=pd.Series(dict(rounds_observed=2,kd_absorbed=2,head_absorbed=20,
        ground_absorbed=10,opponent_control_seconds=120))
    assert physical._damage_exposure(row)==pytest.approx((1+10+5+1)/4)
    s.update(dur_high_exposure_fights=2,dur_high_survivals=1,dur_high_exposure_sum=8,
        dur_high_survived_exposure_sum=3,survived_exposure_sum=6,survived_fights=2,ko_losses=1)
    expected=(.35*.5+.30*(3/8)+.20*((3/4)/2)+.15*.75)/(.35+.30+.20+.15)
    assert physical._dur_score(s,4)==pytest.approx(expected)
