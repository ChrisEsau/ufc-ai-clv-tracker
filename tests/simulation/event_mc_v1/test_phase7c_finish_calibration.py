import json

from pipeline.simulation.event_mc_v1.calibration import DEFAULT_CALIBRATION
from pipeline.simulation.event_mc_v1.diagnostics.phase7c_finish_calibration import calibration_for_finish_midpoint, evaluate
from pipeline.simulation.event_mc_v1.diagnostics.phase7b_kd_calibration import temporal_cohorts


def test_override_changes_only_finish_midpoint_and_keeps_kd_36():
    candidate=calibration_for_finish_midpoint(64)
    assert candidate.section("finish")["midpoint_impact_ratio"]==64
    assert candidate.section("knockdown")["midpoint_impact_ratio"]==36
    for section,values in DEFAULT_CALIBRATION.values.items():
        for key,value in values.items():
            if (section,key)!=("finish","midpoint_impact_ratio"):
                assert candidate.section(section)[key]==value


def test_same_seed_candidate_is_deterministic():
    train,_,fsr=temporal_cohorts(1,1);a=evaluate(train,fsr,10,2,77);b=evaluate(train,fsr,10,2,77);a.pop("runtime_seconds");b.pop("runtime_seconds");assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
