"""Research-only Brain submission funnel audit for Allen, Chairez, Schnell.
Production unchanged.
"""
from __future__ import annotations
import json
import pandas as pd
from pipeline.common.paths import MASTER_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod
from pipeline.simulation.event_clock_mc_v2.causal.state import Side

TARGETS = [
    ("Brendan Allen", "Edmen Shahbazyan", "Brendan Allen"),
    ("Bruno Silva", "Edgar Chairez", "Edgar Chairez"),
    ("Matt Schnell", "Alessandro Costa", "Matt Schnell"),
]
PATHS = 500


def resolve_fight_id(a,b):
    m = pd.read_parquet(MASTER_PATH).copy()
    m["date"] = pd.to_datetime(m["date"])
    hit = m[((m.r_name.eq(a)) & (m.b_name.eq(b))) | ((m.r_name.eq(b)) & (m.b_name.eq(a)))]
    hit = hit[hit["date"].dt.year.eq(2026)].sort_values("date")
    if hit.empty: raise RuntimeError(f"fight not found: {a} vs {b}")
    return str(hit.iloc[-1].fight_id)


def main():
    rows=[]
    for a,b,target in TARGETS:
        fid=resolve_fight_id(a,b)
        pressure_mod.FIGHT_ID=fid; pressure_mod.PATHS=PATHS
        intent_mod.FIGHT_ID=fid; intent_mod.PATHS=PATHS
        fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
        out = intent_mod.run_intent_rate_condition(fight, inputs, priors, horizon, cfg)
        side = Side.RED if str(fight.r_name)==target else Side.BLUE
        key = "red" if side is Side.RED else "blue"
        r=out[key]
        rows.append({
            "matchup":f"{a} vs {b}","fighter":target,"fight_id":fid,
            "mc_win_probability":r["win_probability"],
            "td_attempts_per_path":r["td_attempts_per_path"],
            "td_success_per_path":r["td_success_per_path"],
            "td_success_rate":r["td_success_rate"],
            "ground_control_seconds_per_path":r["ground_control_seconds_per_path"],
            "sub_attempts_per_path":r["sub_attempts_per_path"],
            "sub_conversion":r["sub_conversion"],
            "actions_per_path":r["actions_per_path"],
            "brain_rate_diagnostics":r["brain_rate_diagnostics"],
            "fsr_td_rate_15m":priors[side].takedown_attempt_rate_15m,
            "brain_submission_odds_multiplier":priors[side].submission_odds_multiplier,
            "brain_ground_strike_odds_multiplier":priors[side].ground_strike_odds_multiplier,
        })
    print(json.dumps({"study":"submission Brain suppression three-fight audit","production_changed":False,"paths":PATHS,"rows":rows},indent=2,sort_keys=True))

if __name__=="__main__": main()
