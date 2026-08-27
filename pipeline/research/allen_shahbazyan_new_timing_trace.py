"""Research-only Allen-Shahbazyan path-1 trace with the current calibrated timing stack.

Research architecture only; production is unchanged.
- standing strike clock = 1.0x raw matchup FSR
- RESET_RANGE removed
- takedown clock = 1.0x raw matchup-effective FSR TD rate, live td_factor removed
- clinch-entry clock = PIT clean-round fighter x opponent proxy at fitted 2.349514563106796 scale
- inside-clinch timing unchanged
- expected-control escape research resolver retained
- fighter-level, position-invariant submission intent retained
- IMPROVE_POSITION and ADVANCE_POSITION remain removed
"""
from __future__ import annotations

import math
import pandas as pd

from pipeline.research import allen_shahbazyan_expected_control_escape_trace as target
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.research import allen_shahbazyan_fighter_level_submission_trace as sub_mod
from pipeline.research.clinch_entry_rate_simulation_oos_calibration import (
    build_proxy_table,
    pit_matchup_equiv,
)
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_structural_population import ROUND_STATS

TD_SCALE = 1.0
CLINCH_SCALE = 2.349514563106796
EPS = 1e-12
CLINCH_RATE_BY_SIDE: dict[Side, float] = {}


def _new_timing_rates(state, actor, capabilities, context, priors, config):
    return {
        ActionFamily.STAND_ATTACK: max(float(priors.standing_attempt_rate_15m), EPS),
        ActionFamily.TAKEDOWN_ENTRY: max(float(priors.takedown_attempt_rate_15m) * TD_SCALE, EPS),
        ActionFamily.CLINCH_ENTRY: max(float(CLINCH_RATE_BY_SIDE[actor]), EPS),
    }, 0.0


def _build_clinch_rates():
    pressure_mod.FIGHT_ID = base_trace.FIGHT_ID
    fight, _, _, _, _ = pressure_mod.build_setup()
    target_date = getattr(fight, "date", None) or getattr(fight, "event_date", None)
    if target_date is None:
        raise RuntimeError("fight date unavailable")
    rounds = pd.read_parquet(ROUND_STATS)
    clean, global_equiv, _, _ = build_proxy_table(rounds)
    red_name, blue_name = str(fight.r_name), str(fight.b_name)
    red_eq, red_fn, red_on = pit_matchup_equiv(clean, global_equiv, target_date, red_name, blue_name)
    blue_eq, blue_fn, blue_on = pit_matchup_equiv(clean, global_equiv, target_date, blue_name, red_name)
    rates = {
        Side.RED: 3.0 * red_eq * CLINCH_SCALE,
        Side.BLUE: 3.0 * blue_eq * CLINCH_SCALE,
    }
    print("TIMING_STACK", {
        "standing_scale": 1.0,
        "td_scale": TD_SCALE,
        "live_td_factor": False,
        "clinch_scale": CLINCH_SCALE,
        "inside_clinch_scale": 1.0,
    })
    print("CLINCH_ENTRY_RATES_PER_15", {red_name: rates[Side.RED], blue_name: rates[Side.BLUE]})
    print("CLINCH_PROXY_PRIOR_N", {
        red_name: {"fighter": red_fn, "opponent": red_on},
        blue_name: {"fighter": blue_fn, "opponent": blue_on},
    })
    return rates


def main():
    global CLINCH_RATE_BY_SIDE
    CLINCH_RATE_BY_SIDE = _build_clinch_rates()

    # Install fighter-level submission intent first.
    sub_mod.RATE_PER_15_BY_SIDE = sub_mod._build_submission_rates()
    base_trace.action_probabilities_with_intent_priors = sub_mod._fighter_level_submission_probs

    # Install the same timing function into both chooser and event-clock lookup paths.
    base_trace._standing_rates_no_reset = _new_timing_rates
    target._standing_rates_no_reset = _new_timing_rates
    target.main()


if __name__ == "__main__":
    main()
