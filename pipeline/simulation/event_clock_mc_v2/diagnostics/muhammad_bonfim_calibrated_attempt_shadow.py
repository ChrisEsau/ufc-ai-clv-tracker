"""Research-only Muhammad vs Bonfim Brain shadow at calibrated standing cadence.

Applies the historical standing-attempt calibration scale only to the Brain's
STAND_ATTACK intent rate. FSR inputs, dynamic-pressure logic, takedown/clinch/
reset intent, mechanics, judging, and matched seed semantics remain unchanged.
"""
from __future__ import annotations

import json

from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

FIGHT_ID = "5c69b019e6deee41"
PATHS = 500
STANDING_ATTEMPT_SCALE = 0.25


def main() -> None:
    pressure_mod.FIGHT_ID = FIGHT_ID
    pressure_mod.PATHS = PATHS
    intent_mod.FIGHT_ID = FIGHT_ID
    intent_mod.PATHS = PATHS

    original_standing_rates = intent_mod._standing_rates

    def calibrated_standing_rates(state, actor, capabilities, context, priors, config):
        rates, pressure = original_standing_rates(
            state, actor, capabilities, context, priors, config
        )
        rates = dict(rates)
        rates[ActionFamily.STAND_ATTACK] *= STANDING_ATTEMPT_SCALE
        return rates, pressure

    intent_mod._standing_rates = calibrated_standing_rates
    try:
        fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
        result = intent_mod.run_intent_rate_condition(
            fight, inputs, priors, horizon, cfg
        )
    finally:
        intent_mod._standing_rates = original_standing_rates

    payload = {
        "diagnostic": "Muhammad-Bonfim calibrated Brain standing-attempt shadow",
        "fight_id": FIGHT_ID,
        "paths": PATHS,
        "standing_attempt_scale": STANDING_ATTEMPT_SCALE,
        "seed_set": SEED_SET_VERSION,
        "production_changed": False,
        "fsr_changed": False,
        "mechanics_changed": False,
        "judging_changed": False,
        "red_fighter": str(fight.r_name),
        "blue_fighter": str(fight.b_name),
        "fsr_effective_rates": {
            "red": {
                "standing_rate_15m": priors[Side.RED].standing_attempt_rate_15m,
                "takedown_rate_15m": priors[Side.RED].takedown_attempt_rate_15m,
            },
            "blue": {
                "standing_rate_15m": priors[Side.BLUE].standing_attempt_rate_15m,
                "takedown_rate_15m": priors[Side.BLUE].takedown_attempt_rate_15m,
            },
        },
        "result": result,
    }
    print("MUHAMMAD_BONFIM_CALIBRATED_ATTEMPT_SHADOW")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
