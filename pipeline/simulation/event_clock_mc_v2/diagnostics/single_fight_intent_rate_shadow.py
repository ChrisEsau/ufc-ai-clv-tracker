"""Generic single-fight runner for the research-only intent-rate Brain shadow.

Reuses the validated Leavitt-Brito shadow implementation without changing
production Brain, engine, mechanics, judging, FSR, or seed semantics.
"""
from __future__ import annotations

import argparse
import json

from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fight-id", required=True)
    parser.add_argument("--paths", type=int, default=500)
    args = parser.parse_args()
    if args.paths < 1:
        raise ValueError("paths must be positive")

    fight_id = str(args.fight_id)
    paths = int(args.paths)

    # The existing research modules intentionally use constants so matched-seed
    # diagnostics are fixed. Override only those research constants here.
    pressure_mod.FIGHT_ID = fight_id
    pressure_mod.PATHS = paths
    intent_mod.FIGHT_ID = fight_id
    intent_mod.PATHS = paths

    fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
    result = intent_mod.run_intent_rate_condition(fight, inputs, priors, horizon, cfg)

    payload = {
        "diagnostic": "generic single-fight Brain intent-rate shadow",
        "fight_id": fight_id,
        "paths": paths,
        "seed_set": SEED_SET_VERSION,
        "production_changed": False,
        "red_fighter": str(fight.r_name),
        "blue_fighter": str(fight.b_name),
        "fsr_effective_rates": {
            "red": {
                "standing_rate_15m": priors[pressure_mod.Side.RED].standing_attempt_rate_15m,
                "takedown_rate_15m": priors[pressure_mod.Side.RED].takedown_attempt_rate_15m,
            },
            "blue": {
                "standing_rate_15m": priors[pressure_mod.Side.BLUE].standing_attempt_rate_15m,
                "takedown_rate_15m": priors[pressure_mod.Side.BLUE].takedown_attempt_rate_15m,
            },
        },
        "result": result,
    }
    print("SINGLE_FIGHT_INTENT_RATE_SHADOW")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
