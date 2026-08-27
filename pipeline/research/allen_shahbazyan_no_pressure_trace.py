"""Research-only Allen-Shahbazyan path 1 trace with dynamic pressure disabled.

Keeps the current research setup unchanged except for removing the standing
pressure multiplier from strike intent. FSR standing pace and live tactical
context remain active. RESET_RANGE, IMPROVE_POSITION and ADVANCE_POSITION remain
removed, and the OOS-validated expected-control escape model remains active.
Production code is unchanged.
"""
from __future__ import annotations

from pipeline.research import allen_shahbazyan_expected_control_escape_trace as target
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

_original_standing_rates = intent_mod._standing_rates


def _standing_rates_no_pressure_no_reset(state, actor, capabilities, context, priors, config):
    rates, pressure = _original_standing_rates(
        state, actor, capabilities, context, priors, config
    )
    rates = dict(rates)
    # Original strike rate contains pressure_factor = 0.75 + dynamic_pressure.
    # Divide it back out so strike intent is anchored only by FSR pace and live
    # tactical context. No-pressure means multiplier 1.0, not pressure=0.0
    # (which would still impose a 0.75 multiplier).
    pressure_factor = 0.75 + float(pressure)
    rates[ActionFamily.STAND_ATTACK] = (
        rates[ActionFamily.STAND_ATTACK] / max(pressure_factor, 1e-12)
    )
    rates.pop(ActionFamily.RESET_RANGE, None)
    return rates, 0.0


def main():
    # TraceBrain resolves this function from its defining module at runtime.
    base_trace._standing_rates_no_reset = _standing_rates_no_pressure_no_reset
    # The expected-control trace main resolves its imported function here and
    # installs it into IntentRateBrain for the standing event clock.
    target._standing_rates_no_reset = _standing_rates_no_pressure_no_reset
    target.main()


if __name__ == "__main__":
    main()
