"""Research-only Allen-Shahbazyan path 1 trace with standing pressure and strike-context inflation disabled.

Keeps the current research setup unchanged except:
- standing strike intent is anchored directly to the prefight FSR standing attempt rate;
- dynamic pressure does not multiply standing strike rate;
- live strike_context_factor does not multiply standing strike rate.

TD and clinch live context remain active. RESET_RANGE, IMPROVE_POSITION and
ADVANCE_POSITION remain removed, and the OOS-validated expected-control escape
model remains active. Production code is unchanged.
"""
from __future__ import annotations

from pipeline.research import allen_shahbazyan_expected_control_escape_trace as target
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_intent_rate_shadow as intent_mod

_original_standing_rates = intent_mod._standing_rates


def _standing_rates_raw_fsr_strike_no_reset(state, actor, capabilities, context, priors, config):
    rates, _pressure = _original_standing_rates(
        state, actor, capabilities, context, priors, config
    )
    rates = dict(rates)
    # Remove BOTH dynamic pressure and live strike-context inflation from the
    # standing strike clock. The strike rate is now exactly the prefight FSR
    # standing-attempt prior. TD/clinch context are intentionally unchanged.
    rates[ActionFamily.STAND_ATTACK] = max(
        float(priors.standing_attempt_rate_15m), 1e-12
    )
    rates.pop(ActionFamily.RESET_RANGE, None)
    return rates, 0.0


def main():
    # TraceBrain resolves this function from its defining module at runtime.
    base_trace._standing_rates_no_reset = _standing_rates_raw_fsr_strike_no_reset
    # Expected-control trace installs this function into IntentRateBrain for
    # both the standing event clock and standing action chooser.
    target._standing_rates_no_reset = _standing_rates_raw_fsr_strike_no_reset
    target.main()


if __name__ == "__main__":
    main()
