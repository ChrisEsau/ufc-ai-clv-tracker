"""FSR V1.2 historical benchmark with dynamic adversity enabled.

This wrapper deliberately reuses the established generic historical FSR runner
while swapping only its dynamic-state calibration bundle.

Why this exists
---------------
The base runner imports the frozen V1 power-decay calibration, whose adversity
coefficients are all zero. That intentionally leaves persistent damage and
acute stress disabled. For the FSR V1.2 experiment we want to preserve the
same fight construction, seeds, scoring, finish calibration, and diagnostics
while activating the already-defined V2 adversity calibration.

This wrapper therefore replaces only these runner globals before calling
``main()``:

- ``state_calibration``
- ``phase_effect_calibration``

Finish calibration and transition-effect calibration remain those already used
by the base runner. The V2 adversity module itself preserves the frozen V1
fatigue behavior and activates damage/stress accumulation, recovery, and phase
capability effects.

Shadow/research only.
"""

from __future__ import annotations

from scripts.calibrate_rfs_mc_v2_dynamic_adversity_v2 import (
    phase_effect_calibration as adversity_phase_effect_calibration,
    state_calibration as adversity_state_calibration,
)
from scripts.experimental import run_fsr_historical_fight_v1 as base_runner


def main() -> None:
    """Run the existing historical benchmark with V2 adversity enabled."""

    # The base runner resolves these names dynamically when it builds both the
    # authoritative Monte Carlo population and its deterministic diagnostics
    # replay. Replacing the module globals keeps both passes on the same
    # calibration bundle.
    base_runner.state_calibration = adversity_state_calibration
    base_runner.phase_effect_calibration = adversity_phase_effect_calibration

    base_runner.main()


if __name__ == "__main__":
    main()
