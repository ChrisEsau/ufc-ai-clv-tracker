"""Run the existing 10-second segment diagnostic with the V3.2 phase-stamina engine."""
from __future__ import annotations

from scripts.experimental import fsr_rolling_stamina_segment_diagnostic as diagnostic
from scripts.experimental import fsr_static_mc_ko_tko_v3_2_phase_stamina as v32


# Reuse the validated diagnostic/reporting path while swapping only the engine.
diagnostic.rolling.StaticFSRMCKOTKOV31RollingFSR = v32.StaticFSRMCKOTKOV32PhaseStamina


if __name__ == "__main__":
    diagnostic.main()
