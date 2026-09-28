"""Run the existing aligned single-fight comparison with stamina V3.1."""
from __future__ import annotations

from scripts.experimental import fsr_single_fight_stamina_compare_aligned as compare
from scripts.experimental import fsr_static_mc_ko_tko_v3_1_stamina as stamina31


def main() -> None:
    # Reuse the already-validated aligned cohort, identical seed stream, report,
    # and baseline. Replace only the shadow stamina class under test.
    compare.stamina.StaticFSRMCKOTKOV3Stamina = stamina31.StaticFSRMCKOTKOV31Stamina
    compare.main()


if __name__ == "__main__":
    main()
