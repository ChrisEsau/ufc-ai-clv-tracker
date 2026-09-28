"""Same-seed sensitivity audit using population-centered locked FSR V1.1.

This wrapper reuses the existing sensitivity implementation and the exact V1
FSR -> MC adapter.  Only rating construction/card loading are replaced by the
V1.1 population-centered replay.

Shadow/research only.
"""

from __future__ import annotations

from scripts.experimental import run_fsr_historical_fight_locked_v1_1 as locked_v1_1
from scripts.experimental import run_fsr_locked_family_sensitivity as sensitivity


if __name__ == "__main__":
    # The original sensitivity module keeps the already-tested V1 adapter.
    # Replace only the builder/card-loading entry points and skill registry.
    sensitivity.locked.run_rating_builders = (
        locked_v1_1.run_rating_builders
    )
    sensitivity.locked.build_full_card = (
        locked_v1_1.build_full_card
    )
    sensitivity.locked.LOCKED_SKILLS = (
        locked_v1_1.LOCKED_SKILLS
    )

    sensitivity.main()
