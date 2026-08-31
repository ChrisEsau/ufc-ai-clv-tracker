"""Build the final canonical FSR V3 publication for Event Clock C.

This command builds every previously validated V3 family, then the two final
active-trait promotions (escape and KD resistance), and publishes one canonical
snapshot/uncertainty interface.  It does not modify FSR V2 or Event Clock V1.
"""

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.active_traits import publish_canonical_active_v3
from pipeline.fsr_v3.build import (
    build_ground,
    build_power,
    build_standing,
    build_takedowns,
)
from pipeline.fsr_v3.config import FSRV3Config


def main() -> None:
    config = FSRV3Config()
    paired = build_paired_rounds()
    build_takedowns(paired, config)
    build_standing(paired, config)
    build_ground(paired, config)
    build_power(config)
    prefight, latest, uncertainty = publish_canonical_active_v3()
    print(
        "FINAL CANONICAL FSR V3 BUILT — Event Clock C interface: "
        f"prefight={len(prefight):,}, latest={len(latest):,}, uncertainty={len(uncertainty):,}"
    )


if __name__ == "__main__":
    main()
