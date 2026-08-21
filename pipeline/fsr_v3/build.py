"""Build the currently validated FSR V3 trait families without touching FSR V2."""

from __future__ import annotations

import argparse

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.paths import (
    FSR_V3_HISTORY_DIR,
    GROUND_EFFECTIVENESS_HISTORY_PATH,
    GROUND_SUPPRESSION_HISTORY_PATH,
    GROUND_TENDENCY_HISTORY_PATH,
)
from pipeline.fsr_v3.publish import publish
from pipeline.fsr_v3.replay.ground import (
    build_ground_fighter_fights,
    replay_ground_suppression,
    replay_ground_tendency,
)
from pipeline.fsr_v3.replay.ground_effectiveness import replay_ground_effectiveness


def build_ground(*, publish_snapshots: bool = True) -> None:
    config = FSRV3Config()
    fights = build_ground_fighter_fights()
    tendency = replay_ground_tendency(fights, config)
    suppression = replay_ground_suppression(tendency, config)
    effectiveness = replay_ground_effectiveness(fights, config)

    FSR_V3_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    tendency.to_parquet(GROUND_TENDENCY_HISTORY_PATH, index=False)
    suppression.to_parquet(GROUND_SUPPRESSION_HISTORY_PATH, index=False)
    effectiveness.to_parquet(GROUND_EFFECTIVENESS_HISTORY_PATH, index=False)

    print(
        "FSR V3 ground histories written: "
        f"tendency={len(tendency):,}, "
        f"suppression={len(suppression):,}, "
        f"effectiveness={len(effectiveness):,}"
    )

    if publish_snapshots:
        prefight, latest, uncertainty = publish()
        print(
            "FSR V3 ground overlay published: "
            f"prefight={len(prefight):,}, latest={len(latest):,}, "
            f"uncertainty={len(uncertainty):,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-striking",
        action="store_true",
        help="Build the fully validated V3 ground-striking family.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build every FSR V3 family currently implemented (ground striking only at this checkpoint).",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Write histories only; do not publish V3 snapshot overlays.",
    )
    args = parser.parse_args()
    if not args.ground_striking and not args.all:
        parser.error("select --ground-striking or --all")
    build_ground(publish_snapshots=not args.no_publish)


if __name__ == "__main__":
    main()
