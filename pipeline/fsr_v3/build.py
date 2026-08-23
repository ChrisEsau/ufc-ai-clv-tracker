"""Build validated FSR V3 trait families without touching FSR V2."""

from __future__ import annotations

import argparse

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.paths import (
    FSR_V3_HISTORY_DIR,
    GROUND_EFFECTIVENESS_HISTORY_PATH,
    GROUND_SUPPRESSION_HISTORY_PATH,
    GROUND_TENDENCY_HISTORY_PATH,
    POWER_HISTORY_PATH,
    STANDING_EFFECTIVENESS_HISTORY_PATH,
    STANDING_SUPPRESSION_HISTORY_PATH,
    STANDING_TENDENCY_HISTORY_PATH,
    TAKEDOWN_EFFECTIVENESS_HISTORY_PATH,
    TAKEDOWN_SUPPRESSION_HISTORY_PATH,
    TAKEDOWN_TENDENCY_HISTORY_PATH,
)
from pipeline.fsr_v3.publish import publish
from pipeline.fsr_v3.replay.ground import (
    build_ground_fighter_fights,
    replay_ground_suppression,
    replay_ground_tendency,
)
from pipeline.fsr_v3.replay.ground_effectiveness import replay_ground_effectiveness
from pipeline.fsr_v3.replay.paired_effectiveness import (
    build_effectiveness_fighter_fights,
    replay_paired_effectiveness,
    standing_effectiveness_spec,
    takedown_effectiveness_spec,
)
from pipeline.fsr_v3.replay.power import replay_power
from pipeline.fsr_v3.replay.rate_families import (
    build_rate_fighter_fights,
    replay_suppression,
    replay_tendency,
    standing_spec,
    takedown_spec,
)


def _write(frame, path) -> None:
    FSR_V3_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def build_takedowns(paired, config: FSRV3Config) -> None:
    rate_spec = takedown_spec(config)
    rate_fights = build_rate_fighter_fights(rate_spec, paired_rounds=paired)
    tendency = replay_tendency(rate_fights, rate_spec)
    suppression = replay_suppression(tendency, rate_spec)

    eff_spec = takedown_effectiveness_spec(config)
    eff_fights = build_effectiveness_fighter_fights(eff_spec, paired_rounds=paired)
    effectiveness = replay_paired_effectiveness(eff_fights, eff_spec)

    _write(tendency, TAKEDOWN_TENDENCY_HISTORY_PATH)
    _write(suppression, TAKEDOWN_SUPPRESSION_HISTORY_PATH)
    _write(effectiveness, TAKEDOWN_EFFECTIVENESS_HISTORY_PATH)
    print(
        "FSR V3 takedown histories written: "
        f"tendency={len(tendency):,}, suppression={len(suppression):,}, "
        f"effectiveness={len(effectiveness):,}"
    )


def build_standing(paired, config: FSRV3Config) -> None:
    rate_spec = standing_spec(config)
    rate_fights = build_rate_fighter_fights(rate_spec, paired_rounds=paired)
    tendency = replay_tendency(rate_fights, rate_spec)
    suppression = replay_suppression(tendency, rate_spec)

    eff_spec = standing_effectiveness_spec(config)
    eff_fights = build_effectiveness_fighter_fights(eff_spec, paired_rounds=paired)
    effectiveness = replay_paired_effectiveness(eff_fights, eff_spec)

    _write(tendency, STANDING_TENDENCY_HISTORY_PATH)
    _write(suppression, STANDING_SUPPRESSION_HISTORY_PATH)
    _write(effectiveness, STANDING_EFFECTIVENESS_HISTORY_PATH)
    print(
        "FSR V3 standing histories written: "
        f"tendency={len(tendency):,}, suppression={len(suppression):,}, "
        f"effectiveness={len(effectiveness):,}"
    )


def build_ground(paired, config: FSRV3Config) -> None:
    fights = build_ground_fighter_fights(paired_rounds=paired)
    tendency = replay_ground_tendency(fights, config)
    suppression = replay_ground_suppression(tendency, config)
    effectiveness = replay_ground_effectiveness(fights, config)

    _write(tendency, GROUND_TENDENCY_HISTORY_PATH)
    _write(suppression, GROUND_SUPPRESSION_HISTORY_PATH)
    _write(effectiveness, GROUND_EFFECTIVENESS_HISTORY_PATH)
    print(
        "FSR V3 ground histories written: "
        f"tendency={len(tendency):,}, suppression={len(suppression):,}, "
        f"effectiveness={len(effectiveness):,}"
    )


def build_power(config: FSRV3Config) -> None:
    history = replay_power(config)
    _write(history, POWER_HISTORY_PATH)
    active = history[history["validated_regime"].astype(bool)]
    print(
        "FSR V3 striking power history written: "
        f"rows={len(history):,}, validated_2020plus={len(active):,}, "
        f"sigma={config.power_sigma:.2f}, rho={config.power_rho:.2f}, c=0"
    )


def _all_histories_exist() -> bool:
    paths = (
        TAKEDOWN_TENDENCY_HISTORY_PATH,
        TAKEDOWN_SUPPRESSION_HISTORY_PATH,
        TAKEDOWN_EFFECTIVENESS_HISTORY_PATH,
        STANDING_TENDENCY_HISTORY_PATH,
        STANDING_SUPPRESSION_HISTORY_PATH,
        STANDING_EFFECTIVENESS_HISTORY_PATH,
        GROUND_TENDENCY_HISTORY_PATH,
        GROUND_SUPPRESSION_HISTORY_PATH,
        GROUND_EFFECTIVENESS_HISTORY_PATH,
        POWER_HISTORY_PATH,
    )
    return all(path.is_file() for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takedowns", action="store_true")
    parser.add_argument("--standing-striking", action="store_true")
    parser.add_argument("--ground-striking", action="store_true")
    parser.add_argument("--power", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Build all validated V3 families: takedown, standing striking, "
            "ground striking, and striking power."
        ),
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Write histories only; do not publish canonical V3 snapshots.",
    )
    args = parser.parse_args()
    if not any(
        (
            args.takedowns,
            args.standing_striking,
            args.ground_striking,
            args.power,
            args.all,
        )
    ):
        parser.error(
            "select --takedowns, --standing-striking, --ground-striking, --power, or --all"
        )

    config = FSRV3Config()
    needs_paired = args.all or args.takedowns or args.standing_striking or args.ground_striking
    paired = build_paired_rounds() if needs_paired else None

    if args.all or args.takedowns:
        build_takedowns(paired, config)
    if args.all or args.standing_striking:
        build_standing(paired, config)
    if args.all or args.ground_striking:
        build_ground(paired, config)
    if args.all or args.power:
        build_power(config)

    if not args.no_publish:
        if not _all_histories_exist():
            print(
                "Canonical FSR V3 publication skipped: all ten validated history files "
                "are required. Run `python -m pipeline.fsr_v3.build --all`."
            )
        else:
            prefight, latest, uncertainty = publish()
            print(
                "FSR V3 canonical overlay published: "
                f"prefight={len(prefight):,}, latest={len(latest):,}, "
                f"uncertainty={len(uncertainty):,}"
            )


if __name__ == "__main__":
    main()
