from __future__ import annotations

"""One-command development runner for UFC Raw Signal Discovery V1."""

import argparse
from pathlib import Path

from pipeline.research.raw_signal_discovery_v1 import build_feature_bank
from pipeline.research.raw_signal_discovery_v1.train_discovery import run as train

DEFAULT_CONFIG = Path("pipeline/research/raw_signal_discovery_v1/config.yaml")


def _build_with_master_event_date(config: Path) -> None:
    """Avoid a duplicate event_date merge when round stats already carry one.

    The canonical fight date for this discovery study is the master fight date.
    Some round-stats versions also contain ``event_date``; dropping that copy
    before the builder attaches master metadata prevents pandas from suffixing
    the merged columns to event_date_x/event_date_y.
    """

    original = build_feature_bank._build_fight_observations

    def _without_round_event_date(rounds, master):
        return original(rounds.drop(columns=["event_date"], errors="ignore"), master)

    build_feature_bank._build_fight_observations = _without_round_event_date
    try:
        build_feature_bank.build(config)
    finally:
        build_feature_bank._build_fight_observations = original


def main() -> None:
    parser = argparse.ArgumentParser(description="Run development-only raw UFC signal discovery.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = Path(args.config)
    _build_with_master_event_date(config)
    train(config)


if __name__ == "__main__":
    main()
