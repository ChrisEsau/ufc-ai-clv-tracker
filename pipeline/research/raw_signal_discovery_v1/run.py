from __future__ import annotations

"""One-command development runner for UFC Raw Signal Discovery V1."""

import argparse
from pathlib import Path

from pipeline.research.raw_signal_discovery_v1.build_feature_bank import build
from pipeline.research.raw_signal_discovery_v1.train_discovery import run as train

DEFAULT_CONFIG = Path("pipeline/research/raw_signal_discovery_v1/config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run development-only raw UFC signal discovery.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = Path(args.config)
    build(config)
    train(config)


if __name__ == "__main__":
    main()
