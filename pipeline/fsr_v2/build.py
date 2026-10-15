"""Selective FSR V2 replay CLI."""

import argparse
from pathlib import Path
import time

from pipeline.common.paths import FSR_V2_CHECKPOINT_DIR, FSR_V2_HISTORY_DIR, ROUND_STATS_PATH
from pipeline.fsr_v2.config import FSRV2Config
from pipeline.fsr_v2.replay.checkpoint import load_checkpoint, write_checkpoint
from pipeline.fsr_v2.replay.dependency_graph import dependency_versions, order_groups
from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.replay.versions import source_fingerprint
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds, load_round_stats
from pipeline.fsr_v2.traits.registry import resolve_groups


def build(names: list[str] | None = None, *, resume: bool = False,
          include_experimental: bool = False, output_dir: Path = FSR_V2_HISTORY_DIR,
          checkpoint_dir: Path = FSR_V2_CHECKPOINT_DIR) -> dict[str, str]:
    config = FSRV2Config()
    raw = load_round_stats()
    fingerprint = source_fingerprint(ROUND_STATS_PATH, raw)
    fights = aggregate_fights(build_paired_rounds(rounds=raw, config=config))
    groups = order_groups(resolve_groups(names, include_experimental))
    fingerprints = {group.name: group.fingerprint(config, fingerprint) for group in groups}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    for group in groups:
        history_path = output_dir / f"{group.name}.parquet"
        checkpoint_path = checkpoint_dir / f"{group.name}.json"
        checkpoint = load_checkpoint(checkpoint_path)
        dependencies = dependency_versions(group, fingerprints)
        valid = bool(checkpoint and history_path.exists()
                     and checkpoint.get("version") == fingerprints[group.name]
                     and checkpoint.get("dependency_versions") == dependencies
                     and checkpoint.get("last_processed_date") == str(fights["event_date"].max().date()))
        if resume and valid:
            statuses[group.name] = "CACHE VALID"
            print(f"{group.name}: CACHE VALID")
            continue
        reason = "NEW TRAIT" if checkpoint is None else (
            "DEPENDENCY INVALIDATED" if checkpoint.get("dependency_versions") != dependencies else "REPLAY REQUIRED"
        )
        started = time.perf_counter()
        result = ReplayEngine(config).replay(group, fights)
        result.history.to_parquet(history_path, index=False)
        write_checkpoint(checkpoint_path, {
            "group": group.name, "traits": list(group.traits), "version": fingerprints[group.name],
            "source_fingerprint": fingerprint, "dependency_versions": dependencies,
            "last_processed_date": str(fights["event_date"].max().date()),
            "history_rows": len(result.history), "fighter_state": result.state,
            "population_state": result.population,
            "runtime_seconds": round(time.perf_counter() - started, 6),
        })
        statuses[group.name] = reason
        print(f"{group.name}: {reason} ({len(result.history):,} audit rows)")
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--traits", help="Comma-separated group names or aliases")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-experimental", action="store_true")
    args = parser.parse_args()
    names = None if args.all else [item.strip() for item in args.traits.split(",") if item.strip()]
    build(names, resume=args.resume, include_experimental=args.include_experimental)


if __name__ == "__main__":
    main()
