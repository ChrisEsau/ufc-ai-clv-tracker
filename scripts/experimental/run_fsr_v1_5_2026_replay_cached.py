"""Cached launcher for the large-cohort FSR/MC V1.5 historical replay.

Shadow/research only.

The first version of ``run_fsr_v1_5_2026_replay.py`` intentionally rebuilt the
entire chronological locked-FSR history on every invocation.  That made even a
five-fight smoke test pay almost the full historical preparation cost.

This launcher keeps the benchmark equations and simulator behavior unchanged,
but removes two repeated setup costs:

1. cache the leakage-safe locked-FSR V1.1 PRE-fight snapshot table against the
   exact local RFS history file signature;
2. reuse the already-audited deterministic V1.4 neutral phase exposure instead
   of replaying 5,000 neutral paths on every process start.

Generated cache files live under ``data/simulation`` and must not be committed.
If the RFS parquet changes size/mtime/row count/date range, the cache is rejected
and rebuilt automatically.

Usage is identical to the underlying replay:

PYTHONPATH=. python \
  scripts/experimental/run_fsr_v1_5_2026_replay_cached.py \
  --year 2026 --max-fights 300 --simulations 250
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.experimental import run_fsr_v1_5_2026_replay as replay


CACHE_DIR = Path(
    "data/simulation/rfs_mc_v2_shared_state/v1_5/replay_cache"
)
SNAPSHOT_CACHE = CACHE_DIR / "locked_fsr_v1_1_prefight_snapshots.parquet"
SNAPSHOT_META = CACHE_DIR / "locked_fsr_v1_1_prefight_snapshots.meta.json"

# Exact deterministic V1.4 reference implied by the validated 5,000-path,
# three-round audit already used by the single-fight V1.5 runner:
#   distance segments = 77,014 / 15,000 rounds
#   clinch segments   = 39,337 / 15,000 rounds
#   ground segments   = 33,649 / 15,000 rounds
# Every ground segment has exactly one owner, hence /30,000 fighter-rounds.
V1_4_NEUTRAL_EXPOSURE = {
    "reference_distance_segments_per_round": 77014.0 / 15000.0,
    "reference_clinch_segments_per_round": 39337.0 / 15000.0,
    "reference_ground_segments_per_round": 33649.0 / 15000.0,
    "reference_ground_owner_segments_per_fighter_round": 33649.0 / 30000.0,
}

_ORIGINAL_BUILD_SNAPSHOTS = replay.build_locked_prefight_snapshots


def _rfs_signature(rfs: pd.DataFrame) -> dict[str, object]:
    """Return a cheap signature tying the cache to the local RFS artifact."""

    path = replay.RFS_PATH
    stat = path.stat()

    dates = pd.to_datetime(rfs["date"], errors="raise")

    return {
        "source_path": str(path),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "rows": int(len(rfs)),
        "min_date": str(dates.min()),
        "max_date": str(dates.max()),
        "locked_skills": list(replay.equations_v1.SKILLS),
    }


def _load_meta() -> dict[str, object] | None:
    if not SNAPSHOT_META.exists():
        return None

    try:
        return json.loads(SNAPSHOT_META.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cached_locked_prefight_snapshots(rfs: pd.DataFrame) -> pd.DataFrame:
    """Reuse the expensive chronological FSR snapshot build when safe."""

    signature = _rfs_signature(rfs)
    cached_meta = _load_meta()

    if SNAPSHOT_CACHE.exists() and cached_meta == signature:
        print(
            "FSR snapshot cache hit: "
            f"{SNAPSHOT_CACHE}"
        )
        snapshots = pd.read_parquet(SNAPSHOT_CACHE)

        required = {
            "fight_id",
            "fighter_id",
            "prior_ufc_fights",
            *replay.equations_v1.SKILLS,
        }
        missing = sorted(required - set(snapshots.columns))
        if not missing and not snapshots.duplicated(
            ["fight_id", "fighter_id"]
        ).any():
            return snapshots

        print(
            "FSR snapshot cache failed schema/key validation; rebuilding."
        )

    print(
        "FSR snapshot cache miss. Building the full chronological locked-FSR "
        "history once; the completed snapshot table will be reused by later "
        "smoke/full replay runs."
    )

    snapshots = _ORIGINAL_BUILD_SNAPSHOTS(rfs)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(SNAPSHOT_CACHE, index=False)
    SNAPSHOT_META.write_text(
        json.dumps(signature, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "Saved FSR snapshot cache: "
        f"{SNAPSHOT_CACHE}"
    )

    return snapshots


def fixed_neutral_phase_exposure_v1_4() -> dict[str, float]:
    """Return the frozen deterministic V1.4 neutral phase reference."""

    return dict(V1_4_NEUTRAL_EXPOSURE)


def install_speedups() -> None:
    """Patch benchmark preparation only; simulator behavior is unchanged."""

    replay.build_locked_prefight_snapshots = cached_locked_prefight_snapshots

    # V1.5 installation later assigns this function into V1.2's population
    # baseline adapter. Replacing it here avoids another 5,000-path reference
    # replay while preserving the exact previously audited denominator.
    replay.v1_4.neutral_phase_exposure_v1_4 = (
        fixed_neutral_phase_exposure_v1_4
    )


def main() -> None:
    install_speedups()
    replay.main()


if __name__ == "__main__":
    main()
