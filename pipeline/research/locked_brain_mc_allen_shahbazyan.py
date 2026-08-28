"""LOCKED research harness for the Allen-Shahbazyan Brain MC line.

THIS FILE IS THE ONLY APPROVED ENTRY POINT FOR THIS RESEARCH LINE.
Do not bypass it with ad-hoc Python or alternate runners. Any change to the
mechanics stack, recency construction, path count, standing scale, KO/KD model,
submission model, or dependency lock requires explicit user approval first.

Frozen condition:
- target: Brendan Allen vs Edmen Shahbazyan, fight id from the existing trace
- fighter state: PURE EWM 0.50 FSR shadow (EWM decay=.50, canonical blend=0.0)
- standing attempt scale: 0.25, research only
- current Brain grappling/submission timing stack
- piecewise time-based KO competing clock
- OOS-selected static prefight KD hazard, no within-fight KD escalation
- matched standard Brain seeds
- 500 paths
- exact judge-scored output dump
- canonical FSR V3 snapshot restored byte-for-byte in finally

The harness fails closed if locked source files or core engine/FSR directories
have drifted from the frozen base commit.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.research import fsr_recency_cohort_shadow as recency
from pipeline.research import allen_shahbazyan_new_timing_trace as timing
from pipeline.research import allen_shahbazyan_time_ko_clock_2000 as time_ko
from pipeline.research import allen_shahbazyan_time_ko_validated_kd_2000 as validated_kd
from pipeline.research import allen_shahbazyan_decision_scored_outputs_2000 as scored
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily

LOCKED_BASE_COMMIT = "6ba1dd2d1e82aa7dec643bfd6f1d56bdd61b4e92"
LOCKED_PATHS = 500
LOCKED_EWM_DECAY = 0.50
LOCKED_EWM_CANONICAL_BLEND = 0.0
LOCKED_STANDING_ATTEMPT_SCALE = 0.25
CANONICAL_ARTIFACT_ID = 9494902022
CANONICAL_SOURCE_RUN_ID = 32645607979
CANONICAL_ARTIFACT_DIGEST = "sha256:a6abd062322eaf0c4a47997f215d7fa82c01c7db2755089fad1a30420da7d639"
OUTDIR = Path("data/research/locked_brain_mc_allen_shahbazyan")

# Git blob IDs for the exact direct research seams used by this harness.
LOCKED_BLOBS = {
    "pipeline/research/fsr_recency_cohort_shadow.py": "f126d734d346e674d06d4ac711fc2d776d242e73",
    "pipeline/research/allen_shahbazyan_new_timing_trace.py": "d9ac0a4da67ec0d83222b91337f6b4f396f262e7",
    "pipeline/research/allen_shahbazyan_time_ko_clock_2000.py": "edc24423d5549e0e706e17ee14459ec187a52542",
    "pipeline/research/allen_shahbazyan_time_ko_validated_kd_2000.py": "a333689887bb9a54cadfff2a9ac05a70ee844f64",
    "pipeline/research/allen_shahbazyan_decision_scored_outputs_2000.py": "6c23e1765b941c082c535588bf7a41b53fc6516d",
}

# Core source trees are not allowed to drift relative to LOCKED_BASE_COMMIT.
LOCKED_CORE_PATHS = (
    "pipeline/simulation/event_clock_mc_v2",
    "pipeline/fsr_v2",
    "pipeline/fsr_v3",
    "configs/event_clock_v2",
    "pipeline/research/ko_time_survival_oos.py",
    "pipeline/research/ko_v3_from_scratch_shadow.py",
    "pipeline/research/allen_shahbazyan_ground_opportunity_submission_trace.py",
    "pipeline/research/allen_shahbazyan_fighter_level_submission_trace.py",
    "pipeline/research/allen_shahbazyan_one_path_brain_trace_v1.py",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_locked_sources() -> dict:
    failures: list[str] = []
    blobs: dict[str, str] = {}
    for path, expected in LOCKED_BLOBS.items():
        actual = _git("hash-object", path)
        blobs[path] = actual
        if actual != expected:
            failures.append(f"blob drift: {path}: expected {expected}, got {actual}")

    drift = subprocess.run(
        ["git", "diff", "--quiet", LOCKED_BASE_COMMIT, "--", *LOCKED_CORE_PATHS],
        check=False,
    ).returncode
    if drift != 0:
        changed = _git("diff", "--name-only", LOCKED_BASE_COMMIT, "--", *LOCKED_CORE_PATHS)
        failures.append("core source drift from locked base commit:\n" + changed)

    if failures:
        raise RuntimeError("LOCKED BRAIN HARNESS REFUSED TO RUN\n" + "\n".join(failures))
    return blobs


def _scaled_standing_rates(original):
    def locked_scaled_rates(state, actor, capabilities, context, priors, config):
        rates, pressure = original(state, actor, capabilities, context, priors, config)
        rates = dict(rates)
        rates[ActionFamily.STAND_ATTACK] = (
            float(rates[ActionFamily.STAND_ATTACK]) * LOCKED_STANDING_ATTEMPT_SCALE
        )
        return rates, pressure
    return locked_scaled_rates


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    verified_blobs = assert_locked_sources()

    snapshot_path = Path(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"canonical FSR snapshot missing: {snapshot_path}")

    original_snapshot_sha256 = _sha256(snapshot_path)
    with tempfile.TemporaryDirectory(prefix="locked-brain-canonical-") as td:
        backup = Path(td) / snapshot_path.name
        shutil.copy2(snapshot_path, backup)

        original_decay = recency.EWM_DECAY
        original_blend = recency.EWM_CANONICAL_BLEND
        original_timing = timing._new_timing_rates
        original_time_paths = time_ko.PATHS
        original_scored_paths = scored.PATHS
        original_validated_out = validated_kd.OUTDIR

        try:
            canonical = pd.read_parquet(snapshot_path).copy()
            canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
            canonical["fight_id"] = canonical["fight_id"].astype(str)
            canonical["fighter_id"] = canonical["fighter_id"].astype(str)

            # Exact historical recency builder; PURE EWM means zero canonical blend.
            recency.EWM_DECAY = LOCKED_EWM_DECAY
            recency.EWM_CANONICAL_BLEND = LOCKED_EWM_CANONICAL_BLEND
            ewm = recency.build_variant(canonical, "ewm")
            ewm.to_parquet(snapshot_path, index=False)

            target_fight_id = str(validated_kd.base_trace.FIGHT_ID)
            target = ewm[ewm["fight_id"].eq(target_fight_id)].copy()
            if len(target) != 2:
                raise RuntimeError(f"expected 2 PURE EWM target rows, found {len(target)}")
            target.to_csv(OUTDIR / "pure_ewm05_target_fsr_rows.csv", index=False)

            # One and only one standing cadence intervention: calibrated research 0.25.
            timing._new_timing_rates = _scaled_standing_rates(original_timing)

            # Fixed matched-seed path count for this locked harness.
            time_ko.PATHS = LOCKED_PATHS
            scored.PATHS = LOCKED_PATHS
            validated_kd.OUTDIR = OUTDIR / "run"

            manifest = {
                "entry_point": "pipeline.research.locked_brain_mc_allen_shahbazyan",
                "locked_base_commit": LOCKED_BASE_COMMIT,
                "verified_blobs": verified_blobs,
                "fight_id": target_fight_id,
                "paths": LOCKED_PATHS,
                "ewm_decay": LOCKED_EWM_DECAY,
                "ewm_canonical_blend": LOCKED_EWM_CANONICAL_BLEND,
                "standing_attempt_scale": LOCKED_STANDING_ATTEMPT_SCALE,
                "ko": "piecewise time-based competing clock from allen_shahbazyan_time_ko_clock_2000",
                "kd": "OOS-selected static prefight KD hazard; no within-fight KD escalation",
                "submission": "current locked ground-opportunity/fighter-level submission research stack",
                "canonical_artifact_id": CANONICAL_ARTIFACT_ID,
                "canonical_source_run_id": CANONICAL_SOURCE_RUN_ID,
                "canonical_artifact_digest": CANONICAL_ARTIFACT_DIGEST,
                "canonical_snapshot_sha256_before": original_snapshot_sha256,
                "production_changed": False,
            }
            (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print("LOCKED_BRAIN_MC_MANIFEST")
            print(json.dumps(manifest, indent=2))

            # Exact existing validated-KD wrapper; it calls the locked time-KO and scored runners.
            validated_kd.main()
        finally:
            recency.EWM_DECAY = original_decay
            recency.EWM_CANONICAL_BLEND = original_blend
            timing._new_timing_rates = original_timing
            time_ko.PATHS = original_time_paths
            scored.PATHS = original_scored_paths
            validated_kd.OUTDIR = original_validated_out
            shutil.copy2(backup, snapshot_path)

    restored_sha256 = _sha256(snapshot_path)
    if restored_sha256 != original_snapshot_sha256:
        raise RuntimeError(
            "canonical FSR restore verification failed: "
            f"before={original_snapshot_sha256} after={restored_sha256}"
        )

    restore = {
        "canonical_snapshot_sha256_before": original_snapshot_sha256,
        "canonical_snapshot_sha256_after": restored_sha256,
        "byte_identical_restore": True,
    }
    (OUTDIR / "restore_verification.json").write_text(json.dumps(restore, indent=2) + "\n")
    print("CANONICAL_RESTORE_VERIFIED")
    print(json.dumps(restore, indent=2))


if __name__ == "__main__":
    main()
