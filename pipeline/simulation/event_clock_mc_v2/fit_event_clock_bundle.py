"""Fit the FSR V3 direct models and freeze them into Event Clock V2.

The existing Event Clock V1 frozen bundle is the mechanics/calibration parent.
V2 copies that context and replaces only:

* canonical FSR snapshots -> FSR V3
* direct inference models -> refit on FSR V3 semantic features

All Stage-9, control, stamina, judge, submission scalar/offset, and frozen KO/KD
objects/parameters remain those persisted in the V1 bundle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.fit_event_clock_bundle import (
    DEFAULT_BUNDLE_PATH as V1_BUNDLE_PATH,
)
from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import build_historical_targets
from pipeline.simulation.event_clock_mc_v1.prototype_stage2 import (
    CUTOFF,
    TRAIN_MAX_FIGHTS,
)
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_feature_rows_v3
from pipeline.simulation.event_clock_mc_v2.inference import fit_inference_models_v3


DEFAULT_BUNDLE_PATH = Path(
    "data/models/event_clock_mc_v2/event_clock_v2_fsr_v3_bundle.joblib"
)
DEFAULT_MANIFEST_PATH = Path(
    "data/models/event_clock_mc_v2/event_clock_v2_fsr_v3_bundle_manifest.json"
)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_v1_parent(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(
            f"Frozen Event Clock V1 bundle not found: {path}\n"
            "Build the frozen V1 bundle first if it is absent."
        )
    payload = joblib.load(path)
    if payload.get("schema_version") != 2:
        raise RuntimeError(
            f"Expected frozen V1 bundle schema 2, got {payload.get('schema_version')!r}"
        )
    return payload


def build_training_master_v3() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match the frozen V1 3000-fight training selection using V3 availability."""
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["event_date"] = pd.to_datetime(master["date"], errors="raise")
    master["fight_id"] = master["fight_id"].astype(str)

    fsr = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["event_date"] = pd.to_datetime(fsr["event_date"], errors="raise").dt.normalize()
    valid = fsr.groupby("fight_id").size()
    valid_ids = set(valid[valid == 2].index)

    master = master[
        (master["event_date"] < CUTOFF)
        & master["fight_id"].isin(valid_ids)
        & master["total_rounds"].isin([3, 5])
        & master["match_time_sec"].notna()
    ].copy()
    master = (
        master.sort_values(["event_date", "fight_id"])
        .tail(TRAIN_MAX_FIGHTS)
        .reset_index(drop=True)
    )
    return master, fsr


def build_v3_training_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master, fsr = build_training_master_v3()
    features = build_feature_rows_v3(master, fsr, scheduled_duration=False)
    targets = build_historical_targets()
    targets["fight_id"] = targets["fight_id"].astype(str)
    train = features.merge(
        targets,
        on=["fight_id", "fighter_name"],
        how="inner",
        validate="one_to_one",
    )
    if train["fight_id"].nunique() != len(master):
        missing = sorted(set(master["fight_id"]) - set(train["fight_id"]))
        raise RuntimeError(
            f"V3 direct training lost {len(missing)} fights after target merge: {missing[:10]}"
        )
    return train, master, fsr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-bundle", type=Path, default=V1_BUNDLE_PATH)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()

    print("=" * 120)
    print("EVENT CLOCK MC V2 — FIT FSR V3 DIRECT INFERENCE BUNDLE")
    print("=" * 120)
    print(f"parent V1 bundle: {args.v1_bundle}")
    print("mechanics/calibration refit: NO")
    print("direct V3 inference refit: YES")

    parent = load_v1_parent(args.v1_bundle)
    context = deepcopy(parent["context"])

    train, train_master, fsr = build_v3_training_frame()
    print(f"V3 direct-model training fights: {train['fight_id'].nunique():,}")
    print(f"V3 direct-model fighter-fight rows: {len(train):,}")
    context["inference_models"] = fit_inference_models_v3(train)
    context["fsr_all"] = fsr

    parent_meta = dict(context.get("bundle_metadata", {}))
    metadata = {
        **parent_meta,
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "parent_v1_bundle": str(args.v1_bundle),
        "parent_v1_git_sha": parent_meta.get("git_sha", "unknown"),
        "fsr_version": "v3",
        "direct_inference_schema": "event_clock_mc_v2_fsr_v3_direct_v1",
        "direct_training_fights": int(train["fight_id"].nunique()),
        "direct_training_first_event_date": str(pd.Timestamp(train_master["event_date"].min()).date()),
        "direct_training_last_event_date": str(pd.Timestamp(train_master["event_date"].max()).date()),
        "target_scope": "eligible historical fights with canonical FSR V3 prefight snapshots",
        "mechanics_source": "frozen Event Clock V1 bundle; unchanged",
        "epistemic_modes": ["means_only", "validated_path_sampling"],
        "epistemic_positive_projection": "moment_matched_gamma",
    }
    context["bundle_metadata"] = metadata

    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 3, "context": context}, args.bundle, compress=3)
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"bundle:   {args.bundle}")
    print(f"manifest: {args.manifest}")
    print(f"parent mechanics git SHA: {metadata['parent_v1_git_sha']}")
    print(f"V2 git SHA: {metadata['git_sha']}")
    print("DONE — mechanics preserved; V3 direct inference frozen.")


if __name__ == "__main__":
    main()
