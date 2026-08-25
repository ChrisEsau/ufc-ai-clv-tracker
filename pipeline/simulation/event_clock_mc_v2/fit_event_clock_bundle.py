"""Fit FSR V3 direct inference while preserving frozen Event Clock V1 mechanics.

The parent V1 context remains the source of every calibrated Stage-9, control,
submission, judge, stamina, KD, and KO/TKO mechanic.  ECV2 replaces only inputs
that have independently passed FSR V3 validation:

1. direct flow features use canonical FSR V3 state;
2. the frozen detailed-path profile copy receives V3 striking power only.

All other physical/stamina/submission profile fields remain inherited from the
parent V1 bundle.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.fit_event_clock_bundle import (
    DEFAULT_BUNDLE_PATH as V1_BUNDLE_PATH,
)
from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import ShadowKOKDCalibration
from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import build_historical_targets
from pipeline.simulation.event_clock_mc_v1.prototype_stage2 import CUTOFF, TRAIN_MAX_FIGHTS
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_feature_rows_v3
from pipeline.simulation.event_clock_mc_v2.inference import fit_inference_models_v3
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import (
    legacy_power_equivalent,
)

DEFAULT_BUNDLE_PATH = Path("data/models/event_clock_mc_v2/event_clock_v2_fsr_v3_bundle.joblib")
DEFAULT_MANIFEST_PATH = Path("data/models/event_clock_mc_v2/event_clock_v2_fsr_v3_bundle_manifest.json")
POWER_NATIVE_COLUMN = "striking_power_v3"


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_v1_parent(path):
    if not path.exists():
        raise RuntimeError(f"Frozen Event Clock V1 bundle not found: {path}")
    payload = joblib.load(path)
    if payload.get("schema_version") != 2:
        raise RuntimeError(f"Expected V1 bundle schema 2, got {payload.get('schema_version')!r}")
    return payload


def overlay_v3_power_on_frozen_profiles(
    frozen_profiles: pd.DataFrame,
    fsr_v3: pd.DataFrame,
) -> pd.DataFrame:
    """Replace only striking power in the detailed-mechanics profile snapshot."""
    required = {"fight_id", "fighter_id", POWER_NATIVE_COLUMN}
    missing = required.difference(fsr_v3.columns)
    if missing:
        raise RuntimeError(f"canonical FSR V3 missing power columns: {sorted(missing)}")

    base = frozen_profiles.copy()
    if "striking_power" not in base.columns:
        raise RuntimeError("parent V1 mechanics profiles are missing striking_power")
    base["fight_id"] = base["fight_id"].astype(str)
    base["fighter_id"] = base["fighter_id"].astype(str)

    power = fsr_v3[["fight_id", "fighter_id", POWER_NATIVE_COLUMN]].copy()
    power["fight_id"] = power["fight_id"].astype(str)
    power["fighter_id"] = power["fighter_id"].astype(str)
    if power.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("duplicate canonical V3 power rows")

    merged = base.merge(
        power,
        on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )
    if merged[POWER_NATIVE_COLUMN].isna().any():
        count = int(merged[POWER_NATIVE_COLUMN].isna().sum())
        raise RuntimeError(f"V3 power overlay missing {count} frozen mechanics profile rows")

    merged["striking_power"] = legacy_power_equivalent(
        merged[POWER_NATIVE_COLUMN].to_numpy(float)
    )
    return merged.drop(columns=[POWER_NATIVE_COLUMN])


def build_training_master_v3():
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
    master = master.sort_values(["event_date", "fight_id"]).tail(TRAIN_MAX_FIGHTS).reset_index(drop=True)
    return master, fsr


def build_v3_training_frame():
    master, fsr = build_training_master_v3()
    features = build_feature_rows_v3(master, fsr, scheduled_duration=False)
    targets = build_historical_targets()
    targets["fight_id"] = targets["fight_id"].astype(str)
    train = features.merge(
        targets, on=["fight_id", "fighter_name"], how="inner", validate="one_to_one"
    )
    if train["fight_id"].nunique() != len(master):
        missing = sorted(set(master["fight_id"]) - set(train["fight_id"]))
        raise RuntimeError(f"V3 direct training lost {len(missing)} fights: {missing[:10]}")
    return train, master, fsr


def main():
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
    print("detailed-profile change: V3 striking power ONLY")

    parent = load_v1_parent(args.v1_bundle)
    context = deepcopy(parent["context"])
    if "fsr_all" not in context:
        raise RuntimeError("Parent V1 bundle is missing frozen mechanics FSR snapshots")

    train, train_master, fsr = build_v3_training_frame()
    print(f"V3 direct-model training fights: {train['fight_id'].nunique():,}")
    print(f"V3 direct-model fighter-fight rows: {len(train):,}")
    context["inference_models"] = fit_inference_models_v3(train)

    # Frozen fight mechanics remain unchanged.  Only the newly validated power
    # input is translated into the coordinate expected by the frozen KD/KO
    # hazard; all other detailed-path profile fields stay from the V1 parent.
    context["fsr_all"] = overlay_v3_power_on_frozen_profiles(context["fsr_all"], fsr)

    calibration = ShadowKOKDCalibration()
    parent_meta = dict(context.get("bundle_metadata", {}))
    metadata = {
        **parent_meta,
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "parent_v1_bundle": str(args.v1_bundle),
        "parent_v1_git_sha": parent_meta.get("git_sha", "unknown"),
        "fsr_version": "v3_direct_flow_plus_v3_power_with_v1_other_physics",
        "direct_inference_schema": "event_clock_mc_v2_fsr_v3_direct_v1",
        "direct_training_fights": int(train["fight_id"].nunique()),
        "direct_training_first_event_date": str(pd.Timestamp(train_master["event_date"].min()).date()),
        "direct_training_last_event_date": str(pd.Timestamp(train_master["event_date"].max()).date()),
        "mechanics_source": "frozen Event Clock V1 bundle; unchanged",
        "mechanics_profile_source": (
            "parent V1 fsr_all with only striking_power replaced from validated V3 latent"
        ),
        "power_native_column": POWER_NATIVE_COLUMN,
        "power_native_semantics": "attacker KD logit effect per landed significant strike",
        "power_epistemic_sampling": False,
        "power_translation": "50 + striking_power_v3 / frozen_kd_power_beta",
        "frozen_kd_power_beta": float(calibration.kd_power_beta),
        "implied_ko_beta_per_v3_latent": float(
            calibration.ko_power_beta / calibration.kd_power_beta
        ),
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
    print(
        "V3 power translation: latent -> frozen profile via "
        f"KD beta {metadata['frozen_kd_power_beta']:.6f}; "
        f"implied KO beta/latent={metadata['implied_ko_beta_per_v3_latent']:.4f}"
    )
    print("DONE — mechanics preserved; V3 direct inference and validated power input frozen.")


if __name__ == "__main__":
    main()
