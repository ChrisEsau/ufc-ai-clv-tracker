"""Reusable historical/runtime input bundle for the locked Brain MC harness.

The bundle materializes expensive historical state and the minimal frozen simulator
runtime context once. Runtime fight simulations only perform target-fight/date
lookups against these immutable files; they do not download or rebuild legacy
Event Clock bundles.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.research import fsr_recency_cohort_shadow as recency
from pipeline.research import ko_time_survival_oos as ko_surv
from pipeline.research import sub_time_survival_oos as sub_surv
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as LEGACY_RUNTIME_BUNDLE_PATH

BUNDLE_SCHEMA_VERSION = 2
DEFAULT_BUNDLE_DIR = Path("data/research/locked_brain_bundle")
FILES = {
    "canonical_fsr": "canonical_fsr_v3_prefight_snapshots.parquet",
    "ewm_fsr": "pure_ewm05_prefight_snapshots.parquet",
    "ko_prefight": "ko_survival_prefight.parquet",
    "ko_baselines": "ko_survival_date_baselines.parquet",
    "sub_prefight": "sub_survival_prefight.parquet",
    "sub_baselines": "sub_survival_date_baselines.parquet",
    "runtime_context": "minimal_runtime_context.joblib",
}
RUNTIME_CONTEXT_KEYS = ("conversion_offset", "judge_model", "fsr_all")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _date_baselines(ff: pd.DataFrame, module, event_col: str) -> pd.DataFrame:
    """Exactly reproduce train_baselines(ff[event_date < cutoff]) for every cutoff date."""
    x = ff.copy()
    x["event_date"] = pd.to_datetime(x["event_date"]).dt.normalize()
    t = x["fight_seconds"].to_numpy(float)
    exposure = module.interval_exposure(t)
    event_idx = module.interval_event_index(t)
    event = x[event_col].to_numpy(float)

    parts = pd.DataFrame({"event_date": x["event_date"], "seconds": t, "events": event})
    for i in range(exposure.shape[1]):
        parts[f"exposure_{i}"] = exposure[:, i]
        parts[f"interval_events_{i}"] = event * (event_idx == i)
    daily = parts.groupby("event_date", as_index=False).sum(numeric_only=True).sort_values("event_date")
    numeric = [c for c in daily.columns if c != "event_date"]
    prior = daily[numeric].cumsum().shift(1).fillna(0.0)

    rows = []
    for idx, date in enumerate(daily["event_date"]):
        p = prior.iloc[idx]
        sec = float(p["seconds"])
        ev = float(p["events"])
        if sec <= 0 or ev <= 0:
            continue
        p0 = ev / sec
        prior_e = 2.0
        prior_x = prior_e / p0
        piece = []
        for i in range(exposure.shape[1]):
            piece.append((float(p[f"interval_events_{i}"]) + prior_e) / (float(p[f"exposure_{i}"]) + prior_x))
        row = {"event_date": pd.Timestamp(date), "population_hazard_per_second": p0}
        row.update({f"baseline_{i}": float(v) for i, v in enumerate(piece)})
        rows.append(row)
    return pd.DataFrame(rows)


def _build_minimal_runtime_context(bundle_dir: Path) -> dict:
    source = Path(LEGACY_RUNTIME_BUNDLE_PATH)
    if not source.is_file():
        raise FileNotFoundError(f"frozen simulator runtime bundle missing during one-time bundle build: {source}")
    payload = joblib.load(source)
    context = payload.get("context", {})
    missing = [key for key in RUNTIME_CONTEXT_KEYS if key not in context]
    if missing:
        raise RuntimeError(f"legacy runtime bundle missing required locked Brain context keys: {missing}")
    minimal = {key: context[key] for key in RUNTIME_CONTEXT_KEYS}
    target = bundle_dir / FILES["runtime_context"]
    joblib.dump({"schema_version": 1, "context": minimal}, target, compress=3)
    return {
        "source_schema_version": payload.get("schema_version"),
        "context_keys": list(RUNTIME_CONTEXT_KEYS),
        "rows_fsr_all": int(len(minimal["fsr_all"])),
    }


def build_bundle(bundle_dir: Path | str = DEFAULT_BUNDLE_DIR) -> dict:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    if not snapshot.is_file():
        raise FileNotFoundError(f"canonical FSR snapshot missing: {snapshot}")

    canonical = pd.read_parquet(snapshot).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)
    shutil.copy2(snapshot, bundle_dir / FILES["canonical_fsr"])

    old_decay, old_blend = recency.EWM_DECAY, recency.EWM_CANONICAL_BLEND
    try:
        recency.EWM_DECAY = 0.50
        recency.EWM_CANONICAL_BLEND = 0.0
        ewm = recency.build_variant(canonical, "ewm")
    finally:
        recency.EWM_DECAY, recency.EWM_CANONICAL_BLEND = old_decay, old_blend
    ewm.to_parquet(bundle_dir / FILES["ewm_fsr"], index=False)

    ko_ff = ko_surv.add_prefight(ko_surv.load_fighter_fights())
    ko_ff["event_date"] = pd.to_datetime(ko_ff["event_date"]).dt.normalize()
    ko_ff["fight_id"] = ko_ff["fight_id"].astype(str)
    ko_ff.to_parquet(bundle_dir / FILES["ko_prefight"], index=False)
    _date_baselines(ko_ff, ko_surv, "ko_event").to_parquet(bundle_dir / FILES["ko_baselines"], index=False)

    sub_ff = sub_surv.add_prefight(sub_surv.load_fighter_fights())
    sub_ff["event_date"] = pd.to_datetime(sub_ff["event_date"]).dt.normalize()
    sub_ff["fight_id"] = sub_ff["fight_id"].astype(str)
    sub_ff.to_parquet(bundle_dir / FILES["sub_prefight"], index=False)
    _date_baselines(sub_ff, sub_surv, "sub_event").to_parquet(bundle_dir / FILES["sub_baselines"], index=False)

    runtime = _build_minimal_runtime_context(bundle_dir)
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "architecture": "all reusable historical databases plus minimal frozen simulator runtime context materialized once; normal locked Brain runs require this bundle only",
        "ewm_decay": 0.50,
        "ewm_canonical_blend": 0.0,
        "ko_prior_events": 2.0,
        "sub_prior_events": 1.0,
        "minimal_runtime_context": runtime,
        "files": {key: {"name": name, "sha256": _sha256(bundle_dir / name)} for key, name in FILES.items()},
        "row_counts": {
            "ewm_fsr": int(len(ewm)),
            "ko_prefight": int(len(ko_ff)),
            "sub_prefight": int(len(sub_ff)),
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _verify_bundle(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"locked Brain bundle missing: {manifest_path}; run with --build-bundle first")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("schema_version", -1)) != BUNDLE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported locked Brain bundle schema: {manifest.get('schema_version')}")
    for key, spec in manifest["files"].items():
        path = bundle_dir / spec["name"]
        if not path.is_file():
            raise FileNotFoundError(f"bundle file missing: {path}")
        actual = _sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"bundle checksum mismatch for {key}: expected {spec['sha256']} got {actual}")
    return manifest


def _clock_lookup(ff: pd.DataFrame, baselines: pd.DataFrame, fight_id: str, *, prior_events: float, event_prefix: str):
    target = ff[ff["fight_id"].astype(str).eq(str(fight_id))].copy()
    if len(target) != 2:
        raise RuntimeError(f"expected two {event_prefix} target fighter rows for {fight_id}, got {len(target)}")
    cutoff = pd.Timestamp(target["event_date"].iloc[0]).normalize()
    base = baselines[pd.to_datetime(baselines["event_date"]).dt.normalize().eq(cutoff)]
    if len(base) != 1:
        raise RuntimeError(f"expected one {event_prefix} baseline row for {cutoff.date()}, got {len(base)}")
    b = base.iloc[0]
    p0 = float(b["population_hazard_per_second"])
    piece = np.asarray([b[f"baseline_{i}"] for i in range(5)], float)
    prior_sec = prior_events / p0
    by_name = {}
    for r in target.itertuples(index=False):
        prior_win = float(getattr(r, f"prior_{event_prefix}_win"))
        opp_prior_loss = float(getattr(r, f"opp_prior_{event_prefix}_loss"))
        prior_seconds = float(r.prior_seconds)
        opp_seconds = float(r.opp_prior_seconds)
        att_rate = (prior_win + prior_events) / (prior_seconds + prior_sec)
        def_rate = (opp_prior_loss + prior_events) / (opp_seconds + prior_sec)
        rr = float(np.clip(att_rate * def_rate / (p0 * p0), 0.05, 20.0))
        label = "ko" if event_prefix == "ko" else "sub"
        by_name[str(r.fighter_name)] = {
            f"prior_{label}_wins": prior_win,
            "prior_seconds": prior_seconds,
            f"opponent_prior_{label}_losses": opp_prior_loss,
            "opponent_prior_seconds": opp_seconds,
            "attacker_rate_per_minute": float(att_rate * 60.0),
            "defender_vulnerability_per_minute": float(def_rate * 60.0),
            "rate_ratio": rr,
            "hazards_per_second": piece * rr,
        }
    return cutoff, p0, piece, by_name


def install_bundle_runtime(legacy_module, fight_id: str, bundle_dir: Path | str = DEFAULT_BUNDLE_DIR) -> dict:
    bundle_dir = Path(bundle_dir)
    manifest = _verify_bundle(bundle_dir)

    snapshot = Path(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_dir / FILES["canonical_fsr"], snapshot)

    runtime_payload = joblib.load(bundle_dir / FILES["runtime_context"])
    runtime_context = runtime_payload.get("context", {})
    missing = [key for key in RUNTIME_CONTEXT_KEYS if key not in runtime_context]
    if missing:
        raise RuntimeError(f"locked Brain bundle runtime context missing keys: {missing}")
    legacy_runtime = Path(LEGACY_RUNTIME_BUNDLE_PATH)
    legacy_runtime.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 3, "context": runtime_context}, legacy_runtime, compress=3)

    ewm = pd.read_parquet(bundle_dir / FILES["ewm_fsr"])
    ko_ff = pd.read_parquet(bundle_dir / FILES["ko_prefight"])
    ko_base = pd.read_parquet(bundle_dir / FILES["ko_baselines"])
    sub_ff = pd.read_parquet(bundle_dir / FILES["sub_prefight"])
    sub_base = pd.read_parquet(bundle_dir / FILES["sub_baselines"])

    legacy_module.recency.build_variant = lambda canonical, variant: ewm.copy()
    legacy_module.time_ko._time_clock_inputs = lambda: _clock_lookup(
        ko_ff, ko_base, fight_id, prior_events=float(legacy_module.time_ko.PRIOR_EVENTS), event_prefix="ko"
    )
    legacy_module.sub_time_clock.time_clock_inputs = lambda fid: _clock_lookup(
        sub_ff, sub_base, str(fid), prior_events=float(legacy_module.sub_time_clock.PRIOR_EVENTS), event_prefix="sub"
    )
    return manifest
