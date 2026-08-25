"""Stable experiment identity and compact JSON ledger records."""

from __future__ import annotations
import hashlib, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
from .config import canonical_json, config_hash


def stable_experiment_id(identity: dict) -> str:
    return "ecv2-" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:20]


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def artifact_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def metrics_fingerprint(metrics: dict, invariants: dict) -> str:
    """Digest deterministic measurements, excluding runtime/experiment metadata."""
    return config_hash({"simulator_metrics": metrics, "invariants": invariants})


def build_record(
    *, identity: dict, config: dict, metrics: dict, invariants: dict, comparison=None
) -> dict:
    acceptance = metrics.get("acceptance_results", {})
    acceptance_states = {item.get("status") for item in acceptance.values()}
    run_status = (
        "FAIL"
        if invariants["status"] == "FAIL" or "FAIL" in acceptance_states
        else "WARN" if "WARN" in acceptance_states else "PASS"
    )
    return {
        "schema_version": "event_clock_v2_experiment_ledger_v1",
        "experiment_id": stable_experiment_id(identity),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        **identity,
        "parameter_config": config,
        "parameter_config_hash": config_hash(config),
        "simulator_metrics": metrics,
        "invariant_results": invariants,
        "comparison_to_baseline": comparison or {},
        "run_status": run_status,
        "actions_run_id": os.getenv("GITHUB_RUN_ID"),
        "job_id": os.getenv("GITHUB_JOB"),
        "artifact_id": None,
        "artifact_digest": None,
    }


def write_record(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
