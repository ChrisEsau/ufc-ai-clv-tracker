"""Strict, immutable, mechanics-only calibration overrides."""

from __future__ import annotations
from dataclasses import asdict, replace
import hashlib, json
from pathlib import Path
import yaml
from pipeline.simulation.event_clock_mc_v2.mechanics.config import (
    DEFAULT_MECHANICS_CALIBRATION_CONFIG,
    MechanicsCalibrationConfig,
)

ALLOWLIST = frozenset(
    {
        "impact_scale",
        "trauma_durability_divisor",
        "kd_slope",
        "kd_midpoint",
        "finish_slope",
        "finish_midpoint",
        "post_kd_finish_logit_bonus",
        "action_cost_scale",
        "top_position_cost_per_second",
        "bottom_position_cost_per_second",
        "round_recovery_fraction",
    }
)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(value) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def resolve_overrides(
    overrides: dict | None = None,
) -> tuple[MechanicsCalibrationConfig, dict]:
    explicit = dict(overrides or {})
    unknown = set(explicit) - ALLOWLIST
    if unknown:
        raise ValueError(f"unknown or frozen mechanics overrides: {sorted(unknown)}")
    resolved = (
        DEFAULT_MECHANICS_CALIBRATION_CONFIG
        if not explicit
        else replace(DEFAULT_MECHANICS_CALIBRATION_CONFIG, **explicit)
    )
    return resolved, explicit


def load_override_file(path: Path | None) -> tuple[MechanicsCalibrationConfig, dict]:
    if path is None:
        return resolve_overrides({})
    payload = yaml.safe_load(path.read_text()) or {}
    if set(payload) != {"mechanics"} or not isinstance(payload["mechanics"], dict):
        raise ValueError("override file must contain only a mechanics mapping")
    return resolve_overrides(payload["mechanics"])


def resolved_payload(config: MechanicsCalibrationConfig, explicit: dict) -> dict:
    return {"canonical_resolved": asdict(config), "explicit_overrides": explicit}
