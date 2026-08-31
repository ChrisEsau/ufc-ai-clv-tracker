"""Externalized FSR age-modifier loader/evaluator for the shadow simulator.

Stored FSR rows remain immutable. This module reads trait-specific age rules from
``config/fsr_age_modifiers.yaml`` and produces fight-night effective FSR copies.

No trait-specific age constants belong in simulator code. Only YAML entries with
both ``enabled: true`` and ``calibrated: true`` are applied.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_CONFIG_PATH = Path("config/fsr_age_modifiers.yaml")
SUPPORTED_MODELS = {"polynomial", "power_residual_polynomial"}


def _finite_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


@lru_cache(maxsize=8)
def load_age_modifier_config(path: str = str(DEFAULT_CONFIG_PATH)) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"FSR age modifier config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("FSR age modifier config must be a mapping")
    if int(cfg.get("version", 0)) != 1:
        raise ValueError(f"unsupported FSR age modifier config version: {cfg.get('version')!r}")
    if not isinstance(cfg.get("traits"), dict):
        raise ValueError("FSR age modifier config missing traits mapping")
    bounds = cfg.get("rating_bounds", {})
    low = _finite_float(bounds.get("min", 10.0), name="rating_bounds.min")
    high = _finite_float(bounds.get("max", 90.0), name="rating_bounds.max")
    if low >= high:
        raise ValueError("rating_bounds.min must be less than rating_bounds.max")
    return cfg


def _polynomial_modifier(rule: dict[str, Any], age: float, defaults: dict[str, Any]) -> float:
    center = _finite_float(rule.get("age_center", defaults.get("age_center", 30.0)), name="age_center")
    min_age = rule.get("min_age")
    max_age = rule.get("max_age")
    if min_age is not None and age < _finite_float(min_age, name="min_age"):
        return 0.0
    if max_age is not None and age > _finite_float(max_age, name="max_age"):
        age = _finite_float(max_age, name="max_age")

    coeff = rule.get("coefficients")
    if not isinstance(coeff, dict):
        raise ValueError("calibrated age rule missing coefficients mapping")
    intercept = _finite_float(coeff.get("intercept", 0.0), name="coefficients.intercept")
    linear = _finite_float(coeff.get("linear", 0.0), name="coefficients.linear")
    quadratic = _finite_float(coeff.get("quadratic", 0.0), name="coefficients.quadratic")
    cubic = _finite_float(coeff.get("cubic", 0.0), name="coefficients.cubic")
    x = float(age) - center
    raw = intercept + linear * x + quadratic * x * x + cubic * x * x * x

    minimum = _finite_float(
        rule.get("min_adjustment", defaults.get("min_adjustment", -40.0)),
        name="min_adjustment",
    )
    maximum = _finite_float(
        rule.get("max_adjustment", defaults.get("max_adjustment", 40.0)),
        name="max_adjustment",
    )
    if minimum > maximum:
        raise ValueError("min_adjustment cannot exceed max_adjustment")
    return float(np.clip(raw, minimum, maximum))


def trait_age_modifier(
    trait: str,
    age: float | None,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> float:
    """Return one configured FSR-point age modifier for a trait.

    Disabled or uncalibrated rules intentionally return zero.
    """
    if age is None or pd.isna(age):
        return 0.0
    age_value = _finite_float(age, name="fighter age")
    if age_value < 0.0:
        raise ValueError(f"fighter age must be non-negative, got {age_value}")

    cfg = load_age_modifier_config(str(config_path))
    rule = cfg["traits"].get(str(trait))
    if not isinstance(rule, dict):
        return 0.0
    if not bool(rule.get("enabled", False)) or not bool(rule.get("calibrated", False)):
        return 0.0

    model = str(rule.get("model", "polynomial"))
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported age modifier model for {trait}: {model}")
    defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {}
    return _polynomial_modifier(rule, age_value, defaults)


def apply_age_modifiers(
    profile: pd.Series,
    age: float | None,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[pd.Series, dict[str, float]]:
    """Return an effective profile and audit map without mutating stored FSR."""
    effective = profile.copy(deep=True)
    if age is None or pd.isna(age):
        return effective, {}

    cfg = load_age_modifier_config(str(config_path))
    bounds = cfg.get("rating_bounds", {})
    low = _finite_float(bounds.get("min", 10.0), name="rating_bounds.min")
    high = _finite_float(bounds.get("max", 90.0), name="rating_bounds.max")

    applied: dict[str, float] = {}
    for trait, rule in cfg["traits"].items():
        if not isinstance(rule, dict):
            continue
        if not bool(rule.get("enabled", False)) or not bool(rule.get("calibrated", False)):
            continue
        if trait not in effective.index:
            raise ValueError(f"profile missing enabled age-adjusted trait: {trait}")
        stored = _finite_float(effective[trait], name=f"profile[{trait}]")
        modifier = trait_age_modifier(trait, float(age), config_path=config_path)
        effective[trait] = float(np.clip(stored + modifier, low, high))
        applied[trait] = float(modifier)
    return effective, applied


def enabled_calibrated_traits(*, config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[str, ...]:
    cfg = load_age_modifier_config(str(config_path))
    return tuple(
        str(trait)
        for trait, rule in cfg["traits"].items()
        if isinstance(rule, dict)
        and bool(rule.get("enabled", False))
        and bool(rule.get("calibrated", False))
    )
