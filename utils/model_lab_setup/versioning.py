from __future__ import annotations

import re


_VERSION_PATTERN = re.compile(r"_v(\d+)$")


def safe_model_id(value: str) -> str:
    """Return a normalized model id: lowercase, alphanumeric/underscore only."""

    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def safe_path_key(value: str) -> str:
    """Return a safe path segment, defaulting to unknown_market when empty."""

    return safe_model_id(value) or "unknown_market"


def artifact_dir_for_market(model_id: str, market_key: str) -> str:
    """Return the canonical artifact directory for a model/market pair."""

    return f"models/{safe_path_key(market_key)}/{safe_model_id(model_id)}"


def parse_model_version(model_id: str) -> dict:
    """Parse a trailing _vN model version suffix."""

    safe_id = safe_model_id(model_id)
    match = _VERSION_PATTERN.search(safe_id)
    return {
        "base": _VERSION_PATTERN.sub("", safe_id),
        "version": int(match.group(1)) if match else None,
    }


def next_model_version(base_model_id: str, existing_model_ids: list[str]) -> str:
    """Return the next available version id for a base model id."""

    parsed = parse_model_version(base_model_id)
    base = parsed["base"] or safe_model_id(base_model_id)
    versions: list[int] = []
    for model_id in existing_model_ids:
        candidate = parse_model_version(model_id)
        if candidate["base"] == base and candidate["version"] is not None:
            versions.append(int(candidate["version"]))
    return f"{base}_v{(max(versions) if versions else 0) + 1}"


def generate_new_model_id(
    template_model_id: str,
    market_key: str,
    existing_model_ids: list[str],
) -> str:
    """Generate the next safe draft model id from a template and market."""

    template = parse_model_version(template_model_id)
    base = template["base"] or safe_model_id(template_model_id)
    safe_market = safe_path_key(market_key)
    if safe_market and safe_market != "moneyline" and safe_market not in base:
        base = f"{safe_market}_{base}"
    return next_model_version(base, existing_model_ids)
