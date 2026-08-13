"""Immutable, load-once EVENT MC calibration and partial override resolver."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("config/event_mc_v1.yaml")


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _merge(base, override):
    result = _plain(base)
    for key, value in override.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, Mapping) else value
    return result


@dataclass(frozen=True)
class EventMCCalibration:
    values: Mapping[str, object]
    source_path: str

    def section(self, name: str) -> Mapping[str, object]:
        return self.values[name]

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(_plain(self.values), sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EventMCConfigResolver:
    defaults: Mapping[str, object]
    weight_classes: Mapping[str, object]
    source_path: str

    def for_weight_class(self, key: str | None = None) -> EventMCCalibration:
        override = self.weight_classes.get(key, {}) if key else {}
        return EventMCCalibration(_freeze(_merge(self.defaults, override)), self.source_path)


def load_event_mc_config(path: Path = DEFAULT_CONFIG_PATH) -> EventMCConfigResolver:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document.get("defaults"), dict) or not isinstance(document.get("weight_classes", {}), dict):
        raise ValueError("EVENT MC config requires defaults and weight_classes mappings")
    return EventMCConfigResolver(_freeze(document["defaults"]), _freeze(document.get("weight_classes", {})), str(path))


DEFAULT_RESOLVER = load_event_mc_config()
DEFAULT_CALIBRATION = DEFAULT_RESOLVER.for_weight_class()
