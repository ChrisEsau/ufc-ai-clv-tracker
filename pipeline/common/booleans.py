"""Boolean coercion helpers for artifact values.

Parquet, CSV, and JSON round-trips can surface booleans as native bools,
integers, floats, or strings.  Use this helper where dashboard gates or safety
checks depend on boolean artifact fields so strings like "False" are not
mistakenly treated as truthy by Python's ``bool()`` constructor.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n", ""}


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Return a safe boolean interpretation for common artifact values."""

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    value_type = type(value)
    if value_type.__module__.startswith("numpy") and value_type.__name__ in {
        "bool",
        "bool_",
    }:
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
        return default

    try:
        if math.isnan(value):
            return default
    except TypeError:
        pass

    if isinstance(value, numbers.Number):
        return bool(value)

    return default
