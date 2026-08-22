"""Validated reach matchup translation for Event Clock MC V2.

Reach is not an FSR trait.  The development-only Reach Mechanics V1 study found
stable incremental signal only for distance-attempt volume after controlling
for FSR V3 expectation, division, age, and height.  The robustness pass retained
that effect under a +/-6 inch clipped reach edge.

This module therefore exposes one narrow physical matchup translation:

    distance attempt multiplier = exp(beta * clipped_reach_edge_inches)

No accuracy, takedown, ground, damage, or finish mechanic is changed here.
"""
from __future__ import annotations

from math import exp
from typing import Mapping

import numpy as np

# Mean chronological coefficient from the validated +/-6 inch clipped
# robustness scenario.  See data/research/reach_mechanics_robustness_v1 on the
# isolated research branch.
DISTANCE_REACH_LOG_RATE_PER_INCH = 0.002494322594196729
DISTANCE_REACH_EDGE_CAP_INCHES = 6.0


def _measure_inches(value: object) -> float:
    """Parse master physical measurements into inches when possible."""
    if value is None:
        return np.nan
    try:
        if bool(np.isnan(value)):
            return np.nan
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.number)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else np.nan

    text = str(value).strip().lower()
    if not text or text in {"--", "nan", "none"}:
        return np.nan
    if "'" in text:
        cleaned = text.replace('"', "").replace("in", "")
        feet_text, _, inches_text = cleaned.partition("'")
        try:
            return 12.0 * float(feet_text.strip()) + float(inches_text.strip() or 0.0)
        except ValueError:
            return np.nan

    numeric = "".join(ch if ch.isdigit() or ch in ".-" else " " for ch in text)
    parts = [part for part in numeric.split() if part]
    if not parts:
        return np.nan
    try:
        parsed = float(parts[0])
    except ValueError:
        return np.nan
    return parsed if np.isfinite(parsed) else np.nan


def distance_reach_multiplier(reach_edge_inches: float) -> tuple[float, float]:
    """Return (clipped edge, multiplier), neutral when reach is unavailable."""
    try:
        edge = float(reach_edge_inches)
    except (TypeError, ValueError):
        return np.nan, 1.0
    if not np.isfinite(edge):
        return np.nan, 1.0
    clipped = float(
        np.clip(
            edge,
            -DISTANCE_REACH_EDGE_CAP_INCHES,
            DISTANCE_REACH_EDGE_CAP_INCHES,
        )
    )
    multiplier = float(exp(DISTANCE_REACH_LOG_RATE_PER_INCH * clipped))
    return clipped, multiplier


def directional_reach_inputs(master_row: Mapping[str, object], side: str) -> dict[str, float]:
    """Build directional reach diagnostics/translation from one master fight row."""
    side = str(side).lower()
    if side == "red":
        self_value = master_row.get("r_reach")
        opp_value = master_row.get("b_reach")
    elif side == "blue":
        self_value = master_row.get("b_reach")
        opp_value = master_row.get("r_reach")
    else:
        raise ValueError(f"unknown fight side for reach translation: {side!r}")

    self_reach = _measure_inches(self_value)
    opp_reach = _measure_inches(opp_value)
    if np.isfinite(self_reach) and np.isfinite(opp_reach):
        raw_edge = float(self_reach - opp_reach)
    else:
        raw_edge = np.nan
    clipped_edge, multiplier = distance_reach_multiplier(raw_edge)
    return {
        "self_reach_inches": self_reach,
        "opp_reach_inches": opp_reach,
        "reach_edge_inches": raw_edge,
        "reach_edge_capped_inches": clipped_edge,
        "distance_reach_multiplier": multiplier,
    }
