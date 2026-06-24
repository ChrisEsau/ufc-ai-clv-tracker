from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_IMPLIED_BUCKETS = [
    {"name": "0_10", "min": 0.00, "max": 0.10},
    {"name": "10_20", "min": 0.10, "max": 0.20},
    {"name": "20_30", "min": 0.20, "max": 0.30},
    {"name": "30_40", "min": 0.30, "max": 0.40},
    {"name": "40_50", "min": 0.40, "max": 0.50},
    {"name": "50_60", "min": 0.50, "max": 0.60},
    {"name": "60_70", "min": 0.60, "max": 0.70},
    {"name": "70_80", "min": 0.70, "max": 0.80},
    {"name": "80_90", "min": 0.80, "max": 0.90},
    {"name": "90_100", "min": 0.90, "max": 1.00},
]


@dataclass(frozen=True)
class ImpliedBucketCalibrationResult:
    rows: pd.DataFrame
    report: pd.DataFrame
    config: dict[str, Any]


def apply_implied_bucket_delta_calibration(joined: pd.DataFrame, *, config: dict[str, Any]) -> ImpliedBucketCalibrationResult:
    return ImpliedBucketCalibrationResult(rows=joined.copy(), report=pd.DataFrame(), config=(config.get("post_market_calibration") or {}))
