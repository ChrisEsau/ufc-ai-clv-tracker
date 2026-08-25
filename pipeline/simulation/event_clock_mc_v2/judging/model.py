"""Frozen Event2 total-fight judge equation used as a round transfer.

No trustworthy historical judge-by-round labels are present in the repository.
The source Stage-10D total-fight probability is therefore applied to one completed
round at a time.  This is explicitly a transfer assumption, not round calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Sequence

import numpy as np

FEATURES = ("sig_diff", "kd_diff", "td_diff", "sub_diff", "ctrl_diff")


@dataclass(frozen=True)
class JudgeFeatures:
    sig_diff: float = 0.0
    kd_diff: float = 0.0
    td_diff: float = 0.0
    sub_diff: float = 0.0
    ctrl_diff: float = 0.0

    def vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in FEATURES], dtype=float)


@dataclass(frozen=True)
class Event2JudgeModel:
    """StandardScaler + logistic coefficients frozen from Stage-10D."""

    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    training_decisions: int
    source: str = "EVENT2_TOTAL_JUDGE_ROUND_TRANSFER"

    def __post_init__(self) -> None:
        if not all(len(x) == len(FEATURES) for x in (self.scaler_mean, self.scaler_scale, self.coefficients)):
            raise ValueError("judge parameter dimensions must match FULL_TOTAL")
        if any(x <= 0 for x in self.scaler_scale):
            raise ValueError("judge scaler scales must be positive")

    def probability(self, features: JudgeFeatures) -> float:
        x = (features.vector() - np.asarray(self.scaler_mean)) / np.asarray(self.scaler_scale)
        z = self.intercept + float(np.dot(x, np.asarray(self.coefficients)))
        return 1.0 / (1.0 + exp(-float(np.clip(z, -40.0, 40.0))))

    @classmethod
    def from_sklearn(cls, pipeline, *, training_decisions: int) -> "Event2JudgeModel":
        scaler = pipeline.named_steps["scale"]
        logistic = pipeline.named_steps["logistic"]
        return cls(
            tuple(float(x) for x in scaler.mean_),
            tuple(float(x) for x in scaler.scale_),
            float(logistic.intercept_[0]),
            tuple(float(x) for x in logistic.coef_[0]),
            int(training_decisions),
        )


# Exact raw Stage-10D behavior was recovered from its committed 249-fight output.
# This algebraically equivalent scaler representation is the engine fallback; the
# historical runner replaces it with the exact frozen sklearn pipeline metadata.
EVENT2_TOTAL_JUDGE_ROUND_TRANSFER = Event2JudgeModel(
    scaler_mean=(0.0, 0.0, 0.0, 0.0, 0.0),
    scaler_scale=(1.0, 1.0, 1.0, 1.0, 1.0),
    intercept=0.1892022817508294,
    coefficients=(0.07090539434137932, 1.10664596634779, 0.1101752319151553,
                  0.21066496462540021, 0.00404707492226573),
    training_decisions=0,
)
