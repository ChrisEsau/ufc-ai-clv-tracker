"""Provider boundary for calibrated simulator round parameters.

The provider deliberately returns an absolute exposure-adjusted strike pace and a
gamma-Poisson dispersion value. A trained-parameter simulator must sample from
this distribution directly. It must not reapply the heuristic engine's base pace,
regime pace multiplier, suppression multiplier, fatigue multiplier, or confidence
multiplier to the returned mean.

This first provider supports historical out-of-fold significant-strike attempt
parameters only. It is a replay/research boundary, not a complete live simulator
provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


class RoundParameterProviderError(ValueError):
    """Raised when provider inputs or requested parameters violate the contract."""


@dataclass(frozen=True)
class RoundParameterKey:
    fight_id: str
    fighter_id: str
    round: int

    def __post_init__(self) -> None:
        if not str(self.fight_id).strip():
            raise RoundParameterProviderError("fight_id must be non-empty")
        if not str(self.fighter_id).strip():
            raise RoundParameterProviderError("fighter_id must be non-empty")
        if int(self.round) <= 0:
            raise RoundParameterProviderError("round must be positive")


@dataclass(frozen=True)
class SignificantStrikeAttemptParameters:
    """Calibrated absolute strike-attempt distribution for one fighter-round."""

    key: RoundParameterKey
    mean_rate_per_minute: float
    gamma_poisson_overdispersion: float
    model_name: str
    model_version: str
    calibration_factor: float
    source: str = "historical_out_of_fold"

    def __post_init__(self) -> None:
        numeric = {
            "mean_rate_per_minute": self.mean_rate_per_minute,
            "gamma_poisson_overdispersion": self.gamma_poisson_overdispersion,
            "calibration_factor": self.calibration_factor,
        }
        for name, raw in numeric.items():
            value = float(raw)
            if not np.isfinite(value) or value <= 0:
                raise RoundParameterProviderError(
                    f"{name} must be finite and positive; received {raw!r}"
                )
        if not str(self.model_name).strip():
            raise RoundParameterProviderError("model_name must be non-empty")
        if not str(self.model_version).strip():
            raise RoundParameterProviderError("model_version must be non-empty")

    def expected_count(self, exposure_seconds: float) -> float:
        exposure = float(exposure_seconds)
        if not np.isfinite(exposure) or exposure <= 0:
            raise RoundParameterProviderError(
                f"exposure_seconds must be finite and positive; received {exposure_seconds!r}"
            )
        return float(self.mean_rate_per_minute * exposure / 60.0)

    def sample_count(
        self,
        rng: np.random.Generator,
        exposure_seconds: float,
    ) -> int:
        """Sample a gamma-Poisson count at the requested exposure."""
        mean = self.expected_count(exposure_seconds)
        alpha = float(self.gamma_poisson_overdispersion)
        shape = 1.0 / alpha
        scale = mean * alpha
        mixed_rate = float(rng.gamma(shape=shape, scale=scale))
        return int(rng.poisson(mixed_rate))


@runtime_checkable
class SignificantStrikeParameterProvider(Protocol):
    def significant_strike_attempts(
        self,
        key: RoundParameterKey,
    ) -> SignificantStrikeAttemptParameters:
        """Return one calibrated fighter-round strike distribution."""
        ...


class HistoricalSignificantStrikeProvider:
    """Lookup provider backed by calibrated out-of-fold replay predictions."""

    REQUIRED_COLUMNS = (
        "fight_id",
        "fighter_id",
        "round",
        "model_name",
        "calibration_factor",
        "gamma_poisson_overdispersion",
        "calibrated_rate_per_min",
    )

    def __init__(
        self,
        calibrated_predictions: pd.DataFrame,
        model_name: str,
        model_version: str = "sig_attempt_pace_v0",
    ) -> None:
        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in calibrated_predictions.columns
        ]
        if missing:
            raise RoundParameterProviderError(
                f"calibrated predictions are missing provider columns: {missing}"
            )

        frame = calibrated_predictions.loc[
            calibrated_predictions["model_name"].astype(str).eq(str(model_name))
        ].copy()
        if frame.empty:
            raise RoundParameterProviderError(
                f"calibrated predictions contain no rows for model {model_name!r}"
            )

        frame["round"] = pd.to_numeric(frame["round"], errors="coerce")
        numeric_columns = (
            "calibration_factor",
            "gamma_poisson_overdispersion",
            "calibrated_rate_per_min",
        )
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[["round", *numeric_columns]].isna().any().any():
            raise RoundParameterProviderError(
                "calibrated predictions contain missing provider values"
            )
        if frame["round"].le(0).any():
            raise RoundParameterProviderError("provider round values must be positive")
        if frame[list(numeric_columns)].le(0).any().any():
            raise RoundParameterProviderError(
                "calibrated provider rates, factors, and dispersion must be positive"
            )

        key_columns = ["fight_id", "fighter_id", "round"]
        duplicate_count = int(frame.duplicated(key_columns).sum())
        if duplicate_count:
            raise RoundParameterProviderError(
                f"calibrated provider has duplicate fighter-round keys: {duplicate_count}"
            )

        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)
        frame["round"] = frame["round"].astype(int)
        self.model_name = str(model_name)
        self.model_version = str(model_version)
        self._rows = frame.set_index(key_columns)
        if not self._rows.index.is_unique:
            raise RoundParameterProviderError(
                "calibrated provider index is not unique after normalization"
            )

    def __len__(self) -> int:
        return int(len(self._rows))

    def significant_strike_attempts(
        self,
        key: RoundParameterKey,
    ) -> SignificantStrikeAttemptParameters:
        lookup_key = (str(key.fight_id), str(key.fighter_id), int(key.round))
        try:
            row = self._rows.loc[lookup_key]
        except KeyError as exc:
            raise RoundParameterProviderError(
                f"No historical strike parameters found for {lookup_key}"
            ) from exc
        if isinstance(row, pd.DataFrame):
            raise RoundParameterProviderError(
                f"Historical strike provider resolved multiple rows for {lookup_key}"
            )
        return SignificantStrikeAttemptParameters(
            key=key,
            mean_rate_per_minute=float(row["calibrated_rate_per_min"]),
            gamma_poisson_overdispersion=float(
                row["gamma_poisson_overdispersion"]
            ),
            model_name=self.model_name,
            model_version=self.model_version,
            calibration_factor=float(row["calibration_factor"]),
        )
