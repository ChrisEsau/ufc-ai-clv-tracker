"""Model-agnostic contracts for simulator parameter estimation.

This module does not train a specific algorithm. It defines the target registry,
prediction schema, and provider interfaces that future XGBoost, scikit-learn, or
other calibrated models must satisfy before their outputs can enter simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


class ParameterContractError(ValueError):
    """Raised when parameter model metadata or predictions violate the contract."""


class ParameterTask(str, Enum):
    """Supported statistical target families."""

    COUNT = "count"
    BINOMIAL = "binomial"
    BINARY = "binary"
    ZERO_INFLATED_CONTINUOUS = "zero_inflated_continuous"


class SimulationParameter(str, Enum):
    """Initial round-level parameters required by the simulator roadmap."""

    SIG_ATTEMPTS = "sig_attempts"
    SIG_ACCURACY = "sig_accuracy"
    TD_ATTEMPTS = "td_attempts"
    TD_ACCURACY = "td_accuracy"
    CONTROL_SECONDS = "control_seconds"
    KNOCKDOWNS = "knockdowns"
    KO_TKO_FINISH = "ko_tko_finish"
    SUBMISSION_FINISH = "submission_finish"


@dataclass(frozen=True)
class ParameterModelSpec:
    """Training and prediction contract for one simulator component model."""

    parameter: SimulationParameter
    task: ParameterTask
    target_column: str
    exposure_column: str | None = None
    prediction_mean_column: str = "prediction_mean"
    prediction_probability_column: str = "prediction_probability"
    prediction_dispersion_column: str = "prediction_dispersion"
    prediction_zero_probability_column: str = "prediction_zero_probability"
    minimum_training_rows: int = 250
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.target_column.startswith("target_"):
            raise ParameterContractError(
                f"target_column must use target_* namespace: {self.target_column!r}"
            )
        if self.exposure_column is not None and not self.exposure_column.startswith("target_"):
            raise ParameterContractError(
                f"exposure_column must use target_* namespace: {self.exposure_column!r}"
            )
        if self.minimum_training_rows <= 0:
            raise ParameterContractError("minimum_training_rows must be positive")
        if self.task == ParameterTask.BINOMIAL and self.exposure_column is None:
            raise ParameterContractError("binomial parameter specs require an exposure_column")


DEFAULT_PARAMETER_MODEL_SPECS: Mapping[SimulationParameter, ParameterModelSpec] = {
    SimulationParameter.SIG_ATTEMPTS: ParameterModelSpec(
        parameter=SimulationParameter.SIG_ATTEMPTS,
        task=ParameterTask.COUNT,
        target_column="target_sig_attempted",
        notes="Round significant-strike attempt count with overdispersion.",
    ),
    SimulationParameter.SIG_ACCURACY: ParameterModelSpec(
        parameter=SimulationParameter.SIG_ACCURACY,
        task=ParameterTask.BINOMIAL,
        target_column="target_sig_landed",
        exposure_column="target_sig_attempted",
        notes="Landed significant strikes conditional on attempts.",
    ),
    SimulationParameter.TD_ATTEMPTS: ParameterModelSpec(
        parameter=SimulationParameter.TD_ATTEMPTS,
        task=ParameterTask.COUNT,
        target_column="target_td_attempted",
        notes="Round takedown-attempt count with excess zeros.",
    ),
    SimulationParameter.TD_ACCURACY: ParameterModelSpec(
        parameter=SimulationParameter.TD_ACCURACY,
        task=ParameterTask.BINOMIAL,
        target_column="target_td_landed",
        exposure_column="target_td_attempted",
        notes="Completed takedowns conditional on attempts.",
    ),
    SimulationParameter.CONTROL_SECONDS: ParameterModelSpec(
        parameter=SimulationParameter.CONTROL_SECONDS,
        task=ParameterTask.ZERO_INFLATED_CONTINUOUS,
        target_column="target_control_seconds",
        notes="Control seconds with a separate zero-mass estimate.",
    ),
    SimulationParameter.KNOCKDOWNS: ParameterModelSpec(
        parameter=SimulationParameter.KNOCKDOWNS,
        task=ParameterTask.COUNT,
        target_column="target_knockdowns",
        notes="Rare-event round knockdown count.",
    ),
    SimulationParameter.KO_TKO_FINISH: ParameterModelSpec(
        parameter=SimulationParameter.KO_TKO_FINISH,
        task=ParameterTask.BINARY,
        target_column="target_fighter_ko_tko_finish",
        notes="Fighter KO/TKO competing finish hazard for the round.",
    ),
    SimulationParameter.SUBMISSION_FINISH: ParameterModelSpec(
        parameter=SimulationParameter.SUBMISSION_FINISH,
        task=ParameterTask.BINARY,
        target_column="target_fighter_submission_finish",
        notes="Fighter submission competing finish hazard for the round.",
    ),
}


PARAMETER_PREDICTION_REQUIRED_COLUMNS = [
    "fight_id",
    "fighter_id",
    "opponent_id",
    "round",
    "parameter",
    "prediction_mean",
    "prediction_probability",
    "prediction_dispersion",
    "prediction_zero_probability",
    "model_name",
    "model_version",
]


@dataclass(frozen=True)
class RoundParameterEstimate:
    """Validated parameter estimates for one fighter in one simulated round."""

    fight_id: str
    fighter_id: str
    opponent_id: str
    round: int
    sig_attempts_mean: float
    sig_attempts_dispersion: float
    sig_accuracy_probability: float
    td_attempts_mean: float
    td_attempts_dispersion: float
    td_accuracy_probability: float
    control_seconds_mean: float
    control_zero_probability: float
    knockdowns_mean: float
    ko_tko_finish_probability: float
    submission_finish_probability: float
    model_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in (
            "sig_attempts_mean",
            "sig_attempts_dispersion",
            "td_attempts_mean",
            "td_attempts_dispersion",
            "control_seconds_mean",
            "knockdowns_mean",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ParameterContractError(f"{name} must be finite and nonnegative; received {value!r}")

        for name in (
            "sig_accuracy_probability",
            "td_accuracy_probability",
            "control_zero_probability",
            "ko_tko_finish_probability",
            "submission_finish_probability",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ParameterContractError(f"{name} must be between 0 and 1; received {value!r}")

        if self.round <= 0:
            raise ParameterContractError("round must be positive")
        if self.ko_tko_finish_probability + self.submission_finish_probability > 1.0 + 1e-9:
            raise ParameterContractError(
                "competing finish probabilities cannot sum to more than 1"
            )


@runtime_checkable
class FittedRoundParameterModel(Protocol):
    """Interface required from a fitted component model bundle."""

    spec: ParameterModelSpec
    model_name: str
    model_version: str
    feature_columns: Sequence[str]

    def predict_parameters(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Return normalized long-form parameter predictions."""
        ...


@runtime_checkable
class RoundParameterProvider(Protocol):
    """Interface consumed by a future trained-parameter simulator engine."""

    def predict_round_parameters(self, feature_frame: pd.DataFrame) -> list[RoundParameterEstimate]:
        """Return one complete parameter estimate per fighter-round row."""
        ...


def validate_training_targets(
    training_df: pd.DataFrame,
    specs: Mapping[SimulationParameter, ParameterModelSpec] = DEFAULT_PARAMETER_MODEL_SPECS,
) -> None:
    """Verify that the historical table supports every registered model target."""
    missing: list[str] = []
    for spec in specs.values():
        if spec.target_column not in training_df.columns:
            missing.append(spec.target_column)
        if spec.exposure_column is not None and spec.exposure_column not in training_df.columns:
            missing.append(spec.exposure_column)
    if missing:
        raise ParameterContractError(
            f"training dataset is missing registered parameter targets: {sorted(set(missing))}"
        )


def validate_parameter_prediction_frame(
    predictions_df: pd.DataFrame,
    specs: Mapping[SimulationParameter, ParameterModelSpec] = DEFAULT_PARAMETER_MODEL_SPECS,
) -> None:
    """Validate normalized long-form output from component parameter models."""
    missing = [column for column in PARAMETER_PREDICTION_REQUIRED_COLUMNS if column not in predictions_df.columns]
    if missing:
        raise ParameterContractError(f"parameter predictions are missing columns: {missing}")

    valid_parameters = {parameter.value for parameter in specs}
    unknown = sorted(set(predictions_df["parameter"].dropna().astype(str)) - valid_parameters)
    if unknown:
        raise ParameterContractError(f"parameter predictions contain unknown parameters: {unknown}")

    duplicate_count = int(
        predictions_df.duplicated(subset=["fight_id", "fighter_id", "round", "parameter"]).sum()
    )
    if duplicate_count:
        raise ParameterContractError(
            f"parameter predictions have duplicate fighter-round-parameter keys: {duplicate_count}"
        )

    numeric_columns = [
        "prediction_mean",
        "prediction_probability",
        "prediction_dispersion",
        "prediction_zero_probability",
    ]
    for column in numeric_columns:
        numeric = pd.to_numeric(predictions_df[column], errors="coerce")
        non_missing = numeric.dropna()
        if not np.isfinite(non_missing).all():
            raise ParameterContractError(f"{column} contains non-finite values")

    mean = pd.to_numeric(predictions_df["prediction_mean"], errors="coerce")
    dispersion = pd.to_numeric(predictions_df["prediction_dispersion"], errors="coerce")
    if mean.dropna().lt(0).any():
        raise ParameterContractError("prediction_mean contains negative values")
    if dispersion.dropna().lt(0).any():
        raise ParameterContractError("prediction_dispersion contains negative values")

    for column in ("prediction_probability", "prediction_zero_probability"):
        values = pd.to_numeric(predictions_df[column], errors="coerce").dropna()
        if (~values.between(0.0, 1.0)).any():
            raise ParameterContractError(f"{column} contains values outside [0, 1]")


def pivot_parameter_predictions(predictions_df: pd.DataFrame) -> list[RoundParameterEstimate]:
    """Convert validated long-form model outputs into complete round estimates."""
    validate_parameter_prediction_frame(predictions_df)

    estimates: list[RoundParameterEstimate] = []
    key_columns = ["fight_id", "fighter_id", "opponent_id", "round"]

    for keys, group in predictions_df.groupby(key_columns, dropna=False, sort=False):
        by_parameter = {row["parameter"]: row for _, row in group.iterrows()}
        missing_parameters = [
            parameter.value
            for parameter in DEFAULT_PARAMETER_MODEL_SPECS
            if parameter.value not in by_parameter
        ]
        if missing_parameters:
            raise ParameterContractError(
                f"fighter-round {keys} is missing parameter predictions: {missing_parameters}"
            )

        def value(parameter: SimulationParameter, column: str, default: float | None = None) -> float:
            raw = by_parameter[parameter.value][column]
            if pd.isna(raw):
                if default is None:
                    raise ParameterContractError(
                        f"fighter-round {keys} parameter {parameter.value} is missing {column}"
                    )
                return float(default)
            return float(raw)

        model_versions = {
            parameter: str(row["model_version"])
            for parameter, row in by_parameter.items()
        }

        estimates.append(
            RoundParameterEstimate(
                fight_id=str(keys[0]),
                fighter_id=str(keys[1]),
                opponent_id=str(keys[2]),
                round=int(keys[3]),
                sig_attempts_mean=value(SimulationParameter.SIG_ATTEMPTS, "prediction_mean"),
                sig_attempts_dispersion=value(
                    SimulationParameter.SIG_ATTEMPTS,
                    "prediction_dispersion",
                    default=0.0,
                ),
                sig_accuracy_probability=value(
                    SimulationParameter.SIG_ACCURACY,
                    "prediction_probability",
                ),
                td_attempts_mean=value(SimulationParameter.TD_ATTEMPTS, "prediction_mean"),
                td_attempts_dispersion=value(
                    SimulationParameter.TD_ATTEMPTS,
                    "prediction_dispersion",
                    default=0.0,
                ),
                td_accuracy_probability=value(
                    SimulationParameter.TD_ACCURACY,
                    "prediction_probability",
                ),
                control_seconds_mean=value(
                    SimulationParameter.CONTROL_SECONDS,
                    "prediction_mean",
                ),
                control_zero_probability=value(
                    SimulationParameter.CONTROL_SECONDS,
                    "prediction_zero_probability",
                ),
                knockdowns_mean=value(SimulationParameter.KNOCKDOWNS, "prediction_mean"),
                ko_tko_finish_probability=value(
                    SimulationParameter.KO_TKO_FINISH,
                    "prediction_probability",
                ),
                submission_finish_probability=value(
                    SimulationParameter.SUBMISSION_FINISH,
                    "prediction_probability",
                ),
                model_versions=model_versions,
            )
        )

    return estimates
