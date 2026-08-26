"""Typed provider contract for mutually exclusive fight-round finish hazards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from pipeline.simulation.finish_hazard_model import FINISH_CLASSES


class FinishHazardProviderError(ValueError):
    """Raised when finish-hazard provider data violates the contract."""


@dataclass(frozen=True)
class FinishHazardKey:
    fight_id: str
    round: int

    def __post_init__(self) -> None:
        if not str(self.fight_id).strip():
            raise FinishHazardProviderError("fight_id must be non-empty")
        if int(self.round) <= 0:
            raise FinishHazardProviderError("round must be positive")


@dataclass(frozen=True)
class FinishHazardProbabilities:
    key: FinishHazardKey
    no_finish: float
    red_ko_tko: float
    red_submission: float
    blue_ko_tko: float
    blue_submission: float
    model_name: str
    model_version: str
    source: str = "historical_counterfactual_holdout"

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.no_finish,
                self.red_ko_tko,
                self.red_submission,
                self.blue_ko_tko,
                self.blue_submission,
            ],
            dtype=float,
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise FinishHazardProviderError(
                "Finish probabilities must be finite and nonnegative"
            )
        if not np.isclose(float(values.sum()), 1.0, atol=1e-6):
            raise FinishHazardProviderError(
                f"Finish probabilities must sum to one; received {values.sum()}"
            )
        if not str(self.model_name).strip() or not str(self.model_version).strip():
            raise FinishHazardProviderError(
                "model_name and model_version must be non-empty"
            )

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.no_finish,
                self.red_ko_tko,
                self.red_submission,
                self.blue_ko_tko,
                self.blue_submission,
            ],
            dtype=float,
        )


@runtime_checkable
class FinishHazardProvider(Protocol):
    def finish_hazards(self, key: FinishHazardKey) -> FinishHazardProbabilities:
        """Return one calibrated five-class probability vector."""
        ...


class HistoricalFinishHazardProvider:
    """Lookup provider backed by counterfactual holdout predictions.

    The constructor validates the DataFrame once, then converts every row to a
    compact dictionary. Monte Carlo paths can therefore perform millions of
    provider lookups without repeated pandas indexing inside the simulation loop.
    """

    REQUIRED_COLUMNS = (
        "fight_id",
        "round",
        "model_name",
        *[f"calibrated_prob_{name}" for name in FINISH_CLASSES],
    )

    def __init__(
        self,
        predictions: pd.DataFrame,
        model_name: str,
        model_version: str = "finish_hazard_prefight_v0",
    ) -> None:
        missing = [
            column for column in self.REQUIRED_COLUMNS if column not in predictions.columns
        ]
        if missing:
            raise FinishHazardProviderError(
                f"Finish predictions are missing provider columns: {missing}"
            )
        frame = predictions.loc[
            predictions["model_name"].astype(str).eq(str(model_name))
        ].copy()
        if frame.empty:
            raise FinishHazardProviderError(
                f"No finish predictions found for model {model_name!r}"
            )
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["round"] = pd.to_numeric(frame["round"], errors="coerce")
        probability_columns = [f"calibrated_prob_{name}" for name in FINISH_CLASSES]
        for column in probability_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[["round", *probability_columns]].isna().any().any():
            raise FinishHazardProviderError(
                "Finish predictions contain missing provider values"
            )
        frame["round"] = frame["round"].astype(int)
        probabilities = frame[probability_columns].to_numpy(dtype=float)
        if np.any(probabilities < 0.0) or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-6
        ):
            raise FinishHazardProviderError(
                "Finish prediction probability rows must be nonnegative and sum to one"
            )
        keys = ["fight_id", "round"]
        if frame.duplicated(keys).any():
            raise FinishHazardProviderError(
                "Finish predictions contain duplicate fight-round keys"
            )

        self.model_name = str(model_name)
        self.model_version = str(model_version)
        self._rows: dict[tuple[str, int], tuple[float, float, float, float, float]] = {}
        for row, values in zip(
            frame[["fight_id", "round"]].itertuples(index=False),
            probabilities,
        ):
            self._rows[(str(row.fight_id), int(row.round))] = tuple(
                float(value) for value in values
            )

    def __len__(self) -> int:
        return int(len(self._rows))

    def finish_hazards(self, key: FinishHazardKey) -> FinishHazardProbabilities:
        lookup = (str(key.fight_id), int(key.round))
        try:
            values = self._rows[lookup]
        except KeyError as exc:
            raise FinishHazardProviderError(
                f"No finish hazards found for {lookup}"
            ) from exc
        return FinishHazardProbabilities(
            key=key,
            no_finish=values[0],
            red_ko_tko=values[1],
            red_submission=values[2],
            blue_ko_tko=values[3],
            blue_submission=values[4],
            model_name=self.model_name,
            model_version=self.model_version,
        )
