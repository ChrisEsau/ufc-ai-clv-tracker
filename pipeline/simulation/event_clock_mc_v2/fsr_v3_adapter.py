"""FSR V3 input adapter for Event Clock MC V2.

This module is intentionally isolated from Event Clock V1 mechanics.  It owns
only the semantic boundary between canonical FSR V3 fighter state and the
runtime inputs consumed by the Event Clock simulation.

Key rules
---------
* Standing and takedown suppression are multiplicative in V3:
      effective_rate = attacker_tendency * defender_suppression
  where suppression < 1 reduces opponent event generation.
* Ground striking uses the validated burst + suppressed slope model:
      expected = burst + own_control_seconds / 900 *
                 attacker_ground_tendency * defender_ground_suppression
  The burst is never multiplied by defender suppression.
* Ground landing effectiveness is attacker-only in V3.  There is no
  ``ground_striking_defense`` term.
* Only the four traits whose epistemic variance improved chronological
  prediction are sampled per path:
      takedown_tendency
      takedown_suppression
      standing_striking_tendency
      standing_striking_suppression
* One draw is made at path creation and then held fixed for the entire path.
* NB2 alpha and Beta-Binomial rho are aleatoric observation noise and never
  appear in this sampler.

The FSR V3 uncertainty publication currently contains posterior mean and SD,
not the full posterior grid.  Positive latent traits are therefore projected to
a moment-matched Gamma distribution for ECV2 path initialization.  This keeps
the draw positive while preserving the validated posterior mean and SD.  The
mean-only comparator bypasses all sampling exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import (
    FSR_V3_LATEST_PATH,
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
    FSR_V3_PREFIGHT_UNCERTAINTY_PATH,
)


EPS = 1e-9

SAMPLABLE_EPISTEMIC_TRAITS = frozenset(
    {
        "takedown_tendency",
        "takedown_suppression",
        "standing_striking_tendency",
        "standing_striking_suppression",
    }
)

V3_REBUILT_TRAITS = frozenset(
    {
        "takedown_tendency",
        "takedown_suppression",
        "takedown_offense",
        "takedown_defense",
        "standing_striking_tendency",
        "standing_striking_suppression",
        "standing_striking_offense",
        "standing_striking_defense",
        "ground_striking_tendency",
        "ground_striking_suppression",
        "ground_striking_offense",
    }
)

REQUIRED_MATCHUP_COLUMNS = frozenset(
    {
        "fighter_id",
        "standing_striking_tendency",
        "standing_striking_suppression",
        "standing_striking_offense",
        "standing_striking_defense",
        "standing_accuracy_baseline",
        "takedown_tendency",
        "takedown_suppression",
        "takedown_offense",
        "takedown_defense",
        "takedown_completion_baseline",
        "ground_striking_tendency",
        "ground_striking_suppression",
        "ground_striking_offense",
        "ground_accuracy_baseline",
        "ground_striking_burst_baseline",
    }
)


@dataclass(frozen=True)
class FighterPathTraits:
    """Immutable fighter trait state for one Monte Carlo path."""

    fighter_id: str
    values: Mapping[str, float]
    epistemic_sampled: bool

    def __getitem__(self, key: str) -> float:
        return self.values[key]


@dataclass(frozen=True)
class MatchupRuntimeInputs:
    """V3-derived runtime inputs for one attacker against one defender."""

    attacker_id: str
    defender_id: str
    standing_rate_15m: float
    takedown_rate_15m: float
    ground_slope_rate_15m_own_control: float
    ground_burst_attempts: float
    standing_accuracy: float
    takedown_completion: float
    ground_accuracy: float

    def ground_expected_attempts(self, own_control_seconds: float) -> float:
        seconds = max(float(own_control_seconds), 0.0)
        return float(
            self.ground_burst_attempts
            + seconds / 900.0 * self.ground_slope_rate_15m_own_control
        )


@dataclass(frozen=True)
class PathMatchup:
    """Immutable red/blue path traits plus directional runtime transforms."""

    red: FighterPathTraits
    blue: FighterPathTraits
    red_vs_blue: MatchupRuntimeInputs
    blue_vs_red: MatchupRuntimeInputs


def _clip_probability(value: float) -> float:
    return float(np.clip(float(value), EPS, 1.0 - EPS))


def _logit(value: float) -> float:
    p = _clip_probability(value)
    return log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + exp(-value))


def _as_record(row: Mapping | pd.Series) -> dict[str, object]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _validate_fighter_row(record: Mapping[str, object]) -> None:
    missing = REQUIRED_MATCHUP_COLUMNS.difference(record)
    if missing:
        raise ValueError(f"FSR V3 fighter row missing columns: {sorted(missing)}")
    if "ground_striking_defense" in record:
        # V3 deliberately removed this trait.  Rejecting it here prevents a
        # downstream adapter from accidentally restoring the rejected model.
        raise ValueError("ground_striking_defense is not a valid FSR V3 runtime trait")

    for column in REQUIRED_MATCHUP_COLUMNS.difference({"fighter_id"}):
        value = float(record[column])
        if not np.isfinite(value):
            raise ValueError(f"non-finite FSR V3 value for {column}: {value}")

    for column in (
        "standing_striking_tendency",
        "standing_striking_suppression",
        "takedown_tendency",
        "takedown_suppression",
        "ground_striking_tendency",
        "ground_striking_suppression",
        "ground_striking_burst_baseline",
    ):
        if float(record[column]) < 0.0:
            raise ValueError(f"FSR V3 positive trait {column} cannot be negative")


def load_prefight_snapshots(
    path: Path = FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    if frame.duplicated(["event_date", "fight_id", "fighter_id"]).any():
        raise ValueError("duplicate FSR V3 historical prefight rows")
    if "ground_striking_defense" in frame.columns:
        raise ValueError("rejected ground_striking_defense present in FSR V3 snapshot")
    return frame


def load_latest_profiles(path: Path = FSR_V3_LATEST_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    if frame.duplicated(["fighter_id"]).any():
        raise ValueError("duplicate FSR V3 latest fighter rows")
    if "ground_striking_defense" in frame.columns:
        raise ValueError("rejected ground_striking_defense present in FSR V3 latest")
    return frame


def load_prefight_uncertainty(
    path: Path = FSR_V3_PREFIGHT_UNCERTAINTY_PATH,
) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    key = ["event_date", "fight_id", "fighter_id", "trait"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate FSR V3 uncertainty rows")

    enabled = set(frame.loc[frame["sampling_enabled"].astype(bool), "trait"])
    unexpected = enabled.difference(SAMPLABLE_EPISTEMIC_TRAITS)
    missing = SAMPLABLE_EPISTEMIC_TRAITS.difference(enabled)
    if unexpected:
        raise ValueError(f"unexpected epistemic-sampling traits: {sorted(unexpected)}")
    if missing:
        raise ValueError(f"validated epistemic-sampling traits missing: {sorted(missing)}")
    return frame


def historical_fighter_rows(
    snapshots: pd.DataFrame,
    *,
    event_date,
    fight_id: str,
    fighter_ids: tuple[str, str],
) -> tuple[pd.Series, pd.Series]:
    date = pd.Timestamp(event_date).normalize()
    fight = str(fight_id)
    ids = tuple(str(value) for value in fighter_ids)
    subset = snapshots[
        snapshots["event_date"].eq(date)
        & snapshots["fight_id"].eq(fight)
        & snapshots["fighter_id"].isin(ids)
    ]
    if len(subset) != 2 or set(subset["fighter_id"]) != set(ids):
        raise KeyError(
            f"expected two FSR V3 rows for fight={fight} date={date.date()} ids={ids}; "
            f"found {len(subset)}"
        )
    lookup = subset.set_index("fighter_id", drop=False)
    return lookup.loc[ids[0]], lookup.loc[ids[1]]


def historical_uncertainty_rows(
    uncertainty: pd.DataFrame,
    *,
    event_date,
    fight_id: str,
    fighter_id: str,
) -> pd.DataFrame:
    date = pd.Timestamp(event_date).normalize()
    subset = uncertainty[
        uncertainty["event_date"].eq(date)
        & uncertainty["fight_id"].eq(str(fight_id))
        & uncertainty["fighter_id"].eq(str(fighter_id))
    ].copy()
    if subset.empty:
        raise KeyError(
            f"no FSR V3 uncertainty for fight={fight_id} fighter={fighter_id} date={date.date()}"
        )
    return subset


def _moment_matched_gamma_draw(
    mean: float,
    sd: float,
    rng: np.random.Generator,
) -> float:
    """Draw a positive variate preserving supplied mean and SD."""
    mean = float(mean)
    sd = float(sd)
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"positive posterior mean required for Gamma projection: {mean}")
    if not np.isfinite(sd) or sd < 0.0:
        raise ValueError(f"non-negative posterior SD required: {sd}")
    if sd <= EPS:
        return mean
    shape = (mean / sd) ** 2
    scale = sd * sd / mean
    return float(rng.gamma(shape=shape, scale=scale))


def initialize_fighter_path_traits(
    fighter_row: Mapping | pd.Series,
    uncertainty_rows: pd.DataFrame | None,
    *,
    rng: np.random.Generator,
    sample_epistemic: bool,
) -> FighterPathTraits:
    """Create one immutable fighter state for an entire MC path.

    ``sample_epistemic=False`` is the required means-only comparator.  When
    enabled, only rows explicitly marked sampling_enabled and belonging to the
    validated four-trait set are drawn.
    """
    record = _as_record(fighter_row)
    _validate_fighter_row(record)
    values: dict[str, float] = {}
    for key, value in record.items():
        if key == "fighter_id":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values[key] = numeric

    sampled = False
    if sample_epistemic:
        if uncertainty_rows is None:
            raise ValueError("uncertainty rows are required when epistemic sampling is enabled")
        for row in uncertainty_rows.to_dict("records"):
            trait = str(row["trait"])
            enabled = bool(row["sampling_enabled"])
            multiplier = float(row.get("variance_multiplier", 0.0))
            if not enabled:
                continue
            if trait not in SAMPLABLE_EPISTEMIC_TRAITS:
                raise ValueError(f"trait {trait} is not approved for ECV2 epistemic sampling")
            if multiplier <= 0.0:
                continue
            mean = float(row["posterior_mean"])
            sd = float(row["posterior_sd"]) * float(np.sqrt(multiplier))
            values[trait] = _moment_matched_gamma_draw(mean, sd, rng)
            sampled = True

    return FighterPathTraits(
        fighter_id=str(record["fighter_id"]),
        values=MappingProxyType(values),
        epistemic_sampled=sampled,
    )


def derive_runtime_inputs(
    attacker: FighterPathTraits,
    defender: FighterPathTraits,
) -> MatchupRuntimeInputs:
    """Translate immutable V3 path traits into directional Event Clock inputs."""
    a = attacker.values
    d = defender.values

    standing_rate = max(
        float(a["standing_striking_tendency"])
        * float(d["standing_striking_suppression"]),
        0.0,
    )
    takedown_rate = max(
        float(a["takedown_tendency"])
        * float(d["takedown_suppression"]),
        0.0,
    )
    ground_slope = max(
        float(a["ground_striking_tendency"])
        * float(d["ground_striking_suppression"]),
        0.0,
    )

    standing_accuracy = _sigmoid(
        _logit(float(a["standing_accuracy_baseline"]))
        + float(a["standing_striking_offense"])
        - float(d["standing_striking_defense"])
    )
    takedown_completion = _sigmoid(
        _logit(float(a["takedown_completion_baseline"]))
        + float(a["takedown_offense"])
        - float(d["takedown_defense"])
    )
    # Ground effectiveness is attacker-only in FSR V3.
    ground_accuracy = _sigmoid(
        _logit(float(a["ground_accuracy_baseline"]))
        + float(a["ground_striking_offense"])
    )

    return MatchupRuntimeInputs(
        attacker_id=attacker.fighter_id,
        defender_id=defender.fighter_id,
        standing_rate_15m=standing_rate,
        takedown_rate_15m=takedown_rate,
        ground_slope_rate_15m_own_control=ground_slope,
        ground_burst_attempts=max(float(a["ground_striking_burst_baseline"]), 0.0),
        standing_accuracy=standing_accuracy,
        takedown_completion=takedown_completion,
        ground_accuracy=ground_accuracy,
    )


def initialize_path_matchup(
    red_row: Mapping | pd.Series,
    blue_row: Mapping | pd.Series,
    red_uncertainty: pd.DataFrame | None,
    blue_uncertainty: pd.DataFrame | None,
    *,
    rng: np.random.Generator,
    sample_epistemic: bool,
) -> PathMatchup:
    """Draw path-level FSR state once, then derive both directional matchups."""
    red = initialize_fighter_path_traits(
        red_row,
        red_uncertainty,
        rng=rng,
        sample_epistemic=sample_epistemic,
    )
    blue = initialize_fighter_path_traits(
        blue_row,
        blue_uncertainty,
        rng=rng,
        sample_epistemic=sample_epistemic,
    )
    return PathMatchup(
        red=red,
        blue=blue,
        red_vs_blue=derive_runtime_inputs(red, blue),
        blue_vs_red=derive_runtime_inputs(blue, red),
    )
