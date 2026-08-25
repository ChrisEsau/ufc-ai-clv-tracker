"""FSR V3 -> Standard Fighter V1 capability translation.

This module owns the research boundary between validated/matchup-aware FSR V3
runtime quantities and the dimensionless capability inputs used by the Standard
Fighter decision policy.

Important constraints
---------------------
* It does not change FSR V3 ratings or Event Clock mechanics.
* It does not classify fighters into archetypes.
* Standing, takedown and ground-top capabilities are matchup-aware empirical
  ranks of already-derived Event Clock V2 runtime quantities.
* Clinch, submission, escape and reversal remain explicit neutral placeholders
  until a separate semantic mapping is reviewed.  No hidden proxy is invented.
* Cold start is metadata only: a fighter is flagged when there are zero prior
  UFC FSR snapshots before the audited fight.  The flag does not alter the
  capability values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    FighterPathTraits,
    derive_runtime_inputs,
)
from .policy import Capability

NEUTRAL_CLINCH = 0.35
NEUTRAL_SUBMISSION = 0.30
NEUTRAL_ESCAPE = 0.40
NEUTRAL_REVERSAL = 0.30


def _pct(series: pd.Series, value: float) -> float:
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("empty capability reference distribution")
    return float(np.mean(arr <= float(value)))


def traits_from_row(row: pd.Series) -> FighterPathTraits:
    values: dict[str, float] = {}
    for key, value in row.items():
        if key == "fighter_id":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values[key] = numeric
    return FighterPathTraits(str(row["fighter_id"]), values, False)


def _median_row(frame: pd.DataFrame) -> pd.Series:
    row: dict[str, object] = {"fighter_id": "POP_MEDIAN"}
    for column in frame.columns:
        if column == "fighter_id":
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            row[column] = float(frame[column].median())
    return pd.Series(row)


@dataclass(frozen=True)
class CapabilityReference:
    """Population reference distributions for matchup runtime quantities."""

    runtime: pd.DataFrame

    @classmethod
    def from_latest(cls, latest: pd.DataFrame) -> "CapabilityReference":
        """Build the live/latest reference; forbidden in historical calibration."""
        return cls.from_frame(latest)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "CapabilityReference":
        """Build a fixed reference from an explicitly provenance-bounded frame."""
        median = traits_from_row(_median_row(frame))
        rows: list[dict[str, float | str]] = []
        for _, row in frame.iterrows():
            try:
                runtime = derive_runtime_inputs(traits_from_row(row), median)
            except Exception:
                continue
            rows.append(
                {
                    "fighter_id": str(row["fighter_id"]),
                    "standing_rate": runtime.standing_rate_15m,
                    "standing_acc": runtime.standing_accuracy,
                    "td_rate": runtime.takedown_rate_15m,
                    "td_comp": runtime.takedown_completion,
                    "ground_rate": runtime.ground_slope_rate_15m_own_control,
                    "ground_acc": runtime.ground_accuracy,
                }
            )
        runtime = pd.DataFrame(rows)
        if len(runtime) < 50:
            raise RuntimeError(
                f"too few valid FSR V3 profiles for capability reference: {len(runtime)}"
            )
        return cls(runtime=runtime)

    @classmethod
    def from_prefight_before(
        cls, snapshots: pd.DataFrame, cutoff
    ) -> "CapabilityReference":
        """Chronology-safe calibration reference using only pre-cutoff states."""
        cutoff_date = pd.Timestamp(cutoff).normalize()
        dated = snapshots.loc[
            pd.to_datetime(snapshots["event_date"]).dt.normalize().lt(cutoff_date)
        ].copy()
        if dated.empty:
            raise RuntimeError("no prefight states before capability-reference cutoff")
        # One last known *historical* state per fighter, all observed before the
        # earliest fight in the frozen cohort. Future performance cannot enter.
        dated = dated.sort_values(["event_date", "fight_id"]).drop_duplicates(
            "fighter_id", keep="last"
        )
        return cls.from_frame(dated)


@dataclass(frozen=True)
class CapabilityTranslation:
    fighter_id: str
    opponent_id: str
    capability: Capability
    standing_rate_15m: float
    standing_accuracy: float
    takedown_rate_15m: float
    takedown_completion: float
    ground_rate_15m_own_control: float
    ground_accuracy: float
    standing_rate_percentile: float
    standing_accuracy_percentile: float
    takedown_rate_percentile: float
    takedown_completion_percentile: float
    ground_rate_percentile: float
    ground_accuracy_percentile: float
    prior_ufc_fights: int
    cold_start: bool


def prior_snapshot_count(
    snapshots: pd.DataFrame,
    *,
    fighter_id: str,
    event_date,
) -> int:
    """Count strictly prior-date UFC FSR snapshots for one fighter."""
    date = pd.Timestamp(event_date).normalize()
    fighter = str(fighter_id)
    rows = snapshots[
        snapshots["fighter_id"].astype(str).eq(fighter)
        & pd.to_datetime(snapshots["event_date"]).dt.normalize().lt(date)
    ]
    # One prefight snapshot per UFC fight; same-date rows are intentionally not
    # counted because V3 treats same-date fights as sharing the same prefight state.
    return int(rows[["event_date", "fight_id"]].drop_duplicates().shape[0])


def translate_capability(
    attacker: pd.Series,
    defender: pd.Series,
    reference: CapabilityReference,
    *,
    prior_ufc_fights: int,
) -> CapabilityTranslation:
    """Translate one directional FSR matchup into Standard Fighter capability.

    V1 semantics:
    * standing = mean(empirical standing-rate rank, landing-accuracy rank)
    * counter = landing-accuracy rank
    * pressure = standing-rate rank
    * takedown = mean(empirical TD-generation rank, TD-completion rank)
    * ground_top = mean(empirical top-strike-rate rank, ground-accuracy rank)

    All ranks are measured against the canonical latest-profile population facing
    a median opponent.  Directional attacker-vs-defender runtime inputs are used
    for the audited matchup, so opponent suppression/defense affects capability.
    """
    runtime = derive_runtime_inputs(
        traits_from_row(attacker), traits_from_row(defender)
    )
    pop = reference.runtime

    standing_rate_pct = _pct(pop["standing_rate"], runtime.standing_rate_15m)
    standing_acc_pct = _pct(pop["standing_acc"], runtime.standing_accuracy)
    td_rate_pct = _pct(pop["td_rate"], runtime.takedown_rate_15m)
    td_comp_pct = _pct(pop["td_comp"], runtime.takedown_completion)
    ground_rate_pct = _pct(
        pop["ground_rate"], runtime.ground_slope_rate_15m_own_control
    )
    ground_acc_pct = _pct(pop["ground_acc"], runtime.ground_accuracy)

    capability = Capability(
        standing=(standing_rate_pct + standing_acc_pct) / 2.0,
        counter=standing_acc_pct,
        pressure=standing_rate_pct,
        clinch=NEUTRAL_CLINCH,
        takedown=(td_rate_pct + td_comp_pct) / 2.0,
        ground_top=(ground_rate_pct + ground_acc_pct) / 2.0,
        submission=NEUTRAL_SUBMISSION,
        escape=NEUTRAL_ESCAPE,
        reversal=NEUTRAL_REVERSAL,
    )

    prior = max(int(prior_ufc_fights), 0)
    return CapabilityTranslation(
        fighter_id=str(attacker["fighter_id"]),
        opponent_id=str(defender["fighter_id"]),
        capability=capability,
        standing_rate_15m=float(runtime.standing_rate_15m),
        standing_accuracy=float(runtime.standing_accuracy),
        takedown_rate_15m=float(runtime.takedown_rate_15m),
        takedown_completion=float(runtime.takedown_completion),
        ground_rate_15m_own_control=float(runtime.ground_slope_rate_15m_own_control),
        ground_accuracy=float(runtime.ground_accuracy),
        standing_rate_percentile=standing_rate_pct,
        standing_accuracy_percentile=standing_acc_pct,
        takedown_rate_percentile=td_rate_pct,
        takedown_completion_percentile=td_comp_pct,
        ground_rate_percentile=ground_rate_pct,
        ground_accuracy_percentile=ground_acc_pct,
        prior_ufc_fights=prior,
        cold_start=(prior == 0),
    )
