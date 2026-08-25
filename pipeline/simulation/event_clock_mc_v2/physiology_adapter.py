"""Canonical historical FSR-to-causal-physiology translation.

Native V3 power and KD resistance are latent logit effects.  The coordinate
changes below preserve the already-approved frozen mechanics coefficients;
inherited durability and stamina ratings remain direct prefight values.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from pipeline.fsr_v3.active_config import ActiveTraitConfig
from pipeline.simulation.event_clock_mc_v2.mechanics.config import FighterMechanics

FROZEN_KD_POWER_BETA = 0.020741
REQUIRED_PREFIGHT_PHYSIOLOGY_COLUMNS = frozenset(
    {
        "event_date", "fight_id", "fighter_id", "striking_power_v3",
        "damage_durability", "knockdown_resistance_v3",
        "stamina_capacity", "stamina_depletion_resistance",
        "submission_conversion_baseline", "submission_offense", "submission_defense",
    }
)


def legacy_power_equivalent(native_power):
    """Map native V3 power so beta*(rating-50) equals the native latent."""
    latent = np.asarray(native_power, dtype=float)
    if not np.isfinite(latent).all():
        raise ValueError("non-finite V3 striking power")
    return 50.0 + latent / FROZEN_KD_POWER_BETA


def legacy_kdres_equivalent(native_resistance, config: ActiveTraitConfig | None = None):
    """Map positive native resistance to the frozen negative-beta coordinate."""
    beta = float((config or ActiveTraitConfig()).frozen_event_clock_kdres_beta)
    if not math.isfinite(beta) or beta >= 0:
        raise RuntimeError(f"invalid frozen KD-resistance beta: {beta}")
    latent = np.asarray(native_resistance, dtype=float)
    if not np.isfinite(latent).all():
        raise ValueError("non-finite native KD resistance")
    return 50.0 - latent / beta


@dataclass(frozen=True)
class PhysiologyTraitMapping:
    source_column: str
    source_semantics: str
    transformation: str


PHYSIOLOGY_TRAIT_MAPPING = {
    "striking_power": PhysiologyTraitMapping("striking_power_v3", "historical prefight V3 KD-production latent", "50 + latent / 0.020741"),
    "damage_durability": PhysiologyTraitMapping("damage_durability", "historical prefight inherited 10-90 rating", "identity"),
    "knockdown_resistance": PhysiologyTraitMapping("knockdown_resistance_v3", "historical prefight V3 resistance latent", "50 - latent / -0.014421"),
    "stamina_capacity": PhysiologyTraitMapping("stamina_capacity", "historical canonical fixed capacity", "identity"),
    "stamina_depletion_resistance": PhysiologyTraitMapping("stamina_depletion_resistance", "historical prefight inherited 10-90 rating", "identity"),
}


def fighter_mechanics_from_prefight(
    prefight_row: Mapping,
    runtime,
    *,
    submission_success_probability: float = 0.0,
    ground_escape_probability: float = 0.40,
    ground_reversal_probability: float = 0.30,
    age_years: float = 30.0,
    submission_conversion_offset: float = 0.0,
) -> FighterMechanics:
    """Build mechanics from one exact historical prefight row and matchup runtime."""
    record = dict(prefight_row)
    missing = REQUIRED_PREFIGHT_PHYSIOLOGY_COLUMNS.difference(record)
    if missing:
        raise ValueError(f"canonical prefight row missing physiology columns: {sorted(missing)}")
    for name in REQUIRED_PREFIGHT_PHYSIOLOGY_COLUMNS.difference({"event_date", "fight_id", "fighter_id"}):
        value = float(record[name])
        if not math.isfinite(value):
            raise ValueError(f"non-finite canonical prefight physiology value: {name}")
    if float(record["stamina_capacity"]) != 100.0:
        raise ValueError("canonical stamina_capacity must remain 100.0")
    return FighterMechanics(
        standing_strike_landing_probability=float(runtime.standing_accuracy),
        takedown_completion_probability=float(runtime.takedown_completion),
        ground_strike_landing_probability=float(runtime.ground_accuracy),
        submission_success_probability=submission_success_probability,
        ground_escape_probability=ground_escape_probability,
        ground_reversal_probability=ground_reversal_probability,
        striking_power=float(legacy_power_equivalent(record["striking_power_v3"])),
        damage_durability=float(record["damage_durability"]),
        knockdown_resistance=float(legacy_kdres_equivalent(record["knockdown_resistance_v3"])),
        stamina_capacity=float(record["stamina_capacity"]),
        stamina_depletion_resistance=float(record["stamina_depletion_resistance"]),
        age_years=float(age_years),
        submission_conversion_baseline=float(record["submission_conversion_baseline"]),
        submission_offense=float(record["submission_offense"]),
        submission_defense=float(record["submission_defense"]),
        submission_conversion_offset=float(submission_conversion_offset),
    )


def age_years_on_date(dob, event_date) -> float:
    """Derive exact fight-date age when snapshots do not publish canonical age."""
    if dob is None or pd.isna(dob):
        return 30.0
    birth = pd.Timestamp(dob)
    fight_date = pd.Timestamp(event_date)
    age = (fight_date - birth).days / 365.2425
    if not np.isfinite(age) or age <= 0.0:
        raise ValueError(f"invalid DOB/event date: dob={dob}, event_date={event_date}")
    return float(age)
