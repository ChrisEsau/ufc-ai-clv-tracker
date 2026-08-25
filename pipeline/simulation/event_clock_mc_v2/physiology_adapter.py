"""Canonical historical FSR-to-causal-physiology translation.

Native V3 power and KD resistance are already logit-scale fighter effects.
Event Clock V2 consumes those native effects directly at the physiology
boundary. Inherited durability and stamina ratings remain direct prefight
values.

The legacy coordinate helpers remain only for compatibility with the older
profile boundary; the V2 causal physiology adapter does not round-trip native
latents through those synthetic rating coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import numpy as np

from pipeline.fsr_v3.active_config import ActiveTraitConfig
from pipeline.simulation.event_clock_mc_v2.mechanics.config import FighterMechanics

FROZEN_KD_POWER_BETA = 0.020741
REQUIRED_PREFIGHT_PHYSIOLOGY_COLUMNS = frozenset(
    {
        "event_date",
        "fight_id",
        "fighter_id",
        "striking_power_v3",
        "damage_durability",
        "knockdown_resistance_v3",
        "stamina_capacity",
        "stamina_depletion_resistance",
    }
)


def legacy_power_equivalent(native_power):
    """Legacy profile coordinate preserving beta*(rating-50)=native latent."""
    latent = np.asarray(native_power, dtype=float)
    if not np.isfinite(latent).all():
        raise ValueError("non-finite V3 striking power")
    return 50.0 + latent / FROZEN_KD_POWER_BETA


def legacy_kdres_equivalent(
    native_resistance, config: ActiveTraitConfig | None = None
):
    """Legacy profile coordinate preserving beta*(rating-50)=-native latent."""
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
    "striking_power_log_effect": PhysiologyTraitMapping(
        "striking_power_v3",
        "historical prefight V3 attacker KD-production logit latent",
        "identity: consumed directly as log power effect",
    ),
    "damage_durability": PhysiologyTraitMapping(
        "damage_durability",
        "historical prefight inherited 10-90 rating",
        "identity",
    ),
    "knockdown_resistance_log_effect": PhysiologyTraitMapping(
        "knockdown_resistance_v3",
        "historical prefight V3 defender KD-resistance logit latent",
        "identity: consumed directly as log resistance effect",
    ),
    "stamina_capacity": PhysiologyTraitMapping(
        "stamina_capacity",
        "historical canonical fixed capacity",
        "identity",
    ),
    "stamina_depletion_resistance": PhysiologyTraitMapping(
        "stamina_depletion_resistance",
        "historical prefight inherited 10-90 rating",
        "identity",
    ),
}


def fighter_mechanics_from_prefight(
    prefight_row: Mapping,
    runtime,
    *,
    submission_success_probability: float = 0.0,
    ground_escape_probability: float = 0.40,
    ground_reversal_probability: float = 0.30,
) -> FighterMechanics:
    """Build mechanics from one exact historical prefight row and matchup runtime."""
    record = dict(prefight_row)
    missing = REQUIRED_PREFIGHT_PHYSIOLOGY_COLUMNS.difference(record)
    if missing:
        raise ValueError(
            f"canonical prefight row missing physiology columns: {sorted(missing)}"
        )
    for name in REQUIRED_PREFIGHT_PHYSIOLOGY_COLUMNS.difference(
        {"event_date", "fight_id", "fighter_id"}
    ):
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
        damage_durability=float(record["damage_durability"]),
        stamina_capacity=float(record["stamina_capacity"]),
        stamina_depletion_resistance=float(record["stamina_depletion_resistance"]),
        striking_power_log_effect=float(record["striking_power_v3"]),
        knockdown_resistance_log_effect=float(record["knockdown_resistance_v3"]),
    )
