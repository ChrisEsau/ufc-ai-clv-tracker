"""Locked constants for the final Event Clock-active FSR V3 traits.

These values are promotion outputs from the chronological active-trait audit.
They are separate from simulator mechanics: the FSR layer estimates fighter
state; Event Clock consumes that state without retuning its hazard formulas.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveTraitConfig:
    # Escape / retention: legacy semantics retained, but the historical V2
    # five-entry prior was under-shrunk.  Validation selected ~8 entries and
    # rejected epistemic path sampling.
    escape_prior_entries: float = 8.0
    escape_variance_multiplier: float = 0.0

    # Knockdown resistance native model:
    #   K_absorbed ~ BetaBinomial(N_sig_absorbed, p, rho)
    #   logit(p) = beta + attacker_power + age_context - resistance_defender
    #   resistance_defender ~ Normal(0, sigma^2)
    kd_resistance_rho: float = 0.005
    kd_resistance_sigma: float = 0.70
    kd_resistance_variance_multiplier: float = 1.0
    kd_resistance_grid_min: float = -2.5
    kd_resistance_grid_max: float = 2.5
    kd_resistance_grid_points: int = 501
    kd_resistance_train_state_cutoff: str = "2022-01-01"

    # Frozen historical context coefficients used during native trait
    # estimation.  They are copied here so FSR V3 does not depend on an Event
    # Clock V1 module at build time.
    kd_attacker_age_beta: float = -0.030660
    kd_defender_age_beta: float = 0.029369

    # Coordinate used only when translating the native resistance latent into
    # the frozen Event Clock KD profile field.  Event Clock itself remains
    # unchanged: beta * (rating - 50) == -native_resistance.
    frozen_event_clock_kdres_beta: float = -0.014421
