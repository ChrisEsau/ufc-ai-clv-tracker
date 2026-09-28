"""Locked FSR V3 constants for validated trait families.

Only parameters that survived chronological historical validation belong here.
Aleatoric observation dispersion (NB2 alpha / Beta-Binomial rho) is distinct
from epistemic posterior uncertainty and is never sampled as a fighter trait.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FSRV3Config:
    # ------------------------------------------------------------------
    # Takedown tendency
    # Y ~ NB2(E/900 * q_fighter, alpha)
    # Gamma shrinkage K=468.48 seconds; full posterior uncertainty validated.
    # ------------------------------------------------------------------
    takedown_tendency_prior_seconds: float = 468.48
    takedown_tendency_initial_population_rate_15m: float = 5.1087
    takedown_tendency_initial_alpha: float = 0.2432
    takedown_tendency_q_grid_min: float = 0.001
    takedown_tendency_q_grid_max: float = 60.0
    takedown_tendency_q_grid_points: int = 2400
    takedown_tendency_variance_multiplier: float = 1.0

    # Takedown suppression
    # expected opponent attempts = opponent prefight q * exposure / 900
    # Y ~ NB2(expected * s_defender, alpha), s<1 suppresses.
    takedown_suppression_prior_shape: float = 8.5281
    takedown_suppression_initial_population: float = 0.9885
    takedown_suppression_initial_alpha: float = 0.4574
    takedown_suppression_grid_min: float = 0.03
    takedown_suppression_grid_max: float = 4.0
    takedown_suppression_grid_points: int = 2200
    takedown_suppression_variance_multiplier: float = 1.0

    # Takedown effectiveness
    # logit P(TD success) = beta + O_attacker - D_defender
    # Beta-Binomial rho=.12; paired Gaussian priors; posterior mean only.
    takedown_effectiveness_rho: float = 0.12
    takedown_effectiveness_sigma_offense: float = 0.35
    takedown_effectiveness_sigma_defense: float = 0.50
    takedown_effectiveness_variance_multiplier: float = 0.0

    # ------------------------------------------------------------------
    # Standing striking tendency (DISTANCE significant strikes only)
    # Y ~ NB2(E/900 * q_fighter, alpha)
    # Gamma shrinkage K=87.78 seconds; full posterior uncertainty validated.
    # ------------------------------------------------------------------
    standing_tendency_prior_seconds: float = 87.78
    standing_tendency_initial_population_rate_15m: float = 169.527
    standing_tendency_initial_alpha: float = 0.0824
    standing_tendency_q_grid_min: float = 0.001
    standing_tendency_q_grid_max: float = 600.0
    standing_tendency_q_grid_points: int = 2600
    standing_tendency_variance_multiplier: float = 1.0

    # Standing striking suppression
    # expected opponent attempts = opponent prefight q * standing exposure / 900
    # Y ~ NB2(expected * s_defender, alpha), s<1 suppresses.
    standing_suppression_prior_shape: float = 28.7138
    standing_suppression_initial_population: float = 1.0495
    standing_suppression_initial_alpha: float = 0.0863
    standing_suppression_grid_min: float = 0.03
    standing_suppression_grid_max: float = 4.0
    standing_suppression_grid_points: int = 2200
    standing_suppression_variance_multiplier: float = 1.0

    # Standing effectiveness
    # logit P(land) = beta + O_attacker - D_defender
    # Beta-Binomial rho=.035; sigma=.30/.30; posterior mean only.
    standing_effectiveness_rho: float = 0.035
    standing_effectiveness_sigma_offense: float = 0.30
    standing_effectiveness_sigma_defense: float = 0.30
    standing_effectiveness_variance_multiplier: float = 0.0

    # Shared paired-effect grid.
    effectiveness_grid_min: float = -2.0
    effectiveness_grid_max: float = 2.0
    effectiveness_grid_points: int = 1601

    # ------------------------------------------------------------------
    # Ground striking tendency
    # Y ~ NB2(burst + own_CTRL/900 * q_fighter, alpha)
    # q_fighter has Gamma shrinkage equivalent to 90 own-control seconds.
    # ------------------------------------------------------------------
    ground_tendency_prior_seconds: float = 90.0
    ground_tendency_initial_burst: float = 2.20
    ground_tendency_initial_population_rate_15m: float = 35.0
    ground_tendency_initial_alpha: float = 1.80
    ground_tendency_q_grid_min: float = 0.001
    ground_tendency_q_grid_max: float = 400.0
    ground_tendency_q_grid_points: int = 2400

    # Ground striking suppression:
    # mu = burst + s_defender * (own_CTRL/900 * q_attacker)
    # Gamma prior shape=2 won multi-period robustness.
    ground_suppression_prior_shape: float = 2.0
    ground_suppression_initial_population: float = 0.98
    ground_suppression_initial_alpha: float = 1.80
    ground_suppression_grid_min: float = 0.03
    ground_suppression_grid_max: float = 4.0
    ground_suppression_grid_points: int = 2200

    # Ground striking effectiveness:
    # logit P(land) = beta_population + O_attacker
    # Beta-Binomial rho=.08, O~N(0,.25^2); no defender trait.
    ground_effectiveness_rho: float = 0.08
    ground_effectiveness_sigma: float = 0.25
    ground_effectiveness_grid_min: float = -2.0
    ground_effectiveness_grid_max: float = 2.0
    ground_effectiveness_grid_points: int = 1601

    # All validated ground traits rejected epistemic path sampling.
    ground_tendency_variance_multiplier: float = 0.0
    ground_suppression_variance_multiplier: float = 0.0
    ground_effectiveness_variance_multiplier: float = 0.0

    # ------------------------------------------------------------------
    # Striking power
    # K ~ BetaBinomial(N landed significant strikes, p, rho)
    # logit(p) = beta_population + attacker_power
    # attacker_power ~ Normal(0, .50^2)
    # Selected sequential model: rho=.01, c=0 (posterior mean only).
    # No KO-win bonus and no age baked into the persisted trait.
    # ------------------------------------------------------------------
    power_sigma: float = 0.50
    power_rho: float = 0.01
    power_variance_multiplier: float = 0.0
    power_grid_min: float = -4.0
    power_grid_max: float = 4.0
    power_grid_points: int = 321
    power_train_state_cutoff: str = "2020-01-01"

    # Numerical safeguards.
    probability_epsilon: float = 1e-9
    minimum_positive: float = 1e-12
