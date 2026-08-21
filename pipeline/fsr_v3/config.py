"""Locked FSR V3 constants for validated trait families.

Only parameters that have passed the historical validation work belong here.
Unvalidated V3 families must not be added as guesses.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FSRV3Config:
    # Ground striking tendency:
    #   Y ~ NB2(burst + own_CTRL/900 * q_fighter, alpha)
    #   q_fighter has Gamma shrinkage equivalent to 90 own-control seconds.
    ground_tendency_prior_seconds: float = 90.0
    ground_tendency_initial_burst: float = 2.20
    ground_tendency_initial_population_rate_15m: float = 35.0
    ground_tendency_initial_alpha: float = 1.80
    ground_tendency_q_grid_min: float = 0.001
    ground_tendency_q_grid_max: float = 400.0
    ground_tendency_q_grid_points: int = 2400

    # Ground striking suppression:
    #   mu = burst + s_defender * (own_CTRL/900 * q_attacker)
    # Gamma prior shape=2 won the multi-period robustness study.
    ground_suppression_prior_shape: float = 2.0
    ground_suppression_initial_population: float = 0.98
    ground_suppression_initial_alpha: float = 1.80
    ground_suppression_grid_min: float = 0.03
    ground_suppression_grid_max: float = 4.0
    ground_suppression_grid_points: int = 2200

    # Ground striking effectiveness:
    #   logit P(land) = beta_population + O_attacker
    #   landed|attempted ~ Beta-Binomial(rho=.08)
    #   O_attacker ~ Normal(0, .25^2)
    ground_effectiveness_rho: float = 0.08
    ground_effectiveness_sigma: float = 0.25
    ground_effectiveness_grid_min: float = -2.0
    ground_effectiveness_grid_max: float = 2.0
    ground_effectiveness_grid_points: int = 1601

    # All three validated ground traits rejected epistemic path sampling.
    ground_tendency_variance_multiplier: float = 0.0
    ground_suppression_variance_multiplier: float = 0.0
    ground_effectiveness_variance_multiplier: float = 0.0

    # Numerical safeguards.
    probability_epsilon: float = 1e-9
    minimum_positive: float = 1e-12
