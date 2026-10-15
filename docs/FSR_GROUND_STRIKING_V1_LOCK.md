# FSR Ground Striking V1 — Locked Shadow Contract

Status: LOCKED for FSR-21 shadow simulator evaluation.

This contract extends the existing FSR-18 snapshot with three leakage-safe Elo-style persistent fighter ratings. It does not modify the locked 13 FSR ratings or the five dynamic-response ratings.

## Ratings

1. `ground_striking_pressure`
2. `ground_striking_precision`
3. `ground_striking_defense`

All three use the established FSR update form:

`R_new = R_old + K * Q * (O - E)`

with:

- base rating: 50
- bounds: 10–90
- rating scale: 12
- base K: 7
- declining K: `7 / sqrt(1 + update_count / 6)`
- prior-date population pools and baselines only
- simultaneous same-date updates
- fighter-fight pre-fight snapshots

## Ground Striking Pressure

Purpose: represent how aggressively a fighter generates significant ground strikes when ground offense is available.

Observation:

- 60% prior-population percentile of `rfs_phase_base_fight_ground_attempts_per_round`
- 40% prior-population percentile of `rfs_phase_base_fight_ground_attempt_share`

Quality:

`Q = 1 - exp(-(ground_attempts / 10))`

No pressure update occurs when no ground strike attempt is observed.

Expectation is population-centered intrinsic Elo expectation; it is not opponent-adjusted.

## Ground Striking Precision

Purpose: represent how efficiently a fighter lands significant ground strikes.

Observation:

Prior-population percentile of `rfs_phase_interact_fight_ground_accuracy`.

Quality:

`Q = 1 - exp(-(ground_attempts / 10))`

Expectation is opponent-adjusted against the opponent's pre-fight `ground_striking_defense` rating.

## Ground Striking Defense

Purpose: represent how well a fighter suppresses an opponent's significant-ground-strike accuracy.

Observation:

`1 - percentile(rfs_phase_interact_fight_ground_accuracy_allowed)`

Quality:

`Q = 1 - exp(-(opponent_ground_attempts / 10))`

Expectation is opponent-adjusted against the opponent's pre-fight `ground_striking_precision` rating.

## FSR-21 Integrity Check

The incremental FSR-21 build produced 13,390 pre-fight fighter-fight rows and preserved all existing FSR-18 values unchanged.

Observed rating ranges in the FSR-21 historical replay:

- `ground_striking_pressure`: 46.6614–56.3158; mean 50.1944
- `ground_striking_precision`: 44.5571–56.9554; mean 50.0958
- `ground_striking_defense`: 44.4375–56.6035; mean 49.9869

## Empirical Acceptance Evidence

### Pressure → realized ground volume

Whole-fight pressure buckets showed a monotonic rise through the well-populated range:

- mean pressure 48.57 → 1.84 ground attempts/round
- mean pressure 49.96 → 2.88
- mean pressure 51.64 → 3.86
- mean pressure 53.63 → 4.62

Spearman: 0.0962.

Conditioning on meaningful grappling/ground access strengthened the simulator interpretation:

- mean pressure 48.58 → 2.56 ground attempts/round
- mean pressure 49.97 → 3.74
- mean pressure 51.66 → 4.88
- mean pressure 53.64 → 5.68

Spearman: 0.0936.

The weak single-fight rank correlation is accepted because realized GNP volume is heavily gated by whether ground position is reached; the bucket calibration is monotonic across the populated range.

### Precision minus opponent defense → realized ground accuracy

Attempt-weighted realized accuracy by pre-fight edge:

- edge < -4: 64.91%
- -4 to -2: 69.00%
- -2 to 0: 69.56%
- 0 to 2: 70.67%
- 2 to 4: 75.32%
- 4+: 76.80%

The roughly 12 percentage-point monotonic swing is accepted for simulator calibration. The rating edge should modify a base ground-strike accuracy rather than directly equal the landing probability.

## Simulator Interpretation

`ground_striking_pressure`
→ ground significant-strike attempt propensity after ground access

`ground_striking_precision - opponent ground_striking_defense`
→ calibrated adjustment to ground-strike landing probability

Existing damage traits such as `striking_power`, `chin_resistance`, and `damage_resistance`
→ consequences of landed offense

## Boundary

UFCStats does not provide exact ground-position exposure seconds. Do not reinterpret control seconds as exact ground time or redefine pressure as strikes per exact ground minute without a separately justified latent exposure model.

This V1 contract is locked for shadow use. Any formula, weighting, quality, expectation, or semantic change requires an explicit new version rather than silently changing V1.
