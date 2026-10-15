# FSR Distance Striking V1 Lock

Status: LOCKED for shadow/research use.

This document records the validated distance-striking FSR family used by the RFS Monte Carlo profile architecture.

## Canonical traits

1. `distance_striking_pressure`
2. `distance_striking_precision`
3. `distance_striking_defense`

For compatibility with existing downstream consumers, the historical columns `distance_precision` and `distance_defense` remain unchanged and are carried forward as aliases of the canonical precision/defense traits.

## Distance striking pressure

Observation:

- 60% prior-date population percentile of `distance_attempts_per_round`
- 40% prior-date population percentile of `distance_attempt_share`

Only updates when observed distance attempts are positive.

Quality weight:

`Q = 1 - exp(-distance_attempts / 10)`

Expectation is population-centered/intrinsic rather than opponent-adjusted.

Meaning: when operating at distance, how strongly a fighter tends to generate significant-strike activity and orient offense toward distance striking.

### Validation

Leakage-safe pre-fight FSR-26 snapshots vs realized current-fight distance attempts per round:

| Pressure bucket | Rows | Mean pressure | Mean distance attempts/round |
|---|---:|---:|---:|
| <45 | 386 | 43.8368 | 18.7135 |
| 45-48 | 2,261 | 46.8562 | 22.2615 |
| 48-51 | 6,752 | 49.6772 | 25.9460 |
| 51-54 | 2,825 | 52.1874 | 33.8402 |
| 54-57 | 912 | 55.2262 | 40.1455 |
| 57-60 | 219 | 58.0639 | 46.2059 |
| 60+ | 35 | 61.2392 | 55.9143 |

Spearman = **0.2654**.

Conclusion: strong, monotonic forward calibration. `distance_striking_pressure` is validated for simulator distance-strike volume/activity.

## Distance striking precision / defense

These are the existing locked distance accuracy ratings exposed under canonical phase-symmetric names.

Matchup edge:

`distance_striking_precision - opponent distance_striking_defense`

### Attempt-weighted accuracy validation

| Edge bucket | Rows | Mean edge | Total landed | Total attempts | Weighted accuracy |
|---|---:|---:|---:|---:|---:|
| <-6 | 462 | -7.9930 | 13,094 | 41,376 | 0.3165 |
| -6 to -3 | 1,385 | -4.2233 | 38,829 | 112,425 | 0.3454 |
| -3 to 0 | 3,473 | -1.3380 | 98,723 | 261,135 | 0.3781 |
| 0 to 3 | 4,993 | 1.1442 | 140,141 | 344,648 | 0.4066 |
| 3 to 6 | 1,957 | 4.2583 | 67,215 | 151,937 | 0.4424 |
| 6+ | 1,047 | 8.3455 | 44,983 | 90,277 | 0.4983 |

Fight-level Spearman edge vs accuracy = **0.3207**.

Conclusion: extremely clean monotonic calibration across the full matchup edge range. The precision/defense pair is validated for simulator distance landing probability.

## FSR-26 integrity

- FSR-25 rows: 13,390
- FSR-26 rows: 13,390
- identical fighter-fight key set
- no pre-existing FSR-25 columns changed
- distance pressure range: 38.8103 to 63.4524
- distance precision range: 39.5073 to 63.4794
- distance defense range: 40.1755 to 60.9068

## Simulator mapping

Within a distance phase:

`distance_striking_pressure`
→ strike attempt propensity/volume

`distance_striking_precision - opponent distance_striking_defense`
→ strike landing probability

Landed strikes then flow into existing power/durability machinery.

## Architecture note

Distance, clinch, and ground striking now share the same conceptual structure:

- phase striking pressure
- phase striking precision
- phase striking defense

The next simulator-design problem is phase-choice/state-transition logic. Do not introduce additional phase-choice ratings until existing FSR-26 traits are tested as a derived phase-preference/transition system.
