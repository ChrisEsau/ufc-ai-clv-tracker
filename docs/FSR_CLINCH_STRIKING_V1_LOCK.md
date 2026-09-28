# FSR Clinch Striking V1 Lock

Status: LOCKED for shadow simulator use

## Traits

- `clinch_striking_pressure`
- `clinch_striking_precision`
- `clinch_striking_defense`

## Elo framework

All three traits use the established FSR update form:

`R_new = R_old + K * Q * (O - E)`

with:

- base rating 50
- bounds 10-90
- decaying K factor
- prior-date population pools/baselines only
- same-date simultaneous updates
- matchup scale 12 for paired precision/defense ratings

## Pressure observation

`clinch_striking_pressure` is a population-centered intrinsic tendency.

Observation:

- 60% percentile of `clinch_attempts_per_round`
- 40% percentile of `clinch_attempt_share`

Quality:

`Q = q_exp(clinch_attempts / 10)`

No observed clinch attempts means no pressure update.

## Precision observation

`clinch_striking_precision` uses the prior-date population percentile of clinch striking accuracy.

Quality:

`Q = q_exp(clinch_attempts / 10)`

Expected value is fighter precision versus opponent clinch striking defense.

## Defense observation

`clinch_striking_defense` uses the complement of the prior-date population percentile of opponent clinch accuracy allowed.

Quality:

`Q = q_exp(opponent_clinch_attempts / 10)`

Expected value is fighter defense versus opponent clinch striking precision.

## Integrity audit

FSR-25 build:

- 13,390 fighter-fight pre-fight rows
- exact key parity with FSR-22
- no existing FSR-22 columns changed

Observed ranges:

- pressure: 43.8876 to 57.4606, mean 49.9188
- precision: 43.1523 to 58.1411, mean 50.1237
- defense: 44.2712 to 55.3251, mean 50.0648

## Pressure calibration

Historical realized clinch attempts per round by pre-fight pressure bucket:

| Pressure bucket | Rows | Mean clinch attempts/round |
|---|---:|---:|
| <47 | 169 | 1.4123 |
| 47-49 | 2,155 | 1.8813 |
| 49-51 | 9,050 | 2.7280 |
| 51-53 | 1,733 | 3.7536 |
| 53-55 | 257 | 4.4673 |
| 55-57 | 24 | 6.0097 |
| 57+ | 2 | 8.1667 |

Spearman pressure versus realized clinch attempts/round: 0.1583.

Interpretation: pressure is validated as a simulator input for clinch-striking volume propensity. Sparse extreme buckets should not be over-interpreted individually.

## Precision-defense calibration

Attempt-weighted historical clinch accuracy by pre-fight edge:

`clinch_striking_precision - opponent clinch_striking_defense`

| Edge bucket | Rows | Weighted accuracy |
|---|---:|---:|
| <-4 | 279 | 0.6475 |
| -4 to -2 | 1,010 | 0.6615 |
| -2 to 0 | 3,229 | 0.6874 |
| 0 to 2 | 4,295 | 0.7140 |
| 2 to 4 | 1,120 | 0.7468 |
| 4+ | 313 | 0.7751 |

Fight-level Spearman edge versus clinch accuracy: 0.1034.

Interpretation: the edge provides a clean monotonic landing-probability calibration with roughly a 13 percentage-point swing from the lowest to highest bucket.

## Simulator interpretation

- `clinch_striking_pressure` -> clinch significant-strike attempt propensity
- `clinch_striking_precision` versus opponent `clinch_striking_defense` -> clinch strike landing probability
- downstream damage remains handled by the existing power/durability ratings

This family does not claim exact clinch duration, clinch entry probability, or clinch separation probability. UFCStats does not directly observe exact phase entries/exits or exact clinch exposure time.

## Change control

Any semantic or formula change to these traits requires a new version rather than silently modifying V1.
