# FSR Reversal Ability V1 Lock

Status: **LOCKED FOR SHADOW SIMULATOR USE**

## Trait

`reversal_ability`

Purpose: persistent Elo-style fighter rating for the ability/tendency to reverse position while under opponent control.

## Source evidence

- `rfs_phase_interact_fight_reversals`
- `rfs_phase_interact_fight_opp_control_seconds`
- opponent pre-fight `control_imposition`

UFCStats reversals are directly observed and carried into RFS. Opponent control time is treated as the opportunity exposure.

## Update semantics

All updates use the established FSR form:

`R_new = R_old + K * Q * (O - E)`

with:

- base rating 50
- bounds 10 to 90
- rating scale 12
- `K = 7 / sqrt(1 + update_count / 6)`
- leakage-safe chronological replay
- prior-date population baselines
- simultaneous same-date updates

### Observation

If opponent control seconds are zero:

- no reversal opportunity
- no update

If opponent control exists and reversals are zero:

- observation = 0.0
- explicit negative evidence

If one or more reversals occur:

- compute reversals per 15 minutes of opponent control
- rank that positive rate against the prior-date positive-rate population
- observation = `0.60 + 0.40 * positive_rate_percentile`

### Quality

`Q = 1 - exp(-(opp_control_seconds / 180 + reversals))`

Thus long opponent-control exposure makes a zero-reversal fight more informative, while observed reversals add event evidence.

### Matchup expectation

`reversal_ability` is opposed by the opponent's `control_imposition` rating.

## Validation

FSR-22 build:

- 13,390 pre-fight fighter rows
- identical fighter-fight key set to FSR-21
- no existing FSR-21 columns changed
- observed pre-fight rating range: 43.7262 to 66.0104
- mean rating: 50.1245

Historical reversal opportunity volume:

- 13,390 fighter-fight rows
- 11,208 rows with opponent control
- 1,396 rows with at least one reversal
- 1,705 total recorded reversals
- 12.429% of opponent-control rows contained at least one reversal

### Matchup calibration

Using:

`reversal_edge = reversal_ability - opponent_control_imposition`

conditional on opponent control:

| Edge bucket | Rows | Pct with reversal | Mean reversals |
|---|---:|---:|---:|
| <-6 | 335 | 8.66% | 0.0985 |
| -6 to -3 | 1,565 | 9.97% | 0.1163 |
| -3 to 0 | 4,361 | 11.63% | 0.1378 |
| 0 to 3 | 3,459 | 13.24% | 0.1657 |
| 3 to 6 | 1,073 | 15.38% | 0.1957 |
| 6+ | 415 | 18.80% | 0.2458 |

Spearman correlations are weak at the individual-fight level because reversals are sparse events:

- edge vs any reversal: 0.0543
- edge vs reversal count: 0.0554

The bucket calibration is monotonic and materially separates reversal probability, so the trait is accepted for simulator transition calibration.

## Simulator interpretation

When a fighter is under opponent control:

1. `control_resistance - opponent control_imposition` calibrates control-break / escape pressure.
2. If control breaks, `reversal_ability - opponent control_imposition` calibrates the chance that the break becomes a reversal rather than a simple escape/separation.

## Lock rule

Do not silently change the observation, quality, opponent mapping, or semantics of `reversal_ability` in V1. Any material change requires a new version and revalidation.
