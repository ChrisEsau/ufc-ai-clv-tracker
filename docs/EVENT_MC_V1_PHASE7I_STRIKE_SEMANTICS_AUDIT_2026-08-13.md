# Phase 7I Strike Semantics Audit

## Code semantics

- `strike`: `DistanceActionRateProvider` schedules one `DistanceCandidate` per attempt from `strike_attempt_rate_per_second`, controlled by `distance_striking_pressure` and stamina output. `DistanceCandidate.resolve` performs one Bernoulli landing draw from `strike_landing_probability`, controlled by attacker distance precision and defender distance defense.
- `clinch_strike`: `FightFlowRateProvider` schedules one `PhaseCandidate` per attempt from `phase_strike_rate_per_second`, controlled by `clinch_striking_pressure` and output. Landing is one Bernoulli draw using clinch precision/defense.
- `ground_strike`: the same phase candidate path uses `ground_striking_pressure`; bottom attempts additionally use the configured bottom strike multiplier. Landing uses ground precision/defense.
- Every landed family emits one `ActionOutcome` and is eligible for the same physiology impact/trauma/KD/finish chain. There is no low-impact/non-significant strike family, target location, or head/body/leg representation. Phase is explicit, but strike severity is sampled only after landing.

Therefore an EVENT MC strike is best interpreted as a **meaningful offensive strike opportunity**, not literally every physical UFCStats total strike. The closest available historical comparator is UFCStats significant strikes, though it is not definition-identical. This conclusion applies to all phases; clinch/ground accuracy resembles significant-strike accuracy especially closely.

## Historical fields

The master exposes total attempts/landed and significant attempts/landed. It also exposes clinch and ground significant attempts/landed. Distance significant strikes are the significant total residual after clinch and ground. UFCStats/master does not expose phase-specific TOTAL strikes or trustworthy historical time-in-phase denominators, so no phase-time rate was fabricated.

## Cohorts and measured baselines

The audit used corrected total-elapsed fight duration, seed `20260813`, and 10 paths per fight. Train contains 100 fights from 2020-01-18 through 2020-07-25 (1,000 simulated paths); holdout contains 50 fights from 2025-01-11 through 2025-03-22 (500 paths).

### Historical strike aggregates

| Split / field | Attempts/fight | Landed/fight | Attempts/15m | Landed/15m | Accuracy |
|---|---:|---:|---:|---:|---:|
| Train TOTAL | 240.34 | 132.12 | 285.681 | 157.045 | 54.97% |
| Train significant | 200.37 | 96.57 | 238.170 | 114.788 | 48.20% |
| Holdout TOTAL | 266.50 | 143.72 | 311.988 | 168.251 | 53.93% |
| Holdout significant | 219.18 | 104.02 | 256.591 | 121.775 | 47.46% |

### Historical significant strikes by phase

Rates below use whole-fight elapsed time, not unavailable phase-specific time.

| Split / phase | Att/fight | Land/fight | Att/15m | Land/15m | Accuracy | Att share | Land share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train distance | 172.95 | 75.99 | 205.577 | 90.326 | 43.94% | 86.32% | 78.69% |
| Train clinch | 13.91 | 9.96 | 16.534 | 11.839 | 71.60% | 6.94% | 10.31% |
| Train ground | 13.51 | 10.62 | 16.059 | 12.623 | 78.61% | 6.74% | 11.00% |
| Holdout distance | 196.90 | 87.14 | 230.508 | 102.014 | 44.26% | 89.83% | 83.77% |
| Holdout clinch | 10.76 | 8.24 | 12.597 | 9.646 | 76.58% | 4.91% | 7.92% |
| Holdout ground | 11.52 | 8.64 | 13.486 | 10.115 | 75.00% | 5.26% | 8.31% |

### EVENT MC modeled strikes

| Split / family | Att/path | Land/path | Att/15m | Land/15m | Accuracy | Att share | Land share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train all | 165.779 | 73.480 | 199.307 | 88.341 | 44.32% | 100.00% | 100.00% |
| Train distance | 152.465 | 64.175 | 183.300 | 77.154 | 42.09% | 91.97% | 87.34% |
| Train clinch | 4.556 | 3.140 | 5.477 | 3.775 | 68.92% | 2.75% | 4.27% |
| Train ground | 8.758 | 6.165 | 10.529 | 7.412 | 70.39% | 5.28% | 8.39% |
| Holdout all | 163.366 | 76.828 | 198.629 | 93.411 | 47.03% | 100.00% | 100.00% |
| Holdout distance | 150.922 | 68.042 | 183.499 | 82.729 | 45.08% | 92.38% | 88.56% |
| Holdout clinch | 3.608 | 2.580 | 4.387 | 3.137 | 71.51% | 2.21% | 3.36% |
| Holdout ground | 8.836 | 6.206 | 10.743 | 7.546 | 70.24% | 5.41% | 8.08% |

### Comparator reconciliation

| Split / comparator | Historical att/15m | Model att/15m | Ratio | Historical land/15m | Model land/15m | Ratio | Historical/model accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train TOTAL | 285.681 | 199.307 | 69.77% | 157.045 | 88.341 | 56.25% | 54.97% / 44.32% |
| Train significant | 238.170 | 199.307 | 83.68% | 114.788 | 88.341 | 76.96% | 48.20% / 44.32% |
| Holdout TOTAL | 311.988 | 198.629 | 63.67% | 168.251 | 93.411 | 55.52% | 53.93% / 47.03% |
| Holdout significant | 256.591 | 198.629 | 77.41% | 121.775 | 93.411 | 76.71% | 47.46% / 47.03% |

## Outcome guardrails

| Split | Historical KO/SUB/DEC | Simulated KO/SUB/DEC | KD/path | KD/100 modeled landed | KD/15m | Mean duration | Mean nondecision finish |
|---|---|---|---:|---:|---:|---:|---:|
| Train | 25.0% / 17.0% / 58.0% | 23.8% / 17.3% / 58.9% | 0.322 | 0.438 | 0.387 | 748.60s | 403.16s |
| Holdout | 28.0% / 18.0% / 54.0% | 25.4% / 18.2% / 56.4% | 0.264 | 0.344 | 0.321 | 740.22s | 387.67s |

## Recommendation

Use historical significant-strike attempts as the primary attempt-generation target and significant-strike landing percentage as the primary landing target. Preserve total strikes as a secondary upper-bound diagnostic. Current modeled attempt exposure remains low versus significant strikes (train 83.7%, holdout 77.4%), while overall landing accuracy is close (train 44.3% vs 48.2%; holdout 47.0% vs 47.5%). The main discrepancy is attempt frequency plus phase composition: modeled clinch share is materially low and distance share high; landing probability is secondary, and total-strike comparisons contain a large field-definition mismatch.

Distance, clinch, and ground should all use significant strikes as their closest comparator. This is not a claim of exact identity: all modeled landed strikes enter the damage pipeline, while UFCStats significance is an external coding definition. Clinch exposure is particularly deficient; ground share is closer, and distance dominates too strongly. No calibration is promoted by this audit.
