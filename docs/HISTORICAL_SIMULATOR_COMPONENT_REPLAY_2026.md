# Historical Simulator Component Replay — 2026 Holdout

## Status

**Shadow-only validation. The learned finish provider shows measurable holdout value. The strike provider remains blocked, and no simulator output is approved for wagering or production.**

## Replay design

The replay scores completed, supported UFC fights in the 2026 holdout through the current authoritative snapshot.

- Scoreable fights: 298
- Monte Carlo paths: 500 per fight and simulator variant
- Holdout paths per variant: 149,000
- Fighter state: fights completed before the target matchup only
- Finish model training: fights before each walk-forward holdout year only
- Finish probability calibration: earlier walk-forward years only
- Counterfactual finish coverage: every scheduled round, including rounds after the actual historical finish
- Realized winner and method: joined after state and feature construction for scoring only
- Elapsed fight time: repaired and validated against `target_elapsed_fight_seconds`
- Excluded from scoring: draws, no-contests, overturned/unsupported methods, and incomplete winner/time rows

The original mechanics engine remains unchanged. Provider variants run through separate shadow paths.

## Evaluation correction

The first replay joined raw master `match_time_sec`. Historical master rows can store either total elapsed time or the clock inside the final round. That made the original duration comparison inconsistent.

The corrected replay:

1. repairs master time with `pipeline.common.fight_time.repair_elapsed_match_time`;
2. verifies exact fight-level agreement with the leakage-safe training target `target_elapsed_fight_seconds`;
3. rejects inconsistent labels;
4. uses the validated elapsed value for scoring.

The corrected actual mean fight time is 635.49 seconds, not 487.50 seconds.

## Simulator variants

### 1. Heuristic simulator

Original V0 mechanics for strike attempts and finish hazards.

### 2. Class finish-hazard provider

A pre-fight five-class competing-risk model predicts, for each round:

- no finish;
- red KO/TKO;
- red submission;
- blue KO/TKO;
- blue submission.

Sequential class calibration uses only earlier walk-forward years.

### 3. Round-survival finish provider

Adds scheduled-round/round-specific terminal-odds calibration using prior walk-forward predictions. It changes only terminal-event mass and preserves the learned conditional red/blue and KO/submission mix.

### 4. Strike plus survival providers

Combines the round-survival finish provider with the static pre-fight absolute strike-rate ablation. This strike provider is deliberately not the final dynamic trained component.

## Corrected 2026 results

| Model | Winner Brier | Winner accuracy | Method log loss | Method accuracy | Distance Brier | Time MAE | Time bias | Strike MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heuristic simulator | 0.2463 | 55.37% | 1.3411 | 35.91% | 0.3204 | 346.8 sec | -182.6 sec | 57.26 |
| Class finish provider | 0.2403 | 60.40% | 1.0118 | 50.00% | 0.2426 | 302.9 sec | +27.2 sec | 54.02 |
| Round-survival finish provider | **0.2389** | **60.40%** | 1.0088 | 50.00% | **0.2424** | **302.7 sec** | +30.8 sec | **53.98** |
| Strike + survival providers | 0.2419 | 58.05% | **1.0078** | **50.00%** | 0.2440 | 303.8 sec | +30.8 sec | 56.01 |
| Historical baseline | 0.2451 | 57.05% | 1.0405 | 46.31% | 0.2489 | 315.8 sec | +18.7 sec | 56.38 |

## Improvement over historical baseline

### Round-survival finish provider

- Winner Brier: 2.57% better
- Winner accuracy: +3.36 percentage points
- Method log loss: 3.05% better
- Method accuracy: +3.69 percentage points
- Goes-distance Brier: 2.60% better
- Fight-time MAE: 4.15% better
- Fighter strike-attempt MAE: 4.26% better

### Strike plus survival providers

- Method log loss: 3.15% better
- Method accuracy: +3.69 percentage points
- Fight-time MAE: 3.82% better
- Strike-attempt MAE: only 0.66% better
- Winner Brier is worse than the finish-only provider

The finish-only path is the stronger general simulator candidate. The static strike provider does not clear a promotion gate.

## Aggregate calibration

| Quantity | Actual | Heuristic | Class finish | Survival finish | Strike + survival |
|---|---:|---:|---:|---:|---:|
| Red win rate | 57.05% | 51.46% | 55.19% | 54.99% | 55.18% |
| Decision rate | 45.30% | 16.93% | 48.75% | 50.25% | 50.34% |
| KO/TKO rate | 35.91% | 73.19% | 33.37% | 32.41% | 32.26% |
| Submission rate | 18.79% | 9.88% | 17.87% | 17.34% | 17.40% |
| Mean fight time | 635.49 sec | 452.90 sec | 662.73 sec | 666.28 sec | 666.27 sec |
| Mean fighter strike attempts | 86.02 | 45.30 | 71.18 | 71.59 | 92.70 |

The learned finish provider removes the original catastrophic KO/TKO inflation and produces realistic decision and submission rates. Remaining biases are much smaller:

- decisions are approximately 5 percentage points high;
- KO/TKO is approximately 3.5 percentage points low;
- submissions are approximately 1.4 percentage points low;
- mean duration is approximately 31 seconds high.

## Round-survival factors

The sequential survival factors are close to one for three-round rounds 1 and 2. Later-round factors reduce terminal odds because the class-calibrated model slightly overpredicts late-round finishes.

| Scheduled rounds | Round | Prior rows | Actual terminal rate | Predicted terminal rate | Odds factor |
|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 1,504 | 25.29% | 25.21% | 1.0043 |
| 3 | 2 | 1,119 | 23.06% | 22.44% | 1.0359 |
| 3 | 3 | 860 | 12.89% | 15.69% | 0.7954 |
| 5 | 1 | 178 | 20.60% | 23.49% | 0.8450 |
| 5 | 2 | 149 | 20.66% | 21.16% | 0.9703 |
| 5 | 3 | 122 | 12.63% | 14.92% | 0.8244 |
| 5 | 4 | 106 | 13.21% | 13.88% | 0.9443 |
| 5 | 5 | 92 | 7.61% | 13.78% | 0.5154 |

Round-survival calibration produces a small improvement over class-only calibration, not a fundamental change. This confirms that the competing-risk model itself fixed the principal finish problem.

## Promotion decision

### Finish provider

**Passes the research replay gate, but remains shadow-only.**

It improves every primary probabilistic task over the historical baseline and corrects the original method-distribution failure. Before production consideration it still requires:

- repeated walk-forward full-fight replay across several holdout years;
- subgroup stability by division, scheduled rounds, title status, and fighter experience;
- direct comparison with existing moneyline and prop models;
- probability calibration plots and confidence intervals;
- ROI and CLV evaluation against closing markets;
- treatment of DQ, doctor stoppage, and other terminal methods.

### Static strike provider

**Blocked.**

It raises aggregate attempts from 71.59 to 92.70 but does not improve fighter-level strike MAE enough and weakens winner performance. The final strike provider must use the trained component with simulated prior-round context rather than a static pre-fight career rate.

## Next implementation priority

1. Retain the round-survival finish provider as the leading shadow component.
2. Run full-fight walk-forward replays for 2022–2026, not only the 2026 snapshot.
3. Add subgroup and uncertainty gates for the finish provider.
4. Build a dynamic strike provider that reconstructs round-2+ model features from simulated prior rounds.
5. Do not merge or promote to live wagering until the complete provider stack passes historical probability, ROI, and CLV gates.
