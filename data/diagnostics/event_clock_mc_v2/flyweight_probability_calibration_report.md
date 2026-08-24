# Men's Flyweight Probability Calibration Report

Research branch: `research/weight-class-audit-20260823`
Source run: `32713718959`
Source artifact: `event-clock-v2-flyweight-decay-shape-screen`
Candidate mechanics held fixed for this report:

- power offset: `clip(15 - t/12, -40, 15)`
- KD finishing sequence: OFF
- cohort: 46 eligible men's flyweight fights
- simulation paths: 20/fight
- seed: 20260823

## Overall discrimination/calibration

- ML accuracy: 67.39%
- ML Brier: 0.23087
- ML log loss: 0.65640
- ML AUC: 0.66472
- mean actual-winner probability: 55.00%
- method accuracy: 47.83%
- method multiclass Brier: 0.55196
- method log loss: 0.89328
- mean actual-method probability: 43.91%

## ML confidence calibration

| Predicted-winner confidence | N | Mean predicted | Actual win rate | Gap (pred-actual) |
|---|---:|---:|---:|---:|
| 50-60% | 18 | 52.78% | 66.67% | -13.89pp |
| 60-70% | 10 | 62.00% | 60.00% | +2.00pp |
| 70-80% | 12 | 71.67% | 75.00% | -3.33pp |
| 80-90% | 5 | 82.00% | 60.00% | +22.00pp |
| 90-100% | 1 | 90.00% | 100.00% | -10.00pp |

Interpretation: the model has useful ordering signal, but confidence calibration is irregular. The clearest concern is the small 80-90% bucket, where 82% mean confidence converted at only 60%. The 50-60% bucket is underconfident. With only 46 fights these bucket estimates are noisy, so this is a diagnostic signal rather than a fitted calibration map.

## Method confidence calibration

| Predicted-method confidence | N | Mean predicted | Actual accuracy | Gap |
|---|---:|---:|---:|---:|
| 34-45% | 4 | 40.00% | 25.00% | +15.00pp |
| 45-55% | 17 | 47.06% | 35.29% | +11.76pp |
| 55-65% | 16 | 57.50% | 56.25% | +1.25pp |
| 65-75% | 7 | 67.14% | 57.14% | +10.00pp |
| 75-100% | 2 | 80.00% | 100.00% | -20.00pp |

Method probabilities are generally more overconfident than ML probabilities, especially in the 45-55% and 65-75% bands.

## Method-specific probability calibration

### DEC

| Predicted DEC band | N | Mean predicted | Actual DEC rate |
|---|---:|---:|---:|
| 10-20% | 2 | 12.50% | 0.00% |
| 20-30% | 2 | 22.50% | 50.00% |
| 30-40% | 4 | 32.50% | 50.00% |
| 40-50% | 9 | 43.89% | 44.44% |
| 50-60% | 15 | 52.67% | 46.67% |
| 60-70% | 10 | 62.00% | 60.00% |
| 70%+ | 4 | 75.00% | 75.00% |

DEC probability is reasonably calibrated once predicted DEC is above roughly 40%, and is the strongest of the three method probabilities.

### KO/TKO

| Predicted KO band | N | Mean predicted | Actual KO rate |
|---|---:|---:|---:|
| 0-10% | 2 | 5.00% | 0.00% |
| 10-20% | 10 | 13.50% | 0.00% |
| 20-30% | 15 | 21.67% | 26.67% |
| 30-40% | 11 | 31.82% | 36.36% |
| 40-50% | 6 | 42.50% | 33.33% |
| 60-70% | 2 | 60.00% | 50.00% |

KO probabilities show useful monotonic structure in the 20-40% region. The 10-20% bucket produced no historical KOs in this sample. Higher buckets are too sparse to draw a strong conclusion.

### Submission

| Predicted SUB band | N | Mean predicted | Actual SUB rate |
|---|---:|---:|---:|
| 0-10% | 4 | 3.75% | 0.00% |
| 10-20% | 9 | 12.22% | 0.00% |
| 20-30% | 18 | 22.50% | 16.67% |
| 30-40% | 9 | 32.78% | 55.56% |
| 40-50% | 5 | 41.00% | 60.00% |
| 70%+ | 1 | 70.00% | 100.00% |

Submission is the clearest method-discrimination weakness. The model places many real submissions in the 30-50% range but does not separate them strongly enough from decisions. This is consistent with the method confusion matrix below.

## Method confusion matrix

Rows are historical method; columns are model-predicted method.

| Actual \\ Predicted | DEC | KO/TKO | SUB |
|---|---:|---:|---:|
| DEC | 19 | 3 | 1 |
| KO/TKO | 8 | 2 | 1 |
| SUB | 10 | 1 | 1 |

The dominant failure mode is **predicting DEC for actual finishes**:

- 8 of 11 historical KO/TKOs were predicted DEC.
- 10 of 12 historical submissions were predicted DEC.

This explains why aggregate method shares can look close while fight-level method discrimination remains weak. The simulator is getting the population-level mix approximately right without identifying which specific fights finish.

## High-confidence ML misses (predicted winner >=65%)

| Fight | Model pick | Confidence | Actual winner | Actual method |
|---|---|---:|---|---|
| Joshua Van vs Tatsuro Taira | Tatsuro Taira | 85% | Joshua Van | KO/TKO |
| Tatsuro Taira vs HyunSung Park | HyunSung Park | 80% | Tatsuro Taira | SUB |
| Andre Lima vs Kevin Borjas | Andre Lima | 75% | Kevin Borjas | DEC |
| Jafel Filho vs Allan Nascimento | Jafel Filho | 70% | Allan Nascimento | DEC |
| Tagir Ulanbekov vs Kyoji Horiguchi | Tagir Ulanbekov | 70% | Kyoji Horiguchi | SUB |
| Charles Johnson vs Lone'er Kavanagh | Lone'er Kavanagh | 65% | Charles Johnson | KO/TKO |
| Brandon Moreno vs Lone'er Kavanagh | Brandon Moreno | 65% | Lone'er Kavanagh | DEC |
| Alex Perez vs Asu Almabayev | Alex Perez | 65% | Asu Almabayev | SUB |

## High-confidence method misses (predicted method >=50%)

There are 14 such misses. Twelve are model DEC predictions on fights that actually finished. Examples include:

- Charles Johnson vs Lone'er Kavanagh: 70% DEC, actual KO/TKO.
- Tagir Ulanbekov vs Kyoji Horiguchi: 65% DEC, actual SUB.
- Steve Erceg vs Ramazan Temirov: 65% DEC, actual KO/TKO.
- Tim Elliott vs Kai Asakura: 60% DEC, actual SUB.
- Asu Almabayev vs Charles Johnson: 55% DEC, actual SUB.

## Research conclusion

For men's flyweight, the current `15/12, sequence off` mechanics are close on aggregate DEC/KO/SUB base rates, but fight-level method discrimination is still poor. The next tuning target should **not** be another global KO-power adjustment.

The largest actionable miss is that the simulator does not identify which fights are submission/KO-prone versus decision-prone. Specifically, SUB probability is compressed and many actual submissions remain classified as DEC. ML discrimination is moderate (AUC ~0.665), so there is also room for stronger fighter-vs-fighter separation, but method discrimination is currently the clearer structural weakness.

Recommended next diagnostic: measure prefight FSR/matchup trait deltas separately for (a) actual finish vs decision and (b) actual winner, then test whether those traits contain out-of-sample signal that the simulator is failing to express. Do not recalibrate output probabilities before checking whether the missing discrimination exists upstream.
