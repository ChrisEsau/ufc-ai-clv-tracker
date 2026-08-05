# Historical Trained-Component Simulator Replay — 2026

## Status

Shadow-only diagnostic. No production prediction, betting, feature-view, ingestion,
or master-data artifact is modified.

## Replay design

- Holdout: 298 scoreable completed fights from 2026.
- Monte Carlo paths: 500 per fight and variant.
- Variants: original heuristic engine, absolute pre-fight strike pace only, and
  absolute strike pace plus trained competing-risk finish hazards.
- Total simulated paths: 447,000.
- Fighter state and calibration inputs use information available before the
  target holdout.
- The finish model is trained on pre-2026 rows and calibrated using completed
  earlier walk-forward holdouts.
- Draws, no-contests, unsupported methods, and incomplete outcomes are excluded
  from scoring only.

## Results

| Variant | Winner Brier | Winner log loss | Method log loss | Distance Brier | Time MAE | Strike MAE |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic | 0.2463 | 0.6860 | 1.3411 | 0.3204 | 304.7 sec | 57.26 |
| Absolute strike only | 0.2465 | 0.6866 | 1.6615 | 0.3752 | 296.3 sec | 56.09 |
| Absolute strike + trained finish | **0.2430** | **0.6794** | **1.0062** | **0.2398** | 359.7 sec | **55.75** |
| Historical baseline | 0.2451 | — | 1.0405 | 0.2489 | 360.7 sec | 56.38 |

The combined variant beats the historical baseline on winner Brier, method log
loss, and goes-distance Brier. It also improves strike-attempt MAE over both the
heuristic engine and the simple fighter-history baseline.

## Aggregate calibration

| Quantity | Actual | Heuristic | Strike only | Strike + trained finish |
|---|---:|---:|---:|---:|
| Decision rate | 45.30% | 16.93% | 8.77% | **48.92%** |
| KO/TKO rate | 35.91% | 73.19% | 83.47% | **33.06%** |
| Submission rate | 18.79% | 9.88% | 7.76% | **18.02%** |
| Mean fighter sig attempts | 86.02 | 45.30 | 50.77 | **92.26** |

Replacing the heuristic finish logits is the decisive correction. Absolute strike
pace alone increases landed volume and makes the heuristic KO defect worse. The
trained competing-risk provider restores realistic decision, KO/TKO, and
submission frequencies.

## Improvement versus heuristic engine

- Winner Brier: 1.35% better.
- Method log loss: 24.97% better.
- Goes-distance Brier: 25.15% better.
- Strike-attempt MAE: 2.63% better.
- Fight-time MAE: 18.06% worse.

## Remaining limitation

Fight duration is now the primary blocker. The trained provider predicts the
correct broad finish mix, but the within-round finish-time sampler remains a
fixed beta heuristic and complete-path statistics are thinned uniformly to the
sampled exposure. This produces near-baseline duration error and is not adequate
for round-total or time-market pricing.

The combined path also uses static pre-fight strike pace in every round. A final
dynamic provider must consume simulated prior-round state rather than actual
historical target-fight rounds.

## Decision

Keep promotion blocked. The next controlled component should model conditional
finish timing or round continuation, then rerun the identical three-variant
holdout. Production and wagering use remain prohibited until duration and count
distribution calibration pass alongside winner, method, and distance gates.
