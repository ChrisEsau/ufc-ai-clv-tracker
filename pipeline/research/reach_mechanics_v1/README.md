# Reach Mechanics V1

Isolated research only. This study does **not** modify FSR V3, Event Clock V1/V2, or any source parquet.

## Question

After the current leakage-safe FSR V3 prefight matchup expectation is known, does physical reach still explain future fight mechanics strongly enough to justify a runtime reach translation in Event Clock?

The primary comparison is:

- FSR expectation + division + age edge + height edge
- the same model + reach edge

This distinguishes a true reach effect from generic body-size, division, and age effects.

## Physical candidates

- raw reach edge: `self_reach - opponent_reach`
- height edge
- ape-index edge: `(reach-height)_self - (reach-height)_opponent`
- saturating reach transform: `tanh(reach_edge / 4 inches)`
- age edge
- division fixed effects

## FSR-backed mechanics

The study reads canonical FSR V3 prefight snapshots and uses the exact Event Clock V2 V3 adapter transforms. It compares reach against residual variation in:

1. distance significant-strike attempt generation
2. distance significant-strike landing probability
3. takedown attempt generation
4. takedown completion probability
5. ground significant-strike attempt generation conditional on own UFCStats control
6. ground significant-strike landing probability

Rate targets use the same native exposures as FSR V3. Count models use the FSR expected count as a log offset. Accuracy models use the FSR expected probability as a logit offset.

## Validation

- development folds: 2020, 2021, 2022, 2023
- training is strictly earlier than the validation year
- `2024-01-01+` remains reserved and is never fit or scored
- the common physical cohort requires both fighters to have reach, height, and age available
- all results are research hypotheses only

## Interpretation gate

Reach is a candidate simulator mechanic only if adding reach to the age+height+division control model improves chronological out-of-sample prediction consistently and the fitted reach direction is stable.

No coefficient from this study is automatically promoted into Event Clock.

## Run

```bash
PYTHONPATH=. python -m pipeline.research.reach_mechanics_v1.run
```

## Outputs

Research outputs are written only under `data/research/reach_mechanics_v1/`:

- `mechanic_fold_metrics.csv`
- `reach_incremental_summary.csv`
- `reach_coefficient_summary.csv`
- `reach_residual_bins.csv`
- `coverage.csv`
- `audit.csv`
- `reach_mechanics_summary.json`
