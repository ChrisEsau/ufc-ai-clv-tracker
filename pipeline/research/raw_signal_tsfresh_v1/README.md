# UFC Raw Signal TSFresh V1

Research-only challenger that automatically extracts time-series features from each fighter's strictly prior UFC fight history.

## Boundaries

Do not modify FSR V2/V3, Event Clock V1/V2, `ufc_round_stats.parquet`, or UFC master data. Outputs live only under `data/research/raw_signal_tsfresh_v1/`.

## Leakage protocol

- Target fights are built only for dates before `2024-01-01`.
- Every sequence contains only fighter observations from strictly earlier calendar dates.
- Same-date history is excluded.
- 2024+ is counted as the reserved outer period but is never extracted, fit, selected, or scored.
- Chronological development folds remain 2020, 2021, 2022, and 2023.

## What tsfresh searches

The sequence is the fighter's prior fight-by-fight history, capped at 12 prior UFC fights. Twenty-three robust rate/control series are passed to a bounded tsfresh calculator set that includes trend, autocorrelation, nonlinear complexity, change magnitude, extrema location, quantiles, entropy, peaks, skew/kurtosis, and related sequence descriptors.

This intentionally avoids the unbounded full tsfresh feature explosion while still searching many patterns not manually encoded in Raw Signal Discovery V1.

## Development comparisons

XGBoost evaluates:

- physical only
- tsfresh only
- physical + tsfresh
- existing manual full bank
- manual full + tsfresh
- prior development-pruned manual bank
- prior development-pruned manual bank + tsfresh

The key question is whether adding tsfresh improves chronological log loss/Brier/AUC over the already-working manual model.

## Outputs

- `tsfresh_fold_metrics.csv`
- `tsfresh_metric_summary.csv`
- `tsfresh_feature_manifest.csv`
- `tsfresh_history_coverage.csv`
- `tsfresh_audit.csv`
- `tsfresh_fold_importance.csv`
- `tsfresh_signal_stability.csv`
- `tsfresh_calculator_summary.csv`
- `tsfresh_summary.json`

Nothing discovered here is automatically promoted into FSR or Event Clock.
