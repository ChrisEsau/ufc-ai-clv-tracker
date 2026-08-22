# UFC Raw Signal Discovery V1

Research-only experiment for discovering predictive signal in historical UFC round data without changing FSR or Event Clock.

## Frozen boundaries

This study must not modify:

- `pipeline/fsr_v2/`
- `data/fsr_v2/`
- `pipeline/fsr_v3/`
- `data/fsr_v3/`
- `pipeline/simulation/event_clock_mc_v1/`
- `pipeline/simulation/event_clock_mc_v2/`
- `data/fight_details/ufc_round_stats.parquet`
- `data/master/ufc_master.parquet`

All generated artifacts belong under:

`data/research/raw_signal_discovery_v1/`

## Question

Can leakage-safe historical round behavior predict future fight winners, and which signals persist chronologically out of sample?

This version is discovery-only. Nothing is promoted into FSR or Event Clock automatically.

## Leakage rules

- Every target fight uses only fighter history from strictly earlier calendar dates.
- Same-date fights snapshot first and update only after the whole date batch.
- Current-fight stats, method, finish time, winner fields, corner, red/blue indicators, odds, market data, FSR, and MC probabilities are not model features.
- Every fight is represented in both directions with weight `0.5` and paired predictions are symmetrized.
- Development folds are 2020, 2021, 2022, and 2023.
- `2024-01-01+` is reserved as the outer holdout and is not scored by the development runner.

## V1 feature factory

Historical raw round statistics are first aggregated into fighter-fight observations in memory. For each future fight, rolling summaries are generated for:

- career
- last 1 fight
- last 3 fights
- prior 730 days

V1 summaries use mean and standard deviation. Comparing recent-window means with career means lets the tree model discover recent shifts, while the standard deviations expose fighter-specific volatility. Chronological slopes, additional windows, and tsfresh are intentionally deferred until this bounded development gate shows useful signal.

Candidate metrics include strike, target, location, takedown, submission, control, damage, round-progression, accuracy, share, and volatility signals.

Current prefight physical context includes age, height, reach, weight, and stance indicators. Fight context includes scheduled rounds and title-fight status.

For each fighter feature `X`, the directional matchup bank contains:

- `self_X`
- `opp_X`
- `diff_X`

## Development models

1. 50/50 coin-flip baseline
2. regularized logistic regression
3. XGBoost

Primary metric: log loss.

Secondary metrics: Brier score, ROC AUC, and accuracy.

XGBoost discovery outputs include fold-level gain, mean absolute SHAP contribution, SHAP rank, and fold-stability summaries.

## Run

From the repository root:

```bash
PYTHONPATH=. python -m pipeline.research.raw_signal_discovery_v1.run
```

The runner builds the feature bank, requires the leakage audit to pass, then runs development folds only.

## Outputs

- `prefight_feature_bank.parquet`
- `feature_manifest.csv`
- `leakage_audit.csv`
- `suspicious_single_feature_auc.csv`
- `development_metrics.csv`
- `development_predictions.csv`
- `fold_feature_importance.csv`
- `signal_stability.csv`
- `research_summary.json`

## Promotion rule

No feature discovered here is a validated FSR trait or approved Event Clock mechanic. Surviving signals become separate empirical research hypotheses only after chronological stability and later reserved outer validation.
