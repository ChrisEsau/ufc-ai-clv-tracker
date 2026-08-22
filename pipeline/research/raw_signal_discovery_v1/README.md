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

Can leakage-safe historical round behavior predict future fight winners, which signals persist chronologically out of sample, and which stable signal families add information beyond simple FSR-like mean levels?

This version is discovery-only. Nothing is promoted into FSR or Event Clock automatically.

## Leakage rules

- Every target fight uses only fighter history from strictly earlier calendar dates.
- Same-date fights snapshot first and update only after the whole date batch.
- Current-fight stats, method, finish time, winner fields, corner, red/blue indicators, odds, market data, FSR, and MC probabilities are not model features.
- Every fight is represented in both directions with weight `0.5` and paired predictions are symmetrized.
- Development folds are 2020, 2021, 2022, and 2023.
- `2024-01-01+` is reserved as the outer holdout and is not scored by either the discovery runner or the signal-dissection runner.

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

## Development-only signal dissection

After the base discovery gate completes, `dissect_signals.py` uses the same 2020-2023 chronological folds and keeps 2024+ sealed.

It runs two complementary diagnostics:

1. **Family/model ablations** — full raw model, physical-only, approximate validated-FSR-like mean levels only, novel-behavior-only, physical + FSR-like, physical + novel, plus leave-one-family-out tests for physical context, volatility, round progression, target mix, wrestling/control, and duration/experience.
2. **Signed SHAP dependence** — the 30 most stable discovery features are summarized by fold for signed direction, quartile effect, rank correlation between feature value and model contribution, and 10-bin dependence curves.

The FSR-like comparison is intentionally approximate: it includes historical mean-level TD, distance, ground, and KD-production information while excluding volatility and round-progression summaries. It is a diagnostic information-family comparison, not a reimplementation of FSR V3.

No dissection result is allowed to promote a feature. It can only nominate a narrower empirical follow-up.

## Run

From the repository root:

```bash
PYTHONPATH=. python -m pipeline.research.raw_signal_discovery_v1.run
PYTHONPATH=. python -m pipeline.research.raw_signal_discovery_v1.dissect_signals
```

The GitHub Actions workflow runs both stages in sequence, uploads the full research artifact, and persists compact result tables on the research branch.

## Outputs

Base discovery:

- `prefight_feature_bank.parquet`
- `feature_manifest.csv`
- `leakage_audit.csv`
- `suspicious_single_feature_auc.csv`
- `development_metrics.csv`
- `development_predictions.csv`
- `fold_feature_importance.csv`
- `signal_stability.csv`
- `research_summary.json`

Signal dissection:

- `family_ablation_metrics.csv`
- `family_ablation_summary.csv`
- `feature_family_counts.csv`
- `signed_shap_fold_summary.csv`
- `signed_shap_direction_summary.csv`
- `signed_shap_dependence_bins.csv`
- `dissection_audit.csv`
- `dissection_summary.json`

## Promotion rule

No feature discovered here is a validated FSR trait or approved Event Clock mechanic. Surviving signals become separate empirical research hypotheses only after chronological stability and later reserved outer validation.
