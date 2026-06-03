# UFC Model Lab Architecture

## Purpose

The Model Lab is the research and model-development workspace.

It answers:

```text
How accurate is the model?
Is the model calibrated?
Which features matter?
What changes improve ROI?
Is the model stable over recent UFC eras?
```

---

## Core Responsibilities

* Train UFC prediction models
* Validate model performance
* Monitor calibration
* Compare model versions
* Run backtests
* Analyze feature importance
* Evaluate ROI by threshold
* Support ensemble modeling

---

## Primary Inputs

* data/master/ufc_master.parquet
* historical feature stores
* training feature registry
* model artifacts under `models/UFC_Model_v5_Experiment/`
* market odds history
* bet result history

---

## Primary Outputs

* Trained model files
* Feature column registry
* Calibration reports
* Backtest results
* ROI summaries
* Model comparison tables

---

## Current Model Direction

The platform is moving toward V5 model development.

Priority areas:

```text
1. V5 model development
2. Prop bet engine
3. Advanced market analytics
4. Closing line value tracking
5. Recent-era validation
6. Confidence-weighted staking
7. Market-aware features
8. Ensemble modeling
```

---

## Core Validation Metrics

* Accuracy
* Log loss
* ROC-AUC
* Calibration error
* Brier score
* ROI
* Flat bet profit
* Kelly bet profit
* Beat closing line rate

---

## Feature Stores

Historical point-in-time feature store:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Live fighter-state feature store:

```text
data/features/ufc_current_fighter_features.parquet
```

---

## Future Enhancements

* Model comparison dashboard
* Calibration dashboard
* Automated retraining
* Feature drift monitoring
* Recent-era validation panel
* Prop model lab
* Ensemble model selector

---

## Current Paused State

Model Lab work is intentionally paused while development focus moves to the Betting Board.

Completed state before pause:

* The Model Lab tab is a read-only diagnostics workspace, not a training or promotion control surface.
* Production model artifacts are loaded from `models/UFC_Model_v5_Experiment/` through `pipeline.common.paths`.
* Feature artifacts are loaded from `data/features/` through `pipeline.common.paths`.
* Prediction and audit artifacts are loaded from `data/predictions/` and `data/audits/` through `pipeline.common.paths`.
* The Streamlit tab currently surfaces artifact readiness, production model metadata, quality summary metrics, SHAP feature importance, and live prediction audit summaries.

No automated retraining, model promotion, rollback, or backtest generation should be added to the Model Lab until Betting Board work is complete.

---

## Implemented Model Lab Dashboard Sections

Current sections:

```text
Model Artifact Status
Model Quality
Feature Importance
Live Prediction Audit
```

Current behavior:

* `Model Artifact Status` verifies expected production artifacts and shows their canonical paths.
* `Model Quality` displays existing quality-summary metrics such as calibrated accuracy, ROC-AUC, log loss, threshold, train fights, and test fights.
* `Feature Importance` reads the existing SHAP importance CSV and displays ranked feature importance.
* `Live Prediction Audit` reads the latest model prediction and live audit parquet files to summarize data-quality and feature-match readiness.

These sections are viewers only. Any future button in Model Lab should dispatch a workflow and read committed artifacts back from `data/model_lab/`; the dashboard should not perform training or backtesting inline.

---

## Deferred Phase 2 Plan

When Model Lab work resumes, the recommended Phase 2 is to build a reproducible evaluation layer before any retraining or model promotion.

Recommended artifacts:

```text
data/model_lab/model_lab_run_manifest.parquet
data/model_lab/model_backtest_results.parquet
data/model_lab/model_threshold_sweep.parquet
data/model_lab/model_calibration_bins.parquet
data/model_lab/model_recent_era_validation.parquet
data/model_lab/model_feature_drift.parquet
data/model_lab/model_comparison_summary.parquet
```

Recommended package layout:

```text
pipeline/model_lab/__init__.py
pipeline/model_lab/run_model_backtest.py
pipeline/model_lab/run_threshold_sweep.py
pipeline/model_lab/run_calibration_report.py
pipeline/model_lab/run_recent_era_validation.py
pipeline/model_lab/run_feature_drift_report.py
```

Recommended dashboard sections after Phase 2:

```text
Backtest Results
Threshold / ROI Sweep
Calibration Report
Recent-Era Validation
Feature Drift
Model Lab Run History
```

Recommended workflow:

```text
.github/workflows/run-model-lab-backtest.yml
```

The workflow should run model-lab pipeline modules, commit generated `data/model_lab/*.parquet` artifacts with `git add -f`, and allow Streamlit to display workflow status using the existing GitHub Actions helpers.

---

## Deferred Model Lab Guardrails

Do not implement these until after Betting Board improvements are stable:

* Automated retraining.
* Production model replacement.
* Model promotion / rollback controls.
* Ensemble model selection.
* Prop model lab.
* Market-aware feature retraining.

The first future Model Lab implementation should be historical evaluation of the current frozen production model, not a new training loop.
