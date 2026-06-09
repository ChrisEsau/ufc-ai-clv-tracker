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
Which model configuration actually makes money?
```

---

## Current Project State (June 2026)

The fighter-state refactor has been validated and is now considered the canonical foundation for future model development.

Validated results:

```text
✓ Fighter-state architecture implemented
✓ Moneyline feature view implemented
✓ 124/124 V5 model features present
✓ 16/16 engineered features present
✓ End-to-end XGBoost training successful
✓ Refactor model metrics approximately equal to legacy model metrics
```

Canonical architecture:

```text
ufc_master.parquet
        ↓
fighter_state_history.parquet
        ↓
moneyline_feature_view.parquet
        ↓
training
        ↓
evaluation
```

The legacy rolling feature artifact remains available for historical reference but should no longer be considered the long-term source of truth.

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
* Maintain experiment history
* Identify profitable feature/model combinations

---

## Primary Inputs

* data/master/ufc_master.parquet
* fighter-state artifacts
* feature-view artifacts
* model configuration YAMLs
* market odds history
* bet result history
* CLV history

---

## Primary Outputs

* Trained model files
* Feature column registry
* Calibration reports
* Backtest results
* ROI summaries
* Model comparison tables
* Experiment registry

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
* Bet volume

---

## Canonical Feature Architecture

Current canonical feature artifacts:

```text
data/features/fighter_state_history.parquet
data/features/latest_fighter_state.parquet
data/features/moneyline_feature_view.parquet
```

Future direction:

```text
feature_view.yaml
        ↓
generic feature-view engine
        ↓
moneyline view
KO/TKO view
submission view
decision view
distance view
custom experiment views
```

The generic feature-view engine is the primary remaining architecture project for Model Lab support.

---

## Model Lab Vision

The dashboard should become a configuration-driven experiment platform.

Users should be able to select:

```text
Bet Type
  - Moneyline
  - KO/TKO
  - Submission
  - Decision
  - Goes Distance
  - Round Props

Feature View
  - Base
  - EWM Heavy
  - Recent Form
  - Custom

Model Family
  - XGBoost
  - Random Forest
  - Logistic Regression
  - Neural Network
  - Ensemble

Split Strategy
  - Temporal
  - Walkforward
  - Rolling Window

Calibration
  - None
  - Isotonic
  - Platt
```

The dashboard should generate configuration files, not contain model logic.

```text
Dashboard
      ↓
experiment YAML
      ↓
pipeline execution
      ↓
artifacts
      ↓
leaderboard
```

---

## Model Registry

Every experiment should automatically generate a registry record.

Recommended artifact:

```text
data/model_registry/model_runs.parquet
```

Recommended fields:

```text
experiment_id
model_id
feature_view_id
feature_count
accuracy
roc_auc
log_loss
brier
roi
clv
bet_count
created_at
```

This registry becomes the source for Model Lab leaderboards.

---

## Future Dashboard Sections

```text
Model Artifact Status
Model Quality
Feature Importance
Live Prediction Audit

Backtest Results
Threshold / ROI Sweep
Calibration Report
Recent-Era Validation
Feature Drift
Experiment Registry
Model Leaderboard
```

---

## Guiding Principle

The goal of Model Lab is not to find the model with the highest ROC-AUC.

The goal is to identify:

```text
Feature sets
Model families
Bet types
Thresholds
Staking approaches
```

that maximize long-term betting performance while maintaining calibration and robustness.

Model Lab should evolve the UFC platform from a single-model workflow into a repeatable research platform capable of rapidly discovering and validating new edges.