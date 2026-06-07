# Current Project Handoff

## Purpose

This document captures the current UFC AI CLV Tracker project state so a new chat, agent, or Codex session can resume work without reconstructing recent decisions.

Current branch:

```text
dev
```

The project has successfully completed the first modular training framework migration. The old notebook architecture is no longer the only training path.

---

## Current Successful Training Command

Run from repository root:

```powershell
python -m pipeline.training.run_train_model `
    --config configs/models/moneyline_xgboost_v5.yaml
```

This command completed successfully after the master dataset cleanup and rolling feature rebuild.

---

## Current Dataset State

Authoritative master dataset:

```text
data/master/ufc_master.parquet
```

Current master row count:

```text
8574
```

Important notes:

- 25 invalid-date rows were removed from the master dataset.
- A backup should exist locally from the cleanup process.
- Master schema remains locked at 128 columns.
- Master/modeling layers should contain IDs, not scraper URLs.
- Temporal training requires valid dates; invalid dates should fail validation or be fixed at the master layer.

---

## Current Feature Warehouse

Current canonical rolling feature warehouse:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Current expected shape:

```text
8574 rows
483 columns
```

Rebuild command:

```powershell
python -m pipeline.features.run_build_rolling_features
```

This builder reads:

```text
data/master/ufc_master.parquet
```

and produces the current rolling feature warehouse used by training.

---

## Current Moneyline Model

Primary model config:

```text
configs/models/moneyline_xgboost_v5.yaml
```

Current model ID:

```text
moneyline_xgboost_v5
```

Current algorithm:

```text
xgboost
```

Current artifact output directory:

```text
models/moneyline/xgboost_v5/
```

The model config is now the single source of truth for:

- model ID
- model family
- algorithm
- feature sources
- explicit feature list
- expected feature count
- temporal split
- calibration method
- hyperparameters
- metric settings
- artifact locations

---

## Current Feature Contract

Current moneyline feature count:

```text
124
```

The 124 feature names are explicitly declared in:

```text
configs/models/moneyline_xgboost_v5.yaml
```

Production training should not infer feature columns from naming rules. Rule-based selection may remain only for notebook parity or validation utilities.

Feature source metadata belongs in:

```text
configs/features/feature_sources.yaml
```

Current active source name:

```text
rolling_moneyline_v5
```

Current feature source path:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

---

## Current Training Framework

Training modules live in:

```text
pipeline/training/
```

Current modules:

```text
feature_selection.py
symmetry.py
temporal_split.py
model_training.py
calibration.py
metrics.py
run_train_model.py
algorithms/xgboost_trainer.py
```

Current training flow:

```text
Load model config
Load feature warehouse
Resolve explicit feature contract
Validate feature count and unsafe columns
Apply symmetry augmentation
Build train/calibration/test temporal split
Train XGBoost
Calibrate probabilities
Evaluate final test split
Generate confidence buckets
Save artifacts
```

---

## Last Successful Training Run

Observed output:

```text
Feature dataframe shape: (8574, 483)
Resolved feature count: 124
Model dataframe shape: (17148, 483)
Model training complete.

Train rows        : 13874
Calibration rows  : 1040
Test rows         : 2234
Feature count     : 124
Best threshold    : 0.47
Accuracy          : 0.8075
ROC-AUC           : 0.8948
Log loss          : 0.3993
Brier score       : 0.1324
```

Notes:

- Symmetry doubled the dataset from 8574 to 17148 rows.
- Split mode is `train_calibration_test`.
- Calibration set is separate from final test set.
- Calibration uses scikit-learn-compatible prefit calibration through `FrozenEstimator` when available.

---

## Current Artifact Outputs

The training runner saves artifacts under:

```text
models/moneyline/xgboost_v5/
```

Expected artifacts include:

```text
raw_model.joblib
calibrated_model.joblib
feature_columns.joblib
feature_columns.json
metrics.json
threshold_sweep.parquet
raw_threshold_sweep.parquet
confidence_buckets.parquet
raw_confidence_buckets.parquet
model_card.yaml
```

Generated model/data artifacts may be large. Follow repository rules before committing binary artifacts.

---

## Architecture Decisions Locked In

### Model config is the model contract

Model-specific behavior belongs in model YAML, not scattered through training code.

### Explicit features only for production models

Production models must list all input features explicitly.

### Feature engineering happens before training

Training should load existing feature columns from feature warehouse files. It should not calculate rolling or diff features itself.

### Feature source registry describes warehouses

Warehouse path, join key, status, and ownership belong in feature source configs.

### Calibration must avoid leakage

Do not fit a calibrator on the same rows used for final evaluation.

### Master data quality controls temporal modeling

Invalid dates belong in validation/fix workflows, not hidden inside model training.

---

## Current Next Priorities

1. Create `configs/models/model_registry.yaml` for active/candidate model selection.
2. Finish artifact registry integration for model outputs.
3. Migrate prediction pipeline to load model config/artifacts instead of notebook assumptions.
4. Wire confidence buckets into the Betting Board and Model Lab.
5. Add Model Lab views for metrics, threshold sweep, feature contract, and calibration buckets.
6. Start prop model architecture after moneyline model loading is stable.
7. Add market-aware model configs later; keep current model pure fighter-performance.
8. Consider deleting or aliasing legacy `moneyline_xgb_base.yaml` after references are resolved.

---

## Commands To Resume Work

Pull latest dev:

```powershell
git checkout dev
git pull origin dev
```

Rebuild rolling features:

```powershell
python -m pipeline.features.run_build_rolling_features
```

Train current model:

```powershell
python -m pipeline.training.run_train_model `
    --config configs/models/moneyline_xgboost_v5.yaml
```

Inspect model artifacts:

```powershell
dir models\moneyline\xgboost_v5
```

---

## Important Warnings

- Do not append to master without validation.
- Do not delete master data without backup.
- Do not commit secrets or API keys.
- Do not casually commit generated binary artifacts unless explicitly approved.
- Do not reintroduce hidden feature-selection rules for production models.
