# Training Framework Architecture

## Purpose

The training framework replaces notebook-only training with a modular, config-driven architecture.

Primary entrypoint:

```powershell
python -m pipeline.training.run_train_model `
    --config configs/models/moneyline_xgboost_v5.yaml
```

The goal is to support multiple model families, multiple algorithms, and multiple feature warehouses without rewriting training logic.

---

## Design Principles

### Config Driven

Model YAML files define:

- algorithm
- feature sources
- explicit feature list
- expected feature count
- temporal split strategy
- calibration method
- metrics settings
- artifact output locations

Training code should not contain model-specific feature lists.

### Feature Warehouse Driven

Training consumes engineered feature warehouses.

Training should not compute rolling features itself.

### Model Agnostic

The framework should support:

```text
XGBoost
CatBoost
LightGBM
Neural Networks
Future Ensembles
```

through trainer adapters.

---

## Current Layout

```text
pipeline/training/
│
├── feature_selection.py
├── symmetry.py
├── temporal_split.py
├── model_training.py
├── calibration.py
├── metrics.py
├── run_train_model.py
│
└── algorithms/
    └── xgboost_trainer.py
```

---

## Module Responsibilities

### run_train_model.py

Primary orchestration layer.

Responsibilities:

- load model config
- load feature warehouse
- resolve feature contract
- apply symmetry
- build temporal split
- train model
- calibrate probabilities
- evaluate metrics
- save artifacts

This file should remain thin and orchestration-focused.

---

### feature_selection.py

Responsible for model feature contracts.

Responsibilities:

- load model config
- resolve feature columns
- validate required columns exist
- validate expected feature count
- reject missing features

Production models use explicit feature lists.

---

### symmetry.py

Responsible for red/blue symmetry augmentation.

Purpose:

- reduce side-order bias
- double training observations
- improve model robustness

Current mode:

```text
flip_all
```

Expected effect:

```text
8574 rows
→
17148 rows
```

---

### temporal_split.py

Responsible for leakage-safe train/test construction.

Current production mode:

```text
train_calibration_test
```

Current configuration:

```text
Train       <= 2022
Calibration = 2023
Test        = 2024+
```

Purpose:

- preserve chronology
- prevent future leakage
- support proper calibration

---

### model_training.py

Algorithm dispatch layer.

Responsibilities:

- select trainer
- pass parameters
- return fitted model

Future trainers should plug in here.

---

### algorithms/xgboost_trainer.py

Current trainer implementation.

Responsibilities:

- build XGBoost classifier
- fit model
- return trained estimator

Future trainer examples:

```text
catboost_trainer.py
lightgbm_trainer.py
nn_trainer.py
```

---

### calibration.py

Responsible for probability calibration.

Current method:

```text
isotonic
```

Purpose:

Convert:

```text
Raw model probability
```

into:

```text
Calibrated probability
```

Current implementation supports modern scikit-learn calibration APIs.

Calibration must use rows separate from training and final evaluation.

---

### metrics.py

Responsible for evaluation outputs.

Current outputs:

- accuracy
- ROC-AUC
- log loss
- Brier score
- threshold sweep
- confidence buckets

Confidence buckets are intended for future dashboard integration.

---

## Current Feature Warehouse

Current source:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Current shape:

```text
8574 rows
483 columns
```

Current model feature count:

```text
124
```

Configured explicitly in:

```text
configs/models/moneyline_xgboost_v5.yaml
```

---

## Current Successful Training Results

Observed successful framework run:

```text
Feature dataframe shape: (8574, 483)
Resolved feature count: 124
Model dataframe shape: (17148, 483)

Train rows        : 13874
Calibration rows  : 1040
Test rows         : 2234

Best threshold    : 0.47
Accuracy          : 0.8075
ROC-AUC           : 0.8948
Log loss          : 0.3993
Brier score       : 0.1324
```

---

## Future Architecture

### Model Registry

Planned:

```text
configs/models/model_registry.yaml
```

Purpose:

- active model selection
- candidate tracking
- deployment metadata

### Multiple Feature Warehouses

Future examples:

```text
rolling_moneyline
rolling_props
market_features
fighter_snapshots
```

Feature ownership should be declared in feature-source configuration.

### Prediction Framework Migration

Prediction runners should load:

- model config
- model artifacts
- feature contract

rather than relying on notebook assumptions.

### Model Lab Integration

Future Model Lab should expose:

- metrics
- calibration buckets
- threshold sweeps
- feature contract
- model card

from saved training artifacts.
