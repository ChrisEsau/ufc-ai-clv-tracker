# UFC Repository Structure

## Purpose

Defines the approved repository layout.

This document is the source of truth for folder organization.

---

## Root Structure

```text
data/
docs/
pipeline/
scrapers/
tabs/
utils/
models/
configs/
archive/
.github/workflows/
```

---

## Data

```text
data/master
data/staging
data/audits
data/status
data/backups
data/features
data/predictions
data/cards
data/market
data/bankroll
data/model_lab
```

### master

Authoritative datasets.

### staging

Temporary ingestion artifacts.

### audits

Validation and audit outputs.

### status

Operational status artifacts.

### backups

Automatic master backups.

### features

Historical and current feature stores.

### predictions

Live card, model prediction, betting-board, watchlist, official-bet, and action-board outputs.

### cards

UFCStats upcoming-event discovery artifacts and selected live-card event markers.

### market

Market odds, market snapshots, normalized snapshots, match audits, closing lines, line movement, and CLV outputs.

### bankroll

Official bet ledger, open exposure, bankroll snapshots, and persistent risk settings.

### model_lab

Future model-lab reports such as backtests, calibration reports, model comparisons, and confidence-bucket exports.

---

## Models

```text
models/
├── UFC_Model_v5_Experiment/
├── UFC_Model_V5_XGBoost/
├── UFC_Model_V5_RF/
└── active_model.json
```

Frozen production model artifacts live outside `data/` under `models/`. Runtime code should access these via `pipeline.common.paths` and future model registry/adapter modules.

Long-term model bundles should follow the contract defined in `docs/MODEL_ADAPTER_ARCHITECTURE.md`.

A production model bundle should include:

```text
model.pkl
feature_columns.pkl
production_config.json
confidence_buckets.parquet
training_metrics.json
```

---

## Configs

```text
configs/
└── models/
    ├── xgboost_v5.yaml
    ├── random_forest_v1.yaml
    └── ensemble_v1.yaml
```

The `configs/` folder is reserved for future model, training, backtesting, and runtime-selection configuration.

Do not hardcode model-specific feature requirements into Betting Board, CLV, or dashboard logic. Model-specific requirements should live in model bundles and/or model config files.

---

## Pipeline

```text
pipeline/common
pipeline/data_maintenance
pipeline/prediction
pipeline/features
pipeline/feature_engineering
pipeline/modeling
pipeline/modeling/adapters
pipeline/training
pipeline/backtesting
pipeline/clv
pipeline/bankroll
```

### common

Shared utilities and path registry.

### data_maintenance

Ingestion and validation workflows.

### prediction

Live prediction runners. These should eventually consume standardized model adapters instead of directly loading a specific model type.

### features / feature_engineering

Feature engineering modules. Long term, feature builders should support model-specific feature contracts.

### modeling

Model registry, model loader, model adapter interfaces, and feature contracts.

Expected future structure:

```text
pipeline/modeling/
├── adapters/
│   ├── base_adapter.py
│   ├── xgboost_adapter.py
│   ├── random_forest_adapter.py
│   └── ensemble_adapter.py
├── model_registry.py
├── model_loader.py
└── feature_contracts.py
```

### training

Python modules converted from training notebooks. Training code should create model bundles but should not directly alter downstream Betting Board or CLV logic.

### backtesting

Walk-forward validation, model evaluation, calibration analysis, and confidence-bucket generation.

### clv

Market tracking and CLV logic.

### bankroll

Ledger, settlement, exposure, and bankroll status runners.

---

## Dashboard

```text
tabs/
utils/
```

Tabs contain workspace rendering.

Utils contain reusable dashboard components.

Dashboard code should consume standardized artifacts and should not depend on whether the active model is XGBoost, Random Forest, Neural Network, or an Ensemble.

---

## Archive

```text
archive/
archive/.github/workflows/
```

The archive stores historical root-level files, duplicate generated artifacts, and retired legacy/audit workflows. Files in `archive/` are retained for reference and should not be treated as active runtime artifacts or production workflow entry points.

---

## Workflows

```text
.github/workflows
```

GitHub Actions entry points only.

Business logic belongs in pipeline modules.

Model training should generally be developed and run from VS Code or external compute/Colab first. GitHub Actions may validate, package, or run lightweight checks, but should not become the primary heavy-training platform unless explicitly approved.

---

## Betting Board Runtime Directories

- `data/cards/` stores UFCStats upcoming-event discovery artifacts and the selected event marker used to build the live card.
- `data/market/` stores market odds, snapshots, match audits, closing lines, line movement, and CLV outputs.
- `data/bankroll/` stores the official wager ledger, open bets, bankroll snapshots, and risk settings.
- `data/predictions/` stores model prediction, betting board, watchlist, action board, and official bet artifacts.
- These directories contain generated parquet files. Source branches should not manually commit ad-hoc generated parquet files; workflows force-add only the canonical artifacts they produce.

---

## Model Adapter Reference

See:

```text
docs/MODEL_ADAPTER_ARCHITECTURE.md
```

Core principle:

```text
Models own their required features.
The pipeline owns the standard prediction output.
```

Downstream systems should consume a stable prediction schema and remain model-agnostic.
