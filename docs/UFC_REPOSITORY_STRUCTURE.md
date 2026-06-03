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

Live card, model prediction, betting-board, and action-board outputs.

### model_lab

Future model-lab reports such as backtests, calibration reports, and model comparisons.

## Models

```text
models/UFC_Model_v5_Experiment
```

Frozen production model artifacts live outside `data/` under `models/`. Runtime code should access these via `pipeline.common.paths`.

---

## Pipeline

```text
pipeline/common
pipeline/data_maintenance
pipeline/prediction
pipeline/features
pipeline/clv
```

### common

Shared utilities and path registry.

### data_maintenance

Ingestion and validation workflows.

### prediction

Live prediction runners.

### features

Feature engineering.

### clv

Market tracking and CLV logic.

---

## Dashboard

```text
tabs/
utils/
```

Tabs contain workspace rendering.

Utils contain reusable dashboard components.

---

## Workflows

```text
.github/workflows
```

GitHub Actions entry points only.

Business logic belongs in pipeline modules.
