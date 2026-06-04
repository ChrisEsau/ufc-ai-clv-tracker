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

Live card, model prediction, betting-board, and action-board outputs.

### bankroll

Official bet ledger, open exposure, bankroll snapshots, and persistent risk settings.

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
pipeline/bankroll
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

## Betting Board Runtime Directories

- `data/cards/` stores UFCStats upcoming-event discovery artifacts and the selected event marker used to build the live card.
- `data/market/` stores market odds, snapshots, match audits, closing lines, line movement, and CLV outputs.
- `data/bankroll/` stores the official wager ledger, open bets, bankroll snapshots, and risk settings.
- These directories contain generated parquet files. Source branches should not manually commit ad-hoc generated parquet files; workflows force-add only the canonical artifacts they produce.
