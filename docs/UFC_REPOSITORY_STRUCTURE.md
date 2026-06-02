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
