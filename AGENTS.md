# AGENTS.md

## Project Name

UFC AI Betting Intelligence Platform

## Primary Goal

Build a production-ready UFC betting intelligence platform with:

* Data ingestion
* UFCStats event scraping
* Master dataset maintenance
* Fight prediction model outputs
* Odds import
* Expected value calculations
* Kelly staking
* Closing line value tracking
* Dashboard control plane

This repository should be treated as an active production-style analytics system, not a notebook-only experiment.

---

## Branching Rules

Use this branch flow:

```text
feature/codex-task
        ↓
       dev
        ↓
      main
```

Rules:

* Never commit directly to `main`.
* Codex changes should normally target `dev`.
* `main` should stay stable.
* Use pull requests whenever possible.
* If Codex Cloud cannot create a PR because of binary files, remove binary changes and retry.

---

## Important Project Rules

### Do Not Commit Binary/Data Artifacts

Do not commit generated data artifacts unless explicitly requested.

Avoid committing:

* `.parquet`
* `.pkl`
* `.pyc`
* `.png`
* `.jpg`
* `.jpeg`
* `.zip`
* `.xlsx`
* `.venv/`
* `__pycache__/`
* `.ipynb_checkpoints/`

Source files are preferred:

* `.py`
* `.md`
* `.yml`
* `.yaml`
* `.toml`
* `.txt`
* `.csv` only when intentionally used as source/reference data

---

## Preferred Repository Structure

```text
data/
  master/
  staging/
  audits/
  status/
  backups/

docs/

pipeline/
  common/
  data_maintenance/
  prediction/
  clv/
  features/

scrapers/

tabs/

utils/

.github/
  workflows/
```

---

## Path Architecture

Use `pipeline.common.paths` as the single source of truth for file paths.

Do not hardcode scattered paths throughout the app.

Important folders:

```text
data/master/
data/staging/
data/audits/
data/status/
data/backups/
```

Important artifact categories:

* `data/master/` = authoritative master dataset
* `data/staging/` = temporary staged scrape/map/enrichment files
* `data/audits/` = validation, scrape, match, and append audit files
* `data/status/` = dashboard status artifacts
* `data/backups/` = backup copies before destructive or append operations

---

## Python Execution Standard

Use Python package/module execution.

Preferred:

```bash
python -m pipeline.data_maintenance.run_dataset_status
```

Avoid:

```bash
python pipeline/data_maintenance/run_dataset_status.py
```

Use `__init__.py` files where needed.

Avoid `sys.path` hacks unless absolutely necessary.

---

## UFC Master Dataset

The authoritative master file is:

```text
data/master/ufc_master.parquet
```

The UFC master dataset has a locked 128-column schema.

Treat this schema as authoritative.

The master file should contain IDs, not scraper URLs.

URLs may exist in scraper, staging, and audit artifacts for traceability, but URLs should be dropped at the mapper boundary before entering the master dataset or modeling layers.

---

## Scraper / Mapper Boundary Rule

Scrapers and staging artifacts may keep:

* UFCStats event URLs
* UFCStats fight URLs
* UFCStats fighter URLs
* Raw scrape metadata
* Debug fields

Master/modeling artifacts should keep:

* Event IDs
* Fight IDs
* Fighter IDs
* Clean structured fields

The mapper is the boundary where URLs are dropped and IDs are retained.

---

## Data Maintenance Ingestion Flow

The long-term ingestion flow is:

```text
Event Check
    ↓
Missing Event Discovery
    ↓
Select Event
    ↓
Ingest Event
    ↓
Validate
    ↓
Append
    ↓
Master Updated
```

Single-event ingestion architecture should use:

```text
pipeline/data_maintenance/map_event_to_master.py
pipeline/data_maintenance/validate_event_append.py
pipeline/data_maintenance/append_event_to_master.py
pipeline/data_maintenance/run_ingest_single_event.py
```

---

## Append Safety Rule

Appending to the master dataset must be gated.

The append button/action should only be enabled when:

```text
append_ready == True
```

from the append precheck artifact.

Validation should check:

* Schema order
* Column count
* Duplicate fight IDs
* Required fields
* Negative stat values
* Date format issues
* Staged rows exist
* Master file exists
* Backup created before append

Never append staged rows blindly.

---

## Data Maintenance Dashboard Architecture

The Data Maintenance dashboard should be mobile-first.

Preferred order:

```text
1. Dataset Health
2. Workflow Status
3. Event Discovery
4. Fight Scrape
5. Enrichment
6. Validation Gate
7. Audit History / Details
8. Append Status
```

Append Status belongs at the bottom as the final action.

The `Append To Master` button belongs only in the final Append Status section.

Workflow buttons should live inside the sections that consume their artifacts.

Examples:

* Dataset Health section contains `Run Dataset Status`
* Event Discovery section contains `Run Event Check`
* Fight Scrape section contains scrape buttons
* Enrichment section contains mapper/enrichment buttons
* Validation Gate contains validation/precheck buttons
* Append Status contains only the gated append button

Avoid a single large generic “Workflow Controls” section.

---

## Dashboard Workspaces

Long-term dashboard workspaces:

```text
1. Betting Board
2. Line Movement / CLV
3. Bet Ledger / Bankroll
4. Model Lab
5. Data Maintenance
```

Each dashboard should be treated as a workspace, not just a visual tab.

---

## Prediction Artifact Separation

Preserve this separation:

```text
ufc_model_predictions.parquet
```

Pure model opinion.

```text
ufc_live_feature_audit.parquet
```

Feature engineering and data-quality QA.

```text
ufc_live_match_audit.parquet
```

Fighter matching and entity-resolution QA.

Do not mix prediction logic, feature validation, and fighter matching diagnostics into one artifact.

---

## Feature Store Architecture

Historical training/backtesting feature store:

```text
ufc_rolling_features_EWM.parquet
```

Live production fighter-state feature store:

```text
ufc_current_fighter_features.parquet
```

Live prediction runners should use the current fighter-state feature store rather than scanning the entire historical rolling feature database.

---

## Betting Model Defaults

Default betting strategy configuration:

```text
EV Threshold: $50
Confidence Threshold: 70%
Bet Sizing: Half-Kelly Criterion
Odds Filter: -250 to +400
```

Do not change these defaults unless explicitly requested.

---

## Current Model Direction

Current priority order:

```text
1. V5 model development
2. Prop bet engine
3. Advanced market analytics
4. Closing line value tracking
5. Recent-era validation
6. Confidence-weighted staking
7. Market-aware features
8. Ensemble modeling
9. Dashboard
10. Automation
11. Notifications
12. Deployment infrastructure
```

Prop roadmap:

* KO/TKO props
* Submission props
* Decision props
* Inside the Distance
* Goes Distance
* Round props
* Correlated props
* Market exploitation

---

## Coding Style

Write code that is:

* Clear
* Modular
* Commented
* Safe around missing files
* Safe around empty DataFrames
* Safe around schema mismatch
* Easy to run locally and in GitHub Actions

Prefer functions over giant script blocks.

Use explanatory comments, especially in notebook/Colab-style code.

When adding dashboard code, avoid breaking existing tabs.

When reading parquet files, gracefully handle missing files and show useful status messages.

---

## GitHub Actions

Workflow YAML files should remain in:

```text
.github/workflows/
```

Use clean workflow names.

Prefer `workflow_dispatch` for manual dashboard-triggered workflows.

Do not create nested workflow folders unless explicitly requested.

Important workflow concepts:

* Dataset status
* Event check
* Fight scrape
* Detail scrape
* Enrichment
* Validation/precheck
* Append to master
* Single-event ingestion

---

## Documentation Files

Preferred docs:

```text
docs/UFC_INGESTION_PIPELINE_REGISTRY.md
docs/UFC_MASTER_SCHEMA.md
docs/UFC_PATH_REGISTRY.md
docs/UFC_DM_DASHBOARD_ARCHITECTURE.md
docs/UFC_PREDICTION_PIPELINE.md
docs/UFC_CLV_TRACKING.md
```

Agents should read these docs before making architectural changes.

If docs and code conflict, flag the conflict instead of guessing.

---

## Local Development

User may develop locally in VS Code and push to GitHub.

Preferred local flow:

```bash
git checkout dev
git pull origin dev
git checkout -b feature/specific-task
# make changes
git add .
git commit -m "Describe change"
git push origin feature/specific-task
```

Then open PR into `dev`.

---

## Codex Instructions

When starting a new Codex task, read:

```text
AGENTS.md
docs/*.md
```

before making changes.

Codex should avoid modifying binary/data artifacts.

Codex should prefer small, reviewable changes.

Before coding or moving files, Codex must provide a concise implementation summary and wait for explicit user approval. Read-only review, planning, and recommendations do not count as approval to edit files.

Codex should explain:

* What files changed
* Why they changed
* How to test
* Any assumptions
* Any follow-up work

---

## Testing Expectations

Before claiming completion, check at least:

```bash
python -m compileall pipeline scrapers tabs utils
```

When applicable, also test:

```bash
streamlit run app.py
```

or the correct dashboard entrypoint.

For ingestion scripts, prefer dry-run or validation mode before append mode.

---

## Safety Rules

Never append to master unless validation passes.

Never delete master data without creating a backup.

Never overwrite master schema casually.

Never remove columns from the 128-column schema unless explicitly approved.

Never commit secrets or API keys.

Never commit `.env`.

Never expose betting API keys in source code.

Use environment variables for secrets.

---

## Current Architectural Decisions Locked In

* Master schema is locked at 128 columns.
* Parquet artifacts live under structured `data/` folders.
* `pipeline.common.paths` is the path registry.
* Dashboard is mobile-first.
* Data Maintenance append action belongs at the bottom.
* Append button is gated by `append_ready`.
* Workflow launch buttons belong inside relevant dashboard sections.
* URLs stay in scraper/staging/audit layers.
* IDs only in master/modeling layers.
* Prediction, feature audit, and match audit artifacts stay separate.
* Use module execution with `python -m`.
* Prefer branch flow: feature branch → `dev` → `main`.
