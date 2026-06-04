# UFC GitHub Workflow Registry

## Purpose

Registry of GitHub Actions workflows used by the UFC platform.

Status values:

```text
ACTIVE  = production workflow
LEGACY  = retained for debugging/history
REPLACE = scheduled for replacement
RETIRED = no longer used
```

Current workflow execution pattern:

```text
Dashboard Button
  ↓
workflow_dispatch
  ↓
GitHub Workflow
  ↓
python -m pipeline.<workspace>.<runner>
  ↓
Canonical data/status/staging/audit artifacts
```

Workflows that commit generated parquet artifacts must use `git add -f` because generated binary artifacts are ignored by `.gitignore`.

Archived workflows live under `archive/.github/workflows/` and are retained only for historical reference. They are not active GitHub Actions entry points and should not be dispatched by the dashboard.

---

## Data Maintenance Workflows

### ACTIVE: run-dataset-status.yml

Purpose:
Generate dataset health metrics.

Runner:

```bash
python -m pipeline.data_maintenance.run_dataset_status
```

Outputs:

* data/status/ufc_dataset_status.parquet
* data/status/ufc_dataset_event_status.parquet

---

### ACTIVE: run-ufcstats-event-check.yml

Purpose:
Discover UFCStats completed events missing from master dataset.

Runner:

```bash
python -m pipeline.data_maintenance.run_ufcstats_event_check
```

Outputs:

* data/status/ufc_ufcstats_event_check.parquet
* data/staging/ufc_missing_events.parquet

---

### ACTIVE: dm-ingest-single-event.yml

Purpose:
Stage and review one selected UFCStats event.

Inputs:

```text
event_id = UFCStats event ID
mode     = full | smoke
```

Runner:

```bash
python -m pipeline.data_maintenance.run_ingest_single_event
```

Pipeline:

```text
Fight scrape
→ Fight detail scrape
→ Master mapper
→ Derived stats
→ Fighter profile enrichment
→ Master column validation
→ Append precheck
→ Final staged review
```

Important rule:

```text
This workflow does not append to master.
```

Outputs committed:

* data/staging/ufc_staged_fight_rows.parquet
* data/staging/ufc_staged_fight_details.parquet
* data/staging/ufc_staged_master_rows.parquet
* data/staging/ufc_staged_master_rows_enriched.parquet
* data/staging/ufc_staged_fighter_profiles.parquet
* data/staging/ufc_staged_master_rows_profiled.parquet
* data/audits/ufc_fight_scrape_audit.parquet
* data/audits/ufc_fight_detail_scrape_audit.parquet
* data/audits/ufc_staged_master_mapping_audit.parquet
* data/audits/ufc_staged_derived_stats_audit.parquet
* data/audits/ufc_fighter_profile_scrape_audit.parquet
* data/audits/ufc_master_column_validation.parquet
* data/audits/ufc_append_precheck.parquet
* data/audits/ufc_append_duplicate_check.parquet
* data/audits/ufc_append_required_field_audit.parquet
* data/audits/ufc_staged_final_review.parquet

---

### ACTIVE: run-append-precheck-validation.yml

Purpose:
Run append precheck and final staged review for currently staged rows.

Runner sequence:

```bash
python -m pipeline.data_maintenance.run_append_precheck_validation
python -m pipeline.data_maintenance.run_staged_final_review
```

Outputs:

* data/audits/ufc_append_precheck.parquet
* data/audits/ufc_append_duplicate_check.parquet
* data/audits/ufc_append_required_field_audit.parquet
* data/audits/ufc_staged_final_review.parquet

---

### ACTIVE: run-append-staged-to-master.yml

Purpose:
Append reviewed staged rows into master dataset.

Runner:

```bash
python -m pipeline.data_maintenance.run_append_staged_to_master
```

Required gates:

```text
append_ready == True
final_review_pass == True
```

Outputs:

* data/master/ufc_master.parquet
* data/backups/ufc_master_backup_<run_id>.parquet
* data/audits/ufc_append_audit.parquet

---

### ACTIVE: Individual Data Maintenance Step Workflows

These workflows remain available for targeted debugging and operational use:

```text
run-ufcstats-fight-scrape.yml
run-ufcstats-fight-detail-scrape.yml
run-staged-master-mapper.yml
run-staged-derived-stats-transformer.yml
run-fighter-profile-enrichment.yml
run-master-column-validation.yml
```

Their preferred execution form is module execution through `python -m pipeline.data_maintenance...`.

---

## Prediction Workflows

### ACTIVE

```text
run-live-prediction.yml
run-model-predictions.yml
```

Workspace:
Prediction

---

## Feature Workflows

### ACTIVE

```text
run-current-fighter-features.yml
```

Workspace:
Features

---

## Market / CLV Workflows

### ACTIVE

```text
run-market-update.yml
run-clv-tracker.yml
```

Workspace:
CLV

---

## Legacy / Audit Workflows

### ARCHIVED

These workflows are no longer active GitHub Actions entry points under `.github/workflows/`. They have been moved to `archive/.github/workflows/` for historical reference while the production pipeline stabilizes.

```text
archive/.github/workflows/run-detail-column-inventory.yml
archive/.github/workflows/run-master-column-inventory.yml
archive/.github/workflows/run-master-first-row.yml
archive/.github/workflows/run-staged-mapping-audit.yml
archive/.github/workflows/run-staged-master-alignment-audit.yml
archive/.github/workflows/run-staged-schema-audit.yml
archive/.github/workflows/run-known-fight-scrape-validation.yml
archive/.github/workflows/run-append-row-match-validation.yml
archive/.github/workflows/run-fight-detail-structure-audit.yml
archive/.github/workflows/run-repair-master-date-format.yml
```

Purpose:
Development and audit workflows retained for reference only.

---

## Dashboard Workflow Status

The Data Maintenance dashboard can track recently launched workflow runs via the GitHub Actions API.

Workflow status display requires these Streamlit secrets:

```text
GITHUB_OWNER
GITHUB_REPO
GITHUB_TOKEN
GITHUB_BRANCH
```

The dashboard queries workflow runs on `GITHUB_BRANCH` and displays:

* status/conclusion
* branch
* run number
* start/update time
* progress indicator
* GitHub run link

---

## Future Naming Convention

Target naming convention:

```text
dm-dataset-status.yml
dm-event-check.yml
dm-ingest-single-event.yml
dm-append-precheck.yml
dm-append.yml

prediction-live.yml
prediction-model.yml

clv-market-update.yml
features-current-fighter.yml
```

## Betting Board Workflows

### `run-refresh-upcoming-events.yml`

Refreshes the UFCStats upcoming event list and upcoming fight-card artifacts used by the Betting Board selector. This workflow is safe to run before each prediction cycle because it only updates generated card artifacts under `data/cards/`.

### `run-betting-board-selected-event.yml`

Accepts a UFCStats `event_id`, rebuilds the live card for that selected event, runs model predictions, refreshes market odds, and produces the Betting Board outputs. This is the preferred entrypoint when an operator wants predictions for a specific upcoming card.

## Bankroll Workflows

### `run-bankroll-status.yml`

Rebuilds derived bankroll status artifacts from the official bet ledger using `python -m pipeline.bankroll.run_bankroll_status`. It writes `data/bankroll/ufc_open_bets.parquet` and `data/bankroll/ufc_bankroll_snapshots.parquet` for the Bankroll workspace.
