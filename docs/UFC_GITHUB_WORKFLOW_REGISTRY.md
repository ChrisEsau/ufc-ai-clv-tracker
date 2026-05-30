UFC GitHub Workflow Registry
Purpose

Registry of all GitHub Actions workflows used by the UFC platform.

Status values:

ACTIVE = production workflow
LEGACY = retained for debugging/history
REPLACE = scheduled for replacement
RETIRED = no longer used
Data Maintenance Workflows
ACTIVE
run-dataset-status.yml

Purpose:
Generate dataset health metrics.

Workspace:
Data Maintenance

Runner:
run_dataset_status.py

run-ufcstats-event-check.yml

Purpose:
Discover events missing from master dataset.

Workspace:
Data Maintenance

Runner:
run_ufcstats_event_check.py

run-ufcstats-fight-scrape.yml

Purpose:
Scrape fight rows for missing events.

Workspace:
Data Maintenance

Runner:
run_ufcstats_fight_scrape.py

run-ufcstats-fight-detail-scrape.yml

Purpose:
Scrape detailed fight statistics.

Workspace:
Data Maintenance

Runner:
run_ufcstats_fight_detail_scrape.py

run-staged-master-mapper.yml

Purpose:
Map UFCStats fight details to canonical schema.

Workspace:
Data Maintenance

Runner:
run_staged_master_mapper.py

run-staged-derived-stats-transformer.yml

Purpose:
Generate derived fight statistics.

Workspace:
Data Maintenance

Runner:
run_staged_derived_stats_transformer.py

run-fighter-profile-enrichment.yml

Purpose:
Enrich staged rows with fighter profiles.

Workspace:
Data Maintenance

Runner:
run_fighter_profile_enrichment.py

run-master-column-validation.yml

Purpose:
Validate staged schema against master schema.

Workspace:
Data Maintenance

Runner:
run_master_column_validation.py

Prediction Workflows
ACTIVE
run-live-prediction.yml

Workspace:
Prediction

Status:
ACTIVE

run-model-predictions.yml

Workspace:
Prediction

Status:
ACTIVE

Features Workflows
ACTIVE
run-current-fighter-features.yml

Workspace:
Features

Status:
ACTIVE

Market / CLV Workflows
ACTIVE
run-market-update.yml

Workspace:
CLV

Status:
ACTIVE

Legacy / Audit Workflows
LEGACY
run-detail-column-inventory.yml
run-master-column-inventory.yml
run-master-first-row.yml
run-staged-mapping-audit.yml
run-staged-master-alignment-audit.yml
run-staged-schema-audit.yml
run-known-fight-scrape-validation.yml

Purpose:
Development and audit workflows retained for reference.

Status:
LEGACY

Future Workflow Architecture

Target naming convention:

dm-dataset-status.yml
dm-event-check.yml
dm-fight-scrape.yml
dm-fight-detail-scrape.yml
dm-profile-enrichment.yml
dm-master-validation.yml

prediction-live.yml
prediction-model.yml

clv-market-update.yml

features-current-fighter.yml

Long-Term Goal

Dashboard Button
↓
workflow_dispatch
↓
GitHub Workflow
↓
Python Module

Example:

python -m pipeline.data_maintenance.run_dataset_status