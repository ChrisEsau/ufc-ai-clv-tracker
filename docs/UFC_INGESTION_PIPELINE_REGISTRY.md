# UFC Ingestion Pipeline Registry

## Overview

Purpose:

Ingest new UFC events from UFCStats into `ufc_master.parquet` using a gated, auditable workflow.

Architecture:

SCRAPE → MAP → ENRICH → VALIDATE → APPEND

---

# Runner 1

## run_dataset_status.py

Purpose:
Generate dataset health metrics for the master dataset.

Input:

* data/master/ufc_master.parquet

Outputs:

* data/status/ufc_dataset_status.parquet
* data/status/ufc_dataset_event_status.parquet

Key Metrics:

* Row count
* Column count
* Date health
* Duplicate fight IDs
* Event count
* Fighter count

---

# Runner 2

## run_ufcstats_event_check.py

Purpose:
Compare UFCStats completed events against local master dataset.

Input:

* data/master/ufc_master.parquet
* UFCStats completed events page

Outputs:

* data/status/ufc_ufcstats_event_check.parquet
* data/staging/ufc_missing_events.parquet

Key Output:

* Missing event IDs

---

# Runner 3

## run_ufcstats_fight_scrape.py

Purpose:
Scrape fight rows for missing events.

Input:

* data/staging/ufc_missing_events.parquet

Outputs:

* data/staging/ufc_staged_fight_rows.parquet
* data/audits/ufc_fight_scrape_audit.parquet

Important Rules:

* Filter UFCStats blank summary row
* Extract event_id
* Extract fight_id

Adds:

* event_id
* fight_id
* event_url
* fight_url

Known Validation:

* Allen vs Costa = 13 fights

---

# Runner 4

## run_ufcstats_fight_detail_scrape.py

Purpose:
Scrape detailed fight statistics.

Input:

* data/staging/ufc_staged_fight_rows.parquet

Outputs:

* data/staging/ufc_staged_fight_details.parquet
* data/audits/ufc_fight_detail_scrape_audit.parquet

Adds:

* KD stats
* Significant strikes
* Total strikes
* TD stats
* Control time
* Zone striking stats

Preserves:

* event_id
* fight_id

---

# Runner 5

## run_staged_master_mapper.py

Purpose:
Map UFCStats fight details into canonical 128-column master schema.

Input:

* data/staging/ufc_staged_fight_details.parquet

Outputs:

* data/staging/ufc_staged_master_rows.parquet
* data/audits/ufc_staged_master_mapping_audit.parquet

Important Fixes:

* event_id propagation
* fight_id propagation
* zone stat mapping

Architecture Boundary:

* URLs dropped here
* IDs retained

---

# Runner 6

## run_staged_derived_stats_transformer.py

Purpose:
Generate derived fight statistics.

Input:

* data/staging/ufc_staged_master_rows.parquet

Outputs:

* data/staging/ufc_staged_master_rows_enriched.parquet
* data/audits/ufc_staged_derived_stats_audit.parquet

Creates:

* strike accuracy
* takedown accuracy
* zone accuracy
* strike distribution metrics

Examples:

* r_sig_str_acc
* b_sig_str_acc
* r_head_acc
* b_head_acc
* r_landed_head_per

---

# Runner 7

## run_fighter_profile_enrichment.py

Purpose:
Scrape fighter profiles and enrich staged rows.

Inputs:

* data/staging/ufc_staged_fight_rows.parquet
* data/staging/ufc_staged_master_rows_enriched.parquet

Outputs:

* data/staging/ufc_staged_fighter_profiles.parquet
* data/staging/ufc_staged_master_rows_profiled.parquet
* data/audits/ufc_fighter_profile_scrape_audit.parquet

Adds:

* r_id
* b_id
* height
* reach
* stance
* DOB
* wins
* losses
* SLpM
* SApM
* TD Avg
* TD Def
* Sub Avg

Future Refactor:
Move to ID-first enrichment.

---

# Runner 8

## run_master_column_validation.py

Purpose:
Validate staged rows against canonical master schema.

Inputs:

* data/master/ufc_master.parquet
* data/staging/ufc_staged_master_rows_profiled.parquet

Outputs:

* data/audits/ufc_master_column_validation.parquet

Checks:

* Column count
* Column order
* Dtype alignment

Expected:

* 128 columns
* Exact order match

---

# Runner 9

## run_append_precheck_validation.py

Purpose:
Determine append readiness.

Inputs:

* data/master/ufc_master.parquet
* data/staging/ufc_staged_master_rows_profiled.parquet

Outputs:

* data/audits/ufc_append_precheck.parquet
* data/audits/ufc_append_duplicate_check.parquet
* data/audits/ufc_append_required_field_audit.parquet

Checks:

* Column count match
* Column order match
* Duplicate fight IDs
* Existing fight IDs
* Required fields
* Negative stats

Result:
append_ready = True | False

---

# Runner 10

## run_append_staged_to_master.py

Purpose:
Append staged rows into master dataset.

Inputs:

* data/master/ufc_master.parquet
* data/staging/ufc_staged_master_rows_profiled.parquet
* data/audits/ufc_append_precheck.parquet

Outputs:

* data/master/ufc_master.parquet
* data/backups/ufc_master_backup_<run_id>.parquet
* data/audits/ufc_append_audit.parquet

Safety Controls:

* Refuse append if append_ready=False
* Refuse duplicate fight IDs
* Backup master before append

---

# Locked Architecture

Scrapers:

* URLs + IDs

Staging:

* URLs + IDs

Audits:

* URLs + IDs

Mapper:

* Drop URLs
* Keep IDs

Master:

* IDs only

Feature Stores:

* IDs only

Prediction Pipeline:

* IDs only

---

# Next Phase

Single Event Ingestion

Workflow:

Event Discovery
↓
Select Event
↓
run_ingest_single_event.py
↓
Fight Scrape
↓
Fight Detail Scrape
↓
Mapper
↓
Derived Stats
↓
Fighter Profiles
↓
Validation
↓
Precheck
↓
Append
↓
Master Updated


## run_ingest_single_event.py

Purpose:
Orchestrates single-event ingestion through validation and append precheck.

Command:
```powershell
python -m pipeline.data_maintenance.run_ingest_single_event