# UFC Ingestion Pipeline Registry

## Overview

Purpose:

Ingest new UFC events from UFCStats into staged artifacts, validate them, run final human-review checks, and append only after operator approval.

Current architecture:

```text
EVENT CHECK
  ↓
SINGLE EVENT INGEST
  ↓
SCRAPE
  ↓
MAP
  ↓
ENRICH
  ↓
VALIDATE
  ↓
APPEND PRECHECK
  ↓
FINAL STAGED REVIEW
  ↓
HUMAN APPROVAL
  ↓
APPEND
```

Single-event ingestion stops after final staged review. It does **not** append.

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
Scrape fight rows for selected/missing events.

Input:

* data/staging/ufc_missing_events.parquet

Outputs:

* data/staging/ufc_staged_fight_rows.parquet
* data/audits/ufc_fight_scrape_audit.parquet

Important Rules:

* Filter UFCStats blank summary row.
* Extract `event_id`.
* Extract `fight_id`.
* Preserve event/fight URLs in staging and audits.

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

Creates / maps:

* `location` from the UFCStats event page location field propagated through staged fight rows.
* `division` from the staged UFCStats weight class after removing title-bout markers.
* `title_fight` as `1` for title fights and `0` for non-title fights.
* `total_rounds` from UFCStats `time_format` when available, with safe fallback to `5` for title fights and `3` otherwise.
* Accuracy fields using `0` when the denominator is zero or missing, rather than storing missing values for zero-attempt calculations.

Architecture Boundary:

* URLs are dropped here.
* IDs are retained.

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

* Strike accuracy
* Takedown accuracy
* Zone accuracy
* Strike distribution metrics

Zero denominator rule:

* Derived accuracy / percentage calculations must return `0` when the denominator is zero or missing.
* New staged rows should not emit `NA` solely because an accuracy calculation evaluates to zero.

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
* winner_id
* height
* reach
* stance
* DOB
* wins/losses/draws
* SLpM
* SApM
* TD Avg
* TD Def
* Sub Avg

Current behavior:

* Builds queue from both red and blue fighter URLs.
* Extracts fighter IDs from UFCStats fighter URLs.
* Deduplicates by fighter ID.
* Maps profiles by fighter ID and normalized fighter name.
* Supports optional `--max-fighters` smoke-test limit.

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
* Duplicate mapped columns
* Missing columns
* Extra columns

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

Blocking checks include:

* Column count match
* Column order match
* Duplicate fight IDs in staged rows
* Fight IDs already in master
* Required blocking fields populated, including event/fight metadata (`location`, `division`, `title_fight`, `total_rounds`)
* Fighter identity complete (`r_id`, `b_id`, `winner_id`)
* Negative stat check

Warning checks include:

* Optional profile completeness fields such as height, reach, stance, DOB

Result:

```text
append_ready = True | False
```

`append_ready` is computed from blocking checks only.

---

# Runner 10

## run_staged_final_review.py

Purpose:
Run semantic staged-row review before append.

Inputs:

* data/master/ufc_master.parquet
* data/staging/ufc_staged_master_rows_profiled.parquet

Outputs:

* data/audits/ufc_staged_final_review.parquet

Blocking checks include:

* Staged rows present
* Identity fields present
* Red and blue fighters distinct
* `winner_id` matches either `r_id` or `b_id`
* `winner` matches either `r_name` or `b_name`
* Fight IDs not already in master
* Fight IDs unique in staged rows
* Date parseable
* Fight metadata present (`location`, `division`, `title_fight`, `total_rounds`)
* `title_fight` is `1` for yes or `0` for no
* `total_rounds` is plausible (`3` or `5`)
* Finish round plausible
* Match time plausible
* Landed stats not greater than attempted stats

Warning checks include:

* Existing master event identity consistency
* Percentage values in range
* Fighter profile plausibility

Result:

```text
final_review_pass = True | False
```

`final_review_pass` is computed from blocking checks only.

---

# Runner 11

## run_append_staged_to_master.py

Purpose:
Append staged rows into master dataset.

Inputs:

* data/master/ufc_master.parquet
* data/staging/ufc_staged_master_rows_profiled.parquet
* data/audits/ufc_append_precheck.parquet
* data/audits/ufc_staged_final_review.parquet

Outputs:

* data/master/ufc_master.parquet
* data/backups/ufc_master_backup_<run_id>.parquet
* data/audits/ufc_append_audit.parquet

Safety Controls:

* Refuse append if `append_ready=False`.
* Refuse append if final review artifact is missing.
* Refuse append if `final_review_pass=False`.
* Refuse duplicate fight IDs.
* Backup master before append.

---

# Orchestrator

## run_ingest_single_event.py

Purpose:
Stage and review one selected UFCStats event.

Workflow:

```text
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
Master Column Validation
  ↓
Append Precheck
  ↓
Final Staged Review
```

Important rule:

```text
run_ingest_single_event.py does not append to master.
```

Modes:

```text
full  = all fights and all staged fighters
smoke = one fight and two fighters
```

Environment variables / CLI:

```text
EVENT_ID or --event-id
INGEST_MODE or --mode
MAX_FIGHTS or --max-fights
MAX_FIGHTERS or --max-fighters
```

Example:

```bash
EVENT_ID=1e75e6c9de99fa76 INGEST_MODE=full \
python -m pipeline.data_maintenance.run_ingest_single_event
```

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
