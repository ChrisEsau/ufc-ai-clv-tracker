# UFC Data Maintenance Dashboard Architecture

## Purpose

The Data Maintenance Dashboard is the UFC ingestion control tower. It is designed for a human-in-the-loop workflow where GitHub Actions runs the ingestion pipeline, Streamlit displays the resulting artifacts, and the operator approves append only after review.

Primary staged-ingestion workflow:

```text
EVENT DISCOVERY
  ↓
SINGLE EVENT INGESTION
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
HUMAN APPEND APPROVAL
  ↓
APPEND
```

Single-event ingestion must **not** append to master. It stages and reviews data only. The append action remains a separate operator-approved workflow.

---

## Current Section Order

The Data Maintenance dashboard currently uses four top-level expanders:

```text
Dataset Health
Event Discovery
Final Staged Review
Audit History
```

Older internal sections such as Fight Scrape, Enrichment, Validation Gate, and Append Status are no longer separate top-level dashboard sections. Their key outputs are consolidated into Final Staged Review.

---

## Dataset Health

Responsibilities:

* Launch dataset status workflow.
* Display master row/column counts.
* Display event/fighter counts.
* Display date, duplicate, and result health.
* Display workflow status for the launched dataset-status workflow.

Primary workflow:

```text
Run Dataset Status
  ↓
run-dataset-status.yml
  ↓
python -m pipeline.data_maintenance.run_dataset_status
```

---

## Event Discovery

Responsibilities:

* Launch UFCStats event check.
* Display missing completed UFCStats events.
* Let the operator select a missing event.
* Let the operator choose ingestion mode.
* Launch single-event ingestion.
* Display workflow status for event check and single-event ingestion.

Primary workflow:

```text
Run Event Check
  ↓
Select Missing Event
  ↓
Choose Mode: full | smoke
  ↓
Ingest Selected Event
  ↓
dm-ingest-single-event.yml
```

### Ingestion Modes

```text
full  = scrape all fights and all staged fighters for the selected event
smoke = scrape one fight and two fighters for lightweight validation
```

The selected event button dispatches `dm-ingest-single-event.yml` with:

```text
event_id
mode
```

---

## Final Staged Review

Final Staged Review is the main staging and append-decision workspace.

Responsibilities:

* Display ingestion output artifact status.
* Display current staged event summary.
* Display staged row preview.
* Display append precheck summary and failed checks.
* Display final review summary and failed checks.
* Display append decision status.
* Require human confirmation before append.
* Launch append workflow only when gates pass and the operator confirms.
* Display workflow status for append precheck/final review and append workflows.

Append is enabled only when:

```text
append_ready == True
final_review_pass == True
human_confirmation == True
```

The Final Staged Review expander should show, at minimum:

* Fight rows staged.
* Fight details staged.
* Mapped master rows.
* Derived/enriched rows.
* Fighter profiles.
* Profiled master rows.
* Append precheck artifact.
* Final staged review artifact.
* Current staged event name/date/event_id.
* Staged fight preview.
* Append allowed / blocked state.

---

## Audit History

Responsibilities:

* Show audit artifact availability.
* Show audit artifact last-modified times.
* Let the operator inspect audit artifacts.
* Include the final staged review artifact in audit selection.

---

## Workflow Status UI

Dashboard workflow buttons should display a status panel after launch.

Workflow status uses GitHub Actions API polling and should show:

* Workflow label.
* GitHub status/conclusion.
* Branch.
* Run number.
* Start/update timestamps.
* Progress bar.
* Link to the GitHub run.
* Manual refresh button.

Supported status mapping:

```text
queued                  → Queued
in_progress             → Running
completed + success     → Succeeded
completed + failure     → Failed
completed + cancelled   → Cancelled
completed + timed_out   → Timed out
```

The dashboard stores the last launched workflow in Streamlit session state and then queries the latest matching GitHub workflow run for the configured branch.

Required Streamlit secrets:

```text
GITHUB_OWNER
GITHUB_REPO
GITHUB_TOKEN
GITHUB_BRANCH
```

`GITHUB_BRANCH` should be set to the active working branch, usually `dev` for development workflows.

---

## Dashboard Safety Rules

* Dashboard launches workflows; pipeline modules perform business logic.
* Single-event ingestion never appends.
* Append button belongs inside Final Staged Review.
* Append button remains disabled until precheck and final review pass.
* Human confirmation is required before append dispatch.
* Generated parquet artifacts are ignored by default and must be force-added in workflows when they are intended to be committed.

---

## Long-Term Goal

The Data Maintenance Dashboard should function as a CI/CD-style ingestion control tower for UFC data operations:

```text
Streamlit Button
  ↓
GitHub workflow_dispatch
  ↓
Python pipeline module
  ↓
Canonical data/audit artifacts
  ↓
Dashboard review
  ↓
Human-approved append
```
