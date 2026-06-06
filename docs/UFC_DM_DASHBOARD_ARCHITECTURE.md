# UFC Data Maintenance Dashboard Architecture

## Purpose

The Data Maintenance Dashboard is the UFC ingestion control tower. It is designed for a human-in-the-loop workflow where GitHub Actions runs the ingestion pipeline, Streamlit displays the resulting artifacts, and the operator approves append only after review.

Primary staged-ingestion workflow:

```text
EVENT DISCOVERY
  ↓
FULL SINGLE EVENT INGESTION
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
SEPARATE APPEND ACTION
  ↓
APPEND
```

Single-event ingestion must **not** append to master. It stages and reviews data only. The append action remains a separate workflow button that is enabled only after machine gates pass.

---

## Current Layout Direction

The Data Maintenance dashboard now follows the mockup-led operational control-plane layout rather than a strict four-expander structure. The page should present visible dashboard panels for:

```text
Top Summary KPIs
Dataset Status Overview
Event Discovery
Ingestion Pipeline Status
Data Quality Summary
Recent Ingestion History / Audit Details
Append Approval
```

Append Approval remains a separate final action at the bottom of the workflow-oriented page.

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
* Launch single-event ingestion.
* Display workflow status for event check and single-event ingestion.

Primary workflow:

```text
Run Event Check
  ↓
Select Missing Event
  ↓
Ingest Selected Event
  ↓
dm-ingest-single-event.yml
```

### Ingestion Mode

Single-event ingestion always runs the full selected event. The selected event button dispatches `dm-ingest-single-event.yml` with:

```text
event_id
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
* Launch append workflow only when gates pass and the operator confirms.
* Display workflow status for append precheck/final review and append workflows.

Append is enabled only when:

```text
append_ready == True
final_review_pass == True
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
