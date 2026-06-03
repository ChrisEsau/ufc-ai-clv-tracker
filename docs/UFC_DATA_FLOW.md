# UFC Data Flow

## Data Maintenance / Ingestion Flow

Current non-append staging flow:

```text
UFCStats
    ↓
Event Check
    ↓
Missing Events
    ↓
Single Event Ingestion
    ↓
Fight Scrape
    ↓
Fight Detail Scrape
    ↓
Master Mapper
    ↓
Derived Stats
    ↓
Profile Enrichment
    ↓
Master Column Validation
    ↓
Append Precheck
    ↓
Final Staged Review
    ↓
Dashboard Human Review
```

Append flow:

```text
Append Precheck PASS
    ↓
Final Staged Review PASS
    ↓
Human confirmation in dashboard
    ↓
Append Workflow
    ↓
Master Backup
    ↓
ufc_master.parquet
    ↓
Append Audit
```

Important rule:

```text
Single Event Ingestion never appends to master.
```

---

## Dashboard Workflow Flow

```text
Streamlit Button
    ↓
GitHub workflow_dispatch
    ↓
GitHub Actions run
    ↓
Pipeline module
    ↓
Committed canonical artifacts
    ↓
Dashboard status/review panels
```

The dashboard can poll GitHub Actions runs for workflow status using configured GitHub secrets.

---

## Prediction Flow

```text
ufc_master.parquet
    ↓
Feature Engineering
    ↓
Historical Feature Store
    ↓
Current Fighter Feature Store
    ↓
Prediction Model
    ↓
Betting Board
```

---

## CLV Flow

```text
Market Odds
    ↓
Snapshots
    ↓
Line Movement
    ↓
Closing Lines
    ↓
CLV Results
```
