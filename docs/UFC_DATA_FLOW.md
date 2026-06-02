# UFC Data Flow

## Ingestion Flow

```text
UFCStats
    ↓
Event Check
    ↓
Missing Events
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
Validation
    ↓
Append Precheck
    ↓
Append
    ↓
ufc_master.parquet
```

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
