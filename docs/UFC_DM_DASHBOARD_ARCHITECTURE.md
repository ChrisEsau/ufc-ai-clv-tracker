# UFC Data Maintenance Dashboard Architecture

## Purpose

The Data Maintenance Dashboard serves as the UFC ingestion control tower.

Primary workflow:

```text
SCRAPE
  ↓
MAP
  ↓
ENRICH
  ↓
VALIDATE
  ↓
APPEND
```

---

## Section Order

### Dataset Health

Responsibilities:

* Dataset row counts
* Event counts
* Dataset coverage
* Dataset status

---

### Workflow Status

Responsibilities:

* GitHub workflow state
* Last run times
* Success/failure status
* Workflow links

---

### Event Discovery

Responsibilities:

* UFCStats event comparison
* Missing event detection
* Event selection
* Ingestion launch

Primary workflow:

```text
Run Event Check
  ↓
Select Event
  ↓
Ingest Event
```

---

### Fight Scrape

Responsibilities:

* Scrape monitoring
* Fight-level audit review

---

### Enrichment

Responsibilities:

* Fighter profile enrichment
* Derived statistics generation
* Mapping review

---

### Validation Gate

Responsibilities:

* Schema validation
* Required field validation
* Duplicate detection
* Append readiness

Output:

```text
append_ready
```

---

### Audit History

Responsibilities:

* Scrape audits
* Validation audits
* Append audits

---

### Append Status

Responsibilities:

* Append readiness display
* Append workflow launch
* Append history

Append button must remain disabled unless:

```text
append_ready == True
```

---

## Long-Term Goal

The Data Maintenance Dashboard should function as a CI/CD-style ingestion control tower for UFC data operations.
