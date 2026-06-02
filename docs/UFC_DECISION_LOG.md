# UFC Decision Log

## Locked Decisions

### Master Schema

128-column schema is authoritative.

---

### Path Management

All paths must originate from:

```python
pipeline.common.paths
```

---

### Module Execution

Preferred execution:

```bash
python -m pipeline...
```

---

### URL Boundary

Keep URLs in:

* scrapers
* staging
* audits

Remove URLs at mapper boundary.

Retain IDs only beyond mapper.

---

### Append Protection

No direct append to master.

Required flow:

```text
Validation
    ↓
Append Precheck
    ↓
Append
```

---

### Dashboard Philosophy

Dashboard launches workflows.

Pipeline performs work.

Dashboard does not contain business logic.

---

### Data Maintenance Layout

Preferred section order:

```text
Dataset Health
Workflow Status
Event Discovery
Fight Scrape
Enrichment
Validation Gate
Audit History
Append Status
```

---

### UFC Betting Defaults

Current production defaults:

```text
EV Threshold = $50
Confidence Threshold = 70%
Odds Range = -250 to +400
Half Kelly staking
```
