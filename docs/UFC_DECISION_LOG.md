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
Final Staged Review
    ↓
Human Confirmation
    ↓
Append
```

Append requires both:

```text
append_ready == True
final_review_pass == True
```

---

### Dashboard Philosophy

Dashboard launches workflows.

Pipeline performs work.

Dashboard does not contain business logic.

---

### Data Maintenance Layout

Current top-level section order:

```text
Dataset Health
Event Discovery
Final Staged Review
Audit History
```

Fight scrape, enrichment, validation, final review, append readiness, and append controls are consolidated into the Final Staged Review workspace.

---

### UFC Betting Defaults

Current production defaults:

```text
EV Threshold = $50
Confidence Threshold = 70%
Odds Range = -250 to +400
Half Kelly staking
```
