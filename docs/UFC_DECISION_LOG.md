# UFC Decision Log

## Locked Decisions

### Change Approval Before Coding

Before coding, moving files, committing, or creating a PR, the agent must summarize the intended implementation and receive explicit user approval.

Read-only scans, reports, and planning can proceed without file changes.

---

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

---

### Model Lab Pause

Model Lab is paused while development focus moves to the Betting Board.

Current locked Model Lab state:

```text
Read-only artifact and diagnostics dashboard
Canonical paths via pipeline.common.paths
Production model artifacts under models/UFC_Model_v5_Experiment/
Feature artifacts under data/features/
Prediction artifacts under data/predictions/
Live audit artifacts under data/audits/
```

Deferred Model Lab work:

```text
Backtest runner
Threshold / ROI sweep
Calibration bins
Recent-era validation
Feature drift report
Model comparison summary
Model Lab workflow dispatch
```

Do not add retraining, promotion, rollback, or ensemble controls until the Betting Board phase is complete and the historical evaluation layer exists.

---

### Phase Status After Repository Cleanup

Completed and verified:

```text
Phase 0 cleanup / archive stabilization
Phase 1.1 Betting Board artifact diagnostics review
Phase 1.2 Odds side-mapping validation
Phase 1.3 Selected-event workflow validation
Phase 2 Data Maintenance validation
```

Deferred by operator decision:

```text
Phase 1.4 Production-vs-scenario comparison polish
Phase 1.5 Betting Board operator checklist
```

Current next focus:

```text
Phase 3 Line Movement / CLV tracking
```

