# UFC Development Rules

## Change Approval Rule

Before coding, moving files, committing, or creating a PR, the agent must first provide a concise summary of the intended implementation and receive explicit user approval.

Read-only review, repository scans, recommendations, and project planning are allowed without approval only when they do not modify files.

Approval for one implementation scope does not imply approval for unrelated follow-up changes.

---

## Paths

Always use:

```python
from pipeline.common.paths import ...
```

Never hardcode file paths.

---

## Module Execution

Preferred execution:

```bash
python -m pipeline.<workspace>.<runner>
```

Avoid direct script execution whenever possible.

---

## Master Dataset

Authoritative dataset:

```text
data/master/ufc_master.parquet
```

The 128-column schema is authoritative.

Never modify schema without updating:

* UFC_MASTER_SCHEMA.md
* Mapping logic
* Validation logic

---

## URL and ID Rules

Keep:

* URLs
* External identifiers

inside:

* Scrapers
* Staging artifacts
* Audit artifacts

Remove URLs at mapper boundary.

Retain IDs only inside:

* Master dataset
* Feature stores
* Modeling layers

---

## Append Rules

Never append directly to master.

Required sequence:

Event Discovery
→ Single Event Ingestion
→ Validation
→ Append Precheck
→ Final Staged Review
→ Human Confirmation
→ Append

Single Event Ingestion must never append to master.

Append must be blocked unless:

```text
append_ready == True
final_review_pass == True
```

---

## Dashboard Rules

Dashboard launches workflows.

Dashboard should not contain ingestion logic.

Business logic belongs in pipeline modules.

---

## GitHub Actions

Workflows execute pipeline runners.

Streamlit acts as orchestration layer.

GitHub Actions acts as execution layer.

---

## Data Architecture

Historical Feature Store:

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
```

Live Feature Store:

```text
data/features/ufc_current_fighter_features.parquet
```

Do not mix responsibilities between stores.
