# UFC Docs Status Index

_Last updated: 2026-06-09_

## Purpose

This index classifies markdown documentation files as current, active specs, bridge docs, or legacy candidates after the V2 architecture review.

Use this file before editing or relying on older architecture documents.

---

## Status Legend

```text
CURRENT_HANDOFF      Read first for current working context.
ACTIVE_SPEC          Still valid as a contract or stable architecture reference.
ACTIVE_AREA_DOC      Still useful for a specific subsystem.
BRIDGE               Useful, but documents a transition state.
LEGACY_CANDIDATE     Superseded or outdated; safe candidate for _LEGACY rename after full-file-safe move.
ARCHIVE              Already archived or historical.
UNKNOWN_REVIEW       Needs more content review before classification.
```

---

## Read First

| File | Status | Notes |
|---|---|---|
| `docs/V2_PRODUCTION_ARCHITECTURE_V3.md` | CURRENT_HANDOFF | Primary current V2 architecture and gap document. |
| `docs/V3_FUTURE_RESEARCH_ROADMAP.md` | CURRENT_HANDOFF | Future feature/model/research roadmap. |
| `docs/DOCS_STATUS_INDEX.md` | CURRENT_HANDOFF | This index. |

---

## Active Specs / Architecture References

| File | Status | Notes |
|---|---|---|
| `docs/UFC_OUTCOME_SCHEMA_SPEC.md` | ACTIVE_SPEC | Canonical outcome-level schema specification. |
| `docs/MODEL_REGISTRY_ARCHITECTURE.md` | ACTIVE_SPEC | Model registry design remains relevant. |
| `docs/MODEL_ADAPTER_ARCHITECTURE.md` | ACTIVE_SPEC | Generic model adapter design remains relevant. |
| `docs/UFC_MARKET_PIPELINE_V2_ARCHITECTURE.md` | ACTIVE_SPEC | Market V2 architecture reference. |
| `docs/UFC_MASTER_SCHEMA.md` | ACTIVE_SPEC | Master schema contract. |
| `docs/UFC_INGESTION_PIPELINE_REGISTRY.md` | ACTIVE_SPEC | Ingestion pipeline registry. |
| `docs/UFC_DM_DASHBOARD_ARCHITECTURE.md` | ACTIVE_AREA_DOC | Data Maintenance dashboard architecture. |
| `docs/UFC_BANKROLL_ARCHITECTURE.md` | ACTIVE_AREA_DOC | Bankroll architecture; may need future outcome-native update. |
| `docs/UFC_LINE_MOVEMENT_CLV_ARCHITECTURE.md` | ACTIVE_AREA_DOC | CLV/line movement reference; implementation may still be legacy. |
| `docs/UFC_PROP_MARKET_SCHEMA.md` | ACTIVE_SPEC | Prop market schema reference. |
| `docs/UFC_PROP_MODEL_ARCHITECTURE.md` | ACTIVE_AREA_DOC | Prop model target architecture. |
| `docs/TRAINING_FRAMEWORK_ARCHITECTURE.md` | ACTIVE_AREA_DOC | Training framework architecture. |
| `docs/UFC_FEATURE_LAYER_ARCHITECTURE.md` | ACTIVE_AREA_DOC | Feature layer reference; check against current feature-view architecture before editing. |
| `docs/UFC_FEATURE_STORE_ARCHITECTURE.md` | ACTIVE_AREA_DOC | Feature store reference; check against current feature-view architecture before editing. |
| `docs/UFC_BETTING_BOARD_ARCHITECTURE.md` | ACTIVE_AREA_DOC | Betting board architecture; verify V2 outcome consumption before changes. |
| `docs/UFC_GITHUB_WORKFLOW_REGISTRY.md` | ACTIVE_AREA_DOC | Workflow registry; should be updated with V2 workflow status. |
| `docs/UFC_REPOSITORY_STRUCTURE.md` | ACTIVE_AREA_DOC | Repo structure reference. |
| `docs/UFC_PROJECT_OVERVIEW.md` | ACTIVE_AREA_DOC | General project overview. |
| `docs/UFC_ARTIFACT_REGISTRY.md` | ACTIVE_AREA_DOC | Artifact registry; should be reconciled with V2/V3 artifact map. |
| `docs/UFC_DATA_FLOW.md` | ACTIVE_AREA_DOC | Data-flow reference; should be reconciled with current V2 feature-view path. |
| `docs/UFC_DECISION_LOG.md` | ACTIVE_AREA_DOC | Decision history remains useful. |
| `docs/ui/ui.md` | ACTIVE_AREA_DOC | UI reference. |

---

## Bridge / Transitional Docs

| File | Status | Notes |
|---|---|---|
| `docs/UFC_PREDICTION_PIPELINE_V2.md` | BRIDGE | Approved target architecture, but older status text says code not implemented. Keep as design reference but read V3 handoff first. |
| `docs/ROLLING_NOTEBOOK_MIGRATION_PLAN.md` | BRIDGE | Useful migration history for old rolling feature notebook; not current source of truth. |
| `README.md` | BRIDGE | Should be reviewed and updated to point to V2/V3 docs. |
| `AGENTS.md` | BRIDGE | Agent instructions; keep unless contradicted by current project rules. |

---

## Legacy Candidates For `_LEGACY` Rename

These appear superseded by `docs/V2_PRODUCTION_ARCHITECTURE_V3.md` or completed implementation work.

Do not delete contents. Prefer safe rename once full file content can be moved without truncation risk.

| Current File | Suggested Rename | Reason |
|---|---|---|
| `docs/V2 PRODUCTION ARCHITECTURE.md` | `docs/V2 PRODUCTION ARCHITECTURE_LEGACY.md` | Superseded by `V2 PRODUCTION ARCHITECTURE 1` and then V3 handoff. |
| `docs/V2 PRODUCTION ARCHITECTURE 1.md` | `docs/V2 PRODUCTION ARCHITECTURE 1_LEGACY.md` | Superseded by `V2_PRODUCTION_ARCHITECTURE_V3.md`. |
| `docs/UFC_FEATURE_BUILDER_V2_DRAFT.md` | `docs/UFC_FEATURE_BUILDER_V2_DRAFT_LEGACY.md` | Draft doc; feature-view architecture now exists. |
| `docs/CURRENT_PROJECT_HANDOFF.md` | `docs/CURRENT_PROJECT_HANDOFF_LEGACY.md` | Older handoff uses old rolling-feature state as current. |
| `docs/CURRENT_REPO_STATE_FOR_FUTURE_CHATS.md` | `docs/CURRENT_REPO_STATE_FOR_FUTURE_CHATS_LEGACY.md` | Older current-state handoff now superseded by V3 handoff. |
| `docs/UFC_PREDICTION_IMPLEMENTATION_PLAN.md` | `docs/UFC_PREDICTION_IMPLEMENTATION_PLAN_LEGACY.md` | Implementation plan says core V2 files are not implemented; many now exist. |

---

## Already Archived

| File | Status | Notes |
|---|---|---|
| `archive/UFC_AI_Betting_Platform_Vision.md` | ARCHIVE | Historical platform vision. |
| `archive/README.md` | ARCHIVE | Archive index. |
| `archive/migration_validation/README.md` | ARCHIVE | Historical migration validation notes. |

---

## Rename Safety Note

GitHub file renames through the contents API require creating a new file with the complete old content and deleting the original. Some markdown files are long and may be truncated by connector reads. For those files, do not perform create/delete rename unless full content is confirmed available.

Preferred local rename command:

```bash
git mv "docs/V2 PRODUCTION ARCHITECTURE.md" "docs/V2 PRODUCTION ARCHITECTURE_LEGACY.md"
git mv "docs/V2 PRODUCTION ARCHITECTURE 1.md" "docs/V2 PRODUCTION ARCHITECTURE 1_LEGACY.md"
git mv docs/UFC_FEATURE_BUILDER_V2_DRAFT.md docs/UFC_FEATURE_BUILDER_V2_DRAFT_LEGACY.md
git mv docs/CURRENT_PROJECT_HANDOFF.md docs/CURRENT_PROJECT_HANDOFF_LEGACY.md
git mv docs/CURRENT_REPO_STATE_FOR_FUTURE_CHATS.md docs/CURRENT_REPO_STATE_FOR_FUTURE_CHATS_LEGACY.md
git mv docs/UFC_PREDICTION_IMPLEMENTATION_PLAN.md docs/UFC_PREDICTION_IMPLEMENTATION_PLAN_LEGACY.md
```

Then commit:

```bash
git commit -m "Mark superseded docs as legacy"
```

---

## Next Documentation Cleanup Steps

1. Update `README.md` to point first to `docs/V2_PRODUCTION_ARCHITECTURE_V3.md` and `docs/V3_FUTURE_RESEARCH_ROADMAP.md`.
2. Update `docs/UFC_GITHUB_WORKFLOW_REGISTRY.md` with the current V2 workflow status.
3. Reconcile `docs/UFC_ARTIFACT_REGISTRY.md` against the artifact map in V3 handoff.
4. Reconcile `docs/UFC_DATA_FLOW.md` against the feature-view architecture.
5. Rename the confirmed legacy candidates using `git mv` locally or a full-content-safe GitHub move.
