# UFC Repo Archive Candidate Plan

_Last updated: 2026-06-09_

## Purpose

This document is a practical cleanup plan for identifying files that appear unused by the current V2 architecture and are candidates to move into `archive/`.

It is based on the V2 architecture review, dashboard dependency trace, workflow trace, and repository search.

This is **not** a delete plan. The recommended action is `git mv` into archive folders so history is preserved and files can be restored if needed.

---

## Current V2 Source Of Truth

Read these docs first:

```text
docs/V2_PRODUCTION_ARCHITECTURE_V3.md
docs/V3_FUTURE_RESEARCH_ROADMAP.md
docs/DOCS_STATUS_INDEX.md
```

Current V2 production spine:

```text
pipeline.features.run_build_fighter_state
pipeline.features.run_build_feature_view
pipeline.training.run_train_model
pipeline.prediction.run_refresh_upcoming_events
pipeline.prediction.run_build_live_card
pipeline.modeling.run_prediction
pipeline.market.run_market_update_v2
pipeline.betting.run_betting_outcomes_v2
dashboard.py
```

Core V2 workflows:

```text
.github/workflows/run-live-features-v2.yml
.github/workflows/run-prediction-v2.yml
.github/workflows/run-market-v2.yml
.github/workflows/run-betting-outcomes-v2.yml
```

---

# Archive Rules

## Safe archive candidates

A file is a good archive candidate if it is:

- a legacy root-level runner replaced by a package/module runner,
- a workflow that calls legacy root-level scripts,
- a superseded handoff or implementation-plan doc,
- already described as historical/draft/legacy,
- not imported by the current V2 module path or dashboard.

## Do not archive yet

Do not archive files that are:

- imported by V2 modules,
- imported by dashboard tabs,
- used by Data Maintenance workflows,
- compatibility shims still feeding current UI,
- shared root utilities still called by V2 code.

---

# High-Confidence Archive Candidates

## 1. Legacy root scripts

These are root-level scripts that appear superseded by package/module runners.

| File | Suggested Archive Path | Reason |
|---|---|---|
| `run_live_prediction.py` | `archive/legacy_root_scripts/run_live_prediction.py` | Legacy live prediction path; V2 uses `pipeline.modeling.run_prediction`. |
| `run_market_update.py` | `archive/legacy_root_scripts/run_market_update.py` | Legacy market update path; V2 uses `pipeline.market.run_market_update_v2`. |
| `run_clv_tracker.py` | `archive/legacy_root_scripts/run_clv_tracker.py` | Legacy root CLV runner; CLV should be reconciled with outcome-level V2 artifacts. |
| `run_master_column_inventory.py` | `archive/legacy_root_scripts/run_master_column_inventory.py` | Inventory/debug runner; similar workflows already exist in archive and Data Maintenance has current validation tooling. |

### Commands

```bash
mkdir -p archive/legacy_root_scripts

git mv run_live_prediction.py archive/legacy_root_scripts/
git mv run_market_update.py archive/legacy_root_scripts/
git mv run_clv_tracker.py archive/legacy_root_scripts/
git mv run_master_column_inventory.py archive/legacy_root_scripts/
```

---

## 2. Legacy / bridge workflows

These workflows appear superseded by V2 or still call legacy scripts.

| File | Suggested Archive Path | Reason |
|---|---|---|
| `.github/workflows/run-live-prediction.yml` | `archive/.github/workflows/run-live-prediction.yml` | Legacy live prediction workflow; V2 uses `run-prediction-v2.yml`. |
| `.github/workflows/run-model-predictions.yml` | `archive/.github/workflows/run-model-predictions.yml` | Legacy model prediction workflow; V2 uses `run-prediction-v2.yml`. |
| `.github/workflows/run-market-update.yml` | `archive/.github/workflows/run-market-update.yml` | Legacy market workflow; V2 uses `run-market-v2.yml`. |
| `.github/workflows/run-betting-decision.yml` | `archive/.github/workflows/run-betting-decision.yml` | Legacy betting decision workflow; V2 uses `run-betting-outcomes-v2.yml`. |
| `.github/workflows/run-betting-board-selected-event.yml` | `archive/.github/workflows/run-betting-board-selected-event.yml` | Builds live card but then calls legacy scripts. Replace with future V2 selected-event workflow. |
| `.github/workflows/run-current-fighter-features.yml` | `archive/.github/workflows/run-current-fighter-features.yml` | Legacy current-fighter feature artifact. V2 target is fighter-state + feature-view architecture. |

### Commands

```bash
mkdir -p archive/.github/workflows

git mv .github/workflows/run-live-prediction.yml archive/.github/workflows/
git mv .github/workflows/run-model-predictions.yml archive/.github/workflows/
git mv .github/workflows/run-market-update.yml archive/.github/workflows/
git mv .github/workflows/run-betting-decision.yml archive/.github/workflows/
git mv .github/workflows/run-betting-board-selected-event.yml archive/.github/workflows/
git mv .github/workflows/run-current-fighter-features.yml archive/.github/workflows/
```

---

## 3. Superseded docs

These docs are superseded by the current V2/V3 handoff and docs status index.

| File | Suggested Archive Path | Reason |
|---|---|---|
| `docs/V2 PRODUCTION ARCHITECTURE.md` | `archive/docs/V2 PRODUCTION ARCHITECTURE.md` | Historical V2 snapshot. |
| `docs/V2 PRODUCTION ARCHITECTURE 1.md` | `archive/docs/V2 PRODUCTION ARCHITECTURE 1.md` | Superseded by `V2_PRODUCTION_ARCHITECTURE_V3.md`. |
| `docs/UFC_FEATURE_BUILDER_V2_DRAFT.md` | `archive/docs/UFC_FEATURE_BUILDER_V2_DRAFT.md` | Draft feature-builder concept; feature-view architecture now exists. |
| `docs/CURRENT_PROJECT_HANDOFF.md` | `archive/docs/CURRENT_PROJECT_HANDOFF.md` | Older current-state handoff. |
| `docs/CURRENT_REPO_STATE_FOR_FUTURE_CHATS.md` | `archive/docs/CURRENT_REPO_STATE_FOR_FUTURE_CHATS.md` | Older current-state handoff. |
| `docs/UFC_PREDICTION_IMPLEMENTATION_PLAN.md` | `archive/docs/UFC_PREDICTION_IMPLEMENTATION_PLAN.md` | Implementation plan says many V2 components are not implemented; they now exist. |

### Commands

```bash
mkdir -p archive/docs

git mv "docs/V2 PRODUCTION ARCHITECTURE.md" archive/docs/
git mv "docs/V2 PRODUCTION ARCHITECTURE 1.md" archive/docs/
git mv docs/UFC_FEATURE_BUILDER_V2_DRAFT.md archive/docs/
git mv docs/CURRENT_PROJECT_HANDOFF.md archive/docs/
git mv docs/CURRENT_REPO_STATE_FOR_FUTURE_CHATS.md archive/docs/
git mv docs/UFC_PREDICTION_IMPLEMENTATION_PLAN.md archive/docs/
```

---

# Keep Active / Do Not Archive

## Root / top-level files

| File | Reason |
|---|---|
| `dashboard.py` | Current Streamlit app entrypoint. |
| `README.md` | Should be updated, not archived. |
| `AGENTS.md` | Project/agent instructions. |
| `requirements.txt` | Runtime dependency file. |

## Root utility modules that look old but are still bridge/shared

Do not archive until imports are removed and V2 replacements exist.

| File | Reason |
|---|---|
| `ufc_odds_utils.py` | Market provider still uses odds helper logic. |
| `ufc_pipeline_utils.py` | Shared helper functions used across older and newer modules. |
| `ufc_feature_engineering.py` | Still used for V5 engineered features and feature-view compatibility. |
| `pipeline_config.py` | Check imports before moving; likely legacy/shared config. |
| `ufc_clv_utils.py` | CLV/line movement may still use it; verify before archive. |

## Dashboard and utilities

Do not archive yet. Some are compatibility shims but active UI still depends on them.

```text
tabs/betting_board_v2.py
tabs/data_maintenance.py
tabs/line_movement.py
tabs/bankroll.py
tabs/model_lab.py
utils/data_loader.py
utils/betting_outcomes_adapter.py
utils/betting_board_artifacts.py
utils/bankroll_artifacts.py
utils/model_lab_artifacts.py
utils/github_actions.py
utils/sidebar.py
utils/theme.py
utils/ui/*
utils/dm_*.py
```

Reason:

- Betting Board V2 is active.
- Data Maintenance is active.
- Line Movement and Model Lab are stale, but still dashboard tabs.
- Compatibility shims should be removed only after dashboard consumers are migrated.

## Pipeline package

Do not archive active package modules under:

```text
pipeline/common/
pipeline/data_maintenance/
pipeline/features/
pipeline/training/
pipeline/prediction/
pipeline/modeling/
pipeline/market/
pipeline/betting/
pipeline/clv/
```

Some modules may later be deprecated, but the package structure is the current standard. Archive only after import tracing confirms no usage.

## Scrapers

Do not archive:

```text
scrapers/
```

Reason: Data Maintenance and event ingestion still depend on scraper modules.

---

# Manual Review Before Archive

These may be legacy, but should not be moved until import/workflow checks are complete.

## Root utilities

```text
ufc_clv_utils.py
pipeline_config.py
utils/panels.py
utils/betting_board_rules.py
```

Reason:

These may be used by CLV, dashboard, or legacy compatibility code. Confirm with grep before moving.

## CLV-related workflows and modules

```text
.github/workflows/run-clv-tracker.yml
pipeline/clv/line_movement.py
pipeline/clv/closing_lines.py
tabs/line_movement.py
```

Reason:

The CLV tab is not fully V2 outcome-native, but CLV remains an active product area. Do not archive until a V2 CLV replacement exists.

## Bankroll/manual bet workflows

```text
.github/workflows/run-append-manual-bet.yml
.github/workflows/run-settle-manual-bet.yml
.github/workflows/run-bankroll-status.yml
.github/workflows/run-save-risk-settings.yml
```

Reason:

These are not core prediction V2, but bankroll/manual bet tracking remains active.

## Data Maintenance workflows

Keep unless specifically superseded:

```text
.github/workflows/dm-ingest-single-event.yml
.github/workflows/run-refresh-upcoming-events.yml
.github/workflows/run-dataset-status.yml
.github/workflows/run-ufcstats-event-check.yml
.github/workflows/run-ufcstats-fight-scrape.yml
.github/workflows/run-ufcstats-fight-detail-scrape.yml
.github/workflows/run-staged-master-mapper.yml
.github/workflows/run-staged-derived-stats-transformer.yml
.github/workflows/run-fighter-profile-enrichment.yml
.github/workflows/run-master-column-validation.yml
.github/workflows/run-append-precheck-validation.yml
.github/workflows/run-staged-final-review.yml
.github/workflows/run-append-staged-to-master.yml
.github/workflows/run-smoke-tests.yml
```

Reason:

These support ingestion, validation, and master data maintenance.

---

# Recommended Future V2 Workflow Additions

Instead of using legacy selected-event workflows, add new workflows later:

```text
.github/workflows/run-build-fighter-state-v2.yml
.github/workflows/run-build-feature-view-v2.yml
.github/workflows/run-selected-event-v2.yml
```

Target selected-event V2 order:

```bash
python -m pipeline.prediction.run_refresh_upcoming_events
python -m pipeline.prediction.run_build_live_card --event-id "$EVENT_ID"
python -m pipeline.features.run_build_fighter_state
python -m pipeline.features.run_build_feature_view --config configs/feature_views/moneyline_base.yaml
python -m pipeline.modeling.run_prediction --model-id moneyline_xgboost_v5
python -m pipeline.market.run_market_update_v2
python -m pipeline.betting.run_betting_outcomes_v2
```

---

# Full Suggested Cleanup Command Block

Run from repo root on `dev` after pulling latest.

```bash
git checkout dev
git pull origin dev

mkdir -p archive/legacy_root_scripts archive/.github/workflows archive/docs

# Legacy root scripts
git mv run_live_prediction.py archive/legacy_root_scripts/
git mv run_market_update.py archive/legacy_root_scripts/
git mv run_clv_tracker.py archive/legacy_root_scripts/
git mv run_master_column_inventory.py archive/legacy_root_scripts/

# Legacy workflows
git mv .github/workflows/run-live-prediction.yml archive/.github/workflows/
git mv .github/workflows/run-model-predictions.yml archive/.github/workflows/
git mv .github/workflows/run-market-update.yml archive/.github/workflows/
git mv .github/workflows/run-betting-decision.yml archive/.github/workflows/
git mv .github/workflows/run-betting-board-selected-event.yml archive/.github/workflows/
git mv .github/workflows/run-current-fighter-features.yml archive/.github/workflows/

# Superseded docs
git mv "docs/V2 PRODUCTION ARCHITECTURE.md" archive/docs/
git mv "docs/V2 PRODUCTION ARCHITECTURE 1.md" archive/docs/
git mv docs/UFC_FEATURE_BUILDER_V2_DRAFT.md archive/docs/
git mv docs/CURRENT_PROJECT_HANDOFF.md archive/docs/
git mv docs/CURRENT_REPO_STATE_FOR_FUTURE_CHATS.md archive/docs/
git mv docs/UFC_PREDICTION_IMPLEMENTATION_PLAN.md archive/docs/

# Review result
git status

# Optional quick import sanity check
python -m compileall pipeline tabs utils dashboard.py

# Commit
git add -A
git commit -m "Archive legacy scripts workflows and superseded docs"
git push origin dev
```

---

# Post-Move Validation Checklist

After moving files, verify:

```bash
python -m compileall pipeline tabs utils dashboard.py
```

Then verify dashboard still imports:

```bash
python - <<'PY'
import dashboard
print('dashboard import ok')
PY
```

Then verify V2 module imports:

```bash
python - <<'PY'
import pipeline.modeling.run_prediction
import pipeline.market.run_market_update_v2
import pipeline.betting.run_betting_outcomes_v2
import pipeline.features.run_build_fighter_state
import pipeline.features.run_build_feature_view
print('v2 module imports ok')
PY
```

If any import fails, restore the moved file with:

```bash
git mv archive/<path>/<file> <original-path>/<file>
```

---

# Important Caveat

This plan is based on repository search and V2 architecture tracing. It is intentionally conservative. Some files marked `manual review` may eventually be archived, but should not be moved until import tracing or grep confirms they are unused.
