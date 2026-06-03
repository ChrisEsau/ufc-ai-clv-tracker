# UFC Artifact Registry

## Canonical Data Layout

Current canonical artifact directories:

```text
data/master
data/staging
data/status
data/audits
data/backups
data/features
data/predictions
models
```

Generated binary artifacts such as parquet files are ignored by `.gitignore` by default. GitHub Actions workflows must use `git add -f` for generated parquet artifacts that should be committed back to the repository.

---

## Master Dataset

```text
data/master/ufc_master.parquet
```

Authoritative UFC historical fight dataset.

---

## Staging Artifacts

```text
data/staging/ufc_missing_events.parquet
data/staging/ufc_staged_fight_rows.parquet
data/staging/ufc_staged_fight_details.parquet
data/staging/ufc_staged_fighter_profiles.parquet
data/staging/ufc_staged_master_rows.parquet
data/staging/ufc_staged_master_rows_enriched.parquet
data/staging/ufc_staged_master_rows_profiled.parquet
```

Purpose:

* `ufc_missing_events.parquet` stores completed UFCStats events not represented in the local master dataset.
* `ufc_staged_fight_rows.parquet` stores fight-level rows from selected event pages.
* `ufc_staged_fight_details.parquet` stores detailed UFCStats fight-page output.
* `ufc_staged_master_rows.parquet` stores staged rows mapped into the canonical master schema.
* `ufc_staged_master_rows_enriched.parquet` stores rows after derived-stat calculation.
* `ufc_staged_fighter_profiles.parquet` stores scraped fighter profile rows.
* `ufc_staged_master_rows_profiled.parquet` stores final staged rows with fighter profile enrichment.

---

## Feature Artifacts

```text
data/features/UFC_enhanced_rolling_features_EWM.parquet
data/features/ufc_current_fighter_features.parquet
```

Purpose:

* `UFC_enhanced_rolling_features_EWM.parquet` stores historical point-in-time training/backtesting features.
* `ufc_current_fighter_features.parquet` stores latest one-row-per-fighter state for live prediction.

---

## Prediction Artifacts

```text
data/predictions/ufc_live_card.parquet
data/predictions/ufc_model_predictions.parquet
data/predictions/ufc_live_action_board.parquet
data/predictions/ufc_live_watchlist.parquet
data/predictions/ufc_betting_board.parquet
data/predictions/ufc_official_bets.parquet
```

---

## Model Artifacts

```text
models/UFC_Model_v5_Experiment/production_config.json
models/UFC_Model_v5_Experiment/production_config.pkl
models/UFC_Model_v5_Experiment/feature_columns.pkl
models/UFC_Model_v5_Experiment/best_threshold.pkl
models/UFC_Model_v5_Experiment/calibrated_model.pkl
models/UFC_Model_v5_Experiment/raw_model.pkl
models/UFC_Model_v5_Experiment/model_quality_summary.csv
models/UFC_Model_v5_Experiment/shap_importance.csv
```

Purpose:

* `ufc_missing_events.parquet` stores completed UFCStats events not represented in the local master dataset.
* `ufc_staged_fight_rows.parquet` stores fight-level rows from selected event pages.
* `ufc_staged_fight_details.parquet` stores detailed UFCStats fight-page output.
* `ufc_staged_master_rows.parquet` stores staged rows mapped into the canonical master schema.
* `ufc_staged_master_rows_enriched.parquet` stores rows after derived-stat calculation.
* `ufc_staged_fighter_profiles.parquet` stores scraped fighter profile rows.
* `ufc_staged_master_rows_profiled.parquet` stores final staged rows with fighter profile enrichment.

---

## Status Artifacts

```text
data/status/ufc_dataset_status.parquet
data/status/ufc_dataset_event_status.parquet
data/status/ufc_ufcstats_event_check.parquet
```

---

## Audit Artifacts

```text
data/audits/ufc_fight_scrape_audit.parquet
data/audits/ufc_fight_detail_scrape_audit.parquet
data/audits/ufc_fighter_profile_scrape_audit.parquet
data/audits/ufc_staged_master_mapping_audit.parquet
data/audits/ufc_staged_derived_stats_audit.parquet
data/audits/ufc_master_column_validation.parquet
data/audits/ufc_append_precheck.parquet
data/audits/ufc_append_duplicate_check.parquet
data/audits/ufc_append_required_field_audit.parquet
data/audits/ufc_staged_final_review.parquet
data/audits/ufc_append_audit.parquet
data/audits/ufc_live_feature_audit.parquet
data/audits/ufc_live_match_audit.parquet
data/audits/ufc_live_odds_audit.parquet
```

### Final Review Artifact

```text
data/audits/ufc_staged_final_review.parquet
```

Purpose:

* Stores semantic staged-row review checks.
* Includes blocking and warning checks.
* Includes `final_review_pass`.
* Required before append.

---

## Backup Artifacts

```text
data/backups/ufc_master_backup_<run_id>.parquet
```

Created before appending staged rows to master.
