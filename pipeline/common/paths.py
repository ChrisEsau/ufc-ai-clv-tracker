from pathlib import Path

ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CONFIGS_DIR = ROOT_DIR / "configs"

MASTER_DIR = DATA_DIR / "master"
STAGING_DIR = DATA_DIR / "staging"
AUDITS_DIR = DATA_DIR / "audits"
STATUS_DIR = DATA_DIR / "status"
BACKUPS_DIR = DATA_DIR / "backups"
FEATURES_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"
CARDS_DIR = DATA_DIR / "cards"
FIGHT_DETAILS_DIR = DATA_DIR / "fight_details"
MARKET_DIR = DATA_DIR / "market"
MARKET_RAW_DIR = MARKET_DIR / "raw"
DRAFTKINGS_RAW_DIR = MARKET_RAW_DIR / "draftkings"
BANKROLL_DIR = DATA_DIR / "bankroll"
MODEL_LAB_DIR = DATA_DIR / "model_lab"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
CLV_DIR = DATA_DIR / "clv"
DOCS_DIR = ROOT_DIR / "docs"
MARKET_CONFIG_DIR = CONFIGS_DIR / "market"

MASTER_PATH = MASTER_DIR / "ufc_master.parquet"

STAGED_FIGHT_ROWS_PATH = STAGING_DIR / "ufc_staged_fight_rows.parquet"
STAGED_FIGHT_DETAILS_PATH = STAGING_DIR / "ufc_staged_fight_details.parquet"
STAGED_FIGHTER_PROFILES_PATH = STAGING_DIR / "ufc_staged_fighter_profiles.parquet"
STAGED_MASTER_ROWS_PATH = STAGING_DIR / "ufc_staged_master_rows.parquet"
STAGED_MASTER_ROWS_ENRICHED_PATH = STAGING_DIR / "ufc_staged_master_rows_enriched.parquet"
STAGED_MASTER_ROWS_PROFILED_PATH = STAGING_DIR / "ufc_staged_master_rows_profiled.parquet"
MISSING_EVENTS_PATH = STAGING_DIR / "ufc_missing_events.parquet"

FIGHT_SCRAPE_AUDIT_PATH = AUDITS_DIR / "ufc_fight_scrape_audit.parquet"
FIGHT_DETAIL_SCRAPE_AUDIT_PATH = AUDITS_DIR / "ufc_fight_detail_scrape_audit.parquet"
FIGHTER_PROFILE_SCRAPE_AUDIT_PATH = AUDITS_DIR / "ufc_fighter_profile_scrape_audit.parquet"
MASTER_COLUMN_VALIDATION_PATH = AUDITS_DIR / "ufc_master_column_validation.parquet"
APPEND_PRECHECK_PATH = AUDITS_DIR / "ufc_append_precheck.parquet"
APPEND_DUPLICATE_CHECK_PATH = AUDITS_DIR / "ufc_append_duplicate_check.parquet"
APPEND_REQUIRED_FIELD_AUDIT_PATH = AUDITS_DIR / "ufc_append_required_field_audit.parquet"
APPEND_AUDIT_PATH = AUDITS_DIR / "ufc_append_audit.parquet"
STAGED_MASTER_MAPPING_AUDIT_PATH = AUDITS_DIR / "ufc_staged_master_mapping_audit.parquet"
STAGED_DERIVED_STATS_AUDIT_PATH = AUDITS_DIR / "ufc_staged_derived_stats_audit.parquet"
STAGED_FINAL_REVIEW_PATH = AUDITS_DIR / "ufc_staged_final_review.parquet"

MODEL_VERSION = "UFC_Model_v5_Experiment"
MODEL_DIR = MODELS_DIR / MODEL_VERSION
MODEL_PRODUCTION_CONFIG_JSON_PATH = MODEL_DIR / "production_config.json"
MODEL_PRODUCTION_CONFIG_PKL_PATH = MODEL_DIR / "production_config.pkl"
MODEL_FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
MODEL_BEST_THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"
MODEL_CALIBRATED_PATH = MODEL_DIR / "calibrated_model.pkl"
MODEL_RAW_PATH = MODEL_DIR / "raw_model.pkl"
MODEL_QUALITY_SUMMARY_PATH = MODEL_DIR / "model_quality_summary.csv"
MODEL_SHAP_IMPORTANCE_PATH = MODEL_DIR / "shap_importance.csv"

ROLLING_FEATURES_PATH = FEATURES_DIR / "UFC_enhanced_rolling_features_EWM.parquet"
CURRENT_FIGHTER_FEATURES_PATH = FEATURES_DIR / "latest_fighter_state.parquet"
FIGHTER_STATE_HISTORY_PATH = FEATURES_DIR / "fighter_state_history.parquet"
LATEST_FIGHTER_STATE_PATH = FEATURES_DIR / "latest_fighter_state.parquet"
MONEYLINE_FEATURE_VIEW_PATH = FEATURES_DIR / "moneyline_feature_view.parquet"
MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_PATH = FEATURES_DIR / "moneyline_favdog_market_feature_view.parquet"
MONEYLINE_FAVDOG_MARKET_FEATURE_VIEW_AUDIT_PATH = AUDITS_DIR / "moneyline_favdog_market_feature_view_validation.parquet"

ROUND_STATS_PATH = FIGHT_DETAILS_DIR / "ufc_round_stats.parquet"
ROUND_FIGHTER_STATE_HISTORY_PATH = FEATURES_DIR / "round_fighter_state_history.parquet"
ROUND_LATEST_FIGHTER_STATE_PATH = FEATURES_DIR / "round_latest_fighter_state.parquet"
ROUND_FIGHTER_STATE_P0_1_VALIDATION_PATH = AUDITS_DIR / "round_fighter_state_p0_1_validation.parquet"
ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH = FEATURES_DIR / "round_fighter_suppression_p0_2_history.parquet"
ROUND_LATEST_FIGHTER_SUPPRESSION_P0_2_PATH = FEATURES_DIR / "round_latest_fighter_suppression_p0_2.parquet"
ROUND_FIGHTER_SUPPRESSION_P0_2_VALIDATION_PATH = AUDITS_DIR / "round_fighter_suppression_p0_2_validation.parquet"
ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH = FEATURES_DIR / "round_fighter_wrestling_p0_3_history.parquet"
ROUND_LATEST_FIGHTER_WRESTLING_P0_3_PATH = FEATURES_DIR / "round_latest_fighter_wrestling_p0_3.parquet"
ROUND_FIGHTER_WRESTLING_P0_3_VALIDATION_PATH = AUDITS_DIR / "round_fighter_wrestling_p0_3_validation.parquet"
ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH = FEATURES_DIR / "round_fighter_defense_p1_4_history.parquet"
ROUND_LATEST_FIGHTER_DEFENSE_P1_4_PATH = FEATURES_DIR / "round_latest_fighter_defense_p1_4.parquet"
ROUND_FIGHTER_DEFENSE_P1_4_VALIDATION_PATH = AUDITS_DIR / "round_fighter_defense_p1_4_validation.parquet"
RFS_MC_V1_SUBMISSION_HISTORY_PATH = FEATURES_DIR / "rfs_mc_v1_submission_history.parquet"

UPCOMING_EVENTS_PATH = CARDS_DIR / "ufcstats_upcoming_events.parquet"
UPCOMING_FIGHTS_PATH = CARDS_DIR / "ufcstats_upcoming_fights.parquet"
SELECTED_LIVE_CARD_EVENT_PATH = CARDS_DIR / "ufc_selected_live_card_event.parquet"

LIVE_CARD_PATH = PREDICTIONS_DIR / "ufc_live_card.parquet"
MODEL_PREDICTIONS_PATH = PREDICTIONS_DIR / "ufc_model_predictions.parquet"
LIVE_ACTION_BOARD_PATH = PREDICTIONS_DIR / "ufc_live_action_board.parquet"
LIVE_WATCHLIST_PATH = PREDICTIONS_DIR / "ufc_live_watchlist.parquet"
BETTING_BOARD_PATH = PREDICTIONS_DIR / "ufc_betting_board.parquet"
BETTING_OUTCOMES_PATH = PREDICTIONS_DIR / "betting_outcomes.parquet"
BETTING_OUTCOMES_AUDIT_PATH = AUDITS_DIR / "ufc_betting_outcomes_audit.parquet"
BETTING_JOIN_KEY_DIAGNOSTIC_PATH = AUDITS_DIR / "ufc_betting_join_key_diagnostic.parquet"
OFFICIAL_BETS_PATH = PREDICTIONS_DIR / "ufc_official_bets.parquet"

MARKET_ODDS_PATH = MARKET_DIR / "ufc_market_odds.parquet"
MARKET_SNAPSHOTS_PATH = MARKET_DIR / "ufc_market_snapshots.parquet"
NORMALIZED_MARKET_SNAPSHOTS_PATH = MARKET_DIR / "ufc_normalized_market_snapshots.parquet"
MARKET_MATCH_AUDIT_PATH = MARKET_DIR / "ufc_market_match_audit.parquet"
CLOSING_LINES_PATH = MARKET_DIR / "ufc_closing_lines.parquet"
LINE_MOVEMENT_PATH = MARKET_DIR / "ufc_line_movement.parquet"
CLV_RESULTS_PATH = MARKET_DIR / "ufc_clv_results.parquet"
MODEL_CANDIDATE_TRACKER_PATH = CLV_DIR / "ufc_model_candidate_tracker.parquet"
MODEL_CANDIDATE_CLV_PATH = CLV_DIR / "ufc_model_candidate_clv.parquet"

MARKET_REGISTRY_PATH = MARKET_CONFIG_DIR / "market_registry.yaml"
MARKET_OUTCOMES_PATH = MARKET_DIR / "market_outcomes.parquet"
HISTORICAL_MARKET_OUTCOMES_PATH = MARKET_DIR / "historical_market_outcomes.parquet"
MARKET_OUTCOME_SNAPSHOTS_PATH = MARKET_DIR / "market_outcome_snapshots.parquet"
MARKET_OUTCOME_AUDIT_PATH = AUDITS_DIR / "ufc_market_outcome_audit.parquet"
MARKET_MATCH_AUDIT_V2_PATH = AUDITS_DIR / "ufc_market_match_audit_v2.parquet"
DRAFTKINGS_MARKET_DIAGNOSTIC_PATH = MARKET_DIR / "draftkings_market_diagnostic.parquet"
DRAFTKINGS_RAW_INDEX_PATH = MARKET_DIR / "draftkings_raw_index.parquet"
DRAFTKINGS_EVENT_INDEX_PATH = MARKET_DIR / "draftkings_event_index.parquet"
DRAFTKINGS_EVENT_CARD_MATCH_PATH = MARKET_DIR / "draftkings_event_card_matches.parquet"

FANDUEL_RAW_DIR = MARKET_RAW_DIR / "fanduel"
FANDUEL_MARKET_DIAGNOSTIC_PATH = MARKET_DIR / "fanduel_market_diagnostic.parquet"
FANDUEL_EVENT_INDEX_PATH = MARKET_DIR / "fanduel_event_index.parquet"
FANDUEL_EVENT_CARD_MATCH_PATH = MARKET_DIR / "fanduel_event_card_matches.parquet"
DRAFTKINGS_MARKET_CATALOG_PATH = MARKET_DIR / "draftkings_market_catalog.parquet"
FANDUEL_MARKET_CATALOG_PATH = MARKET_DIR / "fanduel_market_catalog.parquet"
CANONICAL_MARKET_CATALOG_PATH = MARKET_DIR / "canonical_market_catalog.parquet"
CANONICAL_MARKET_AUDIT_PATH = AUDITS_DIR / "canonical_market_catalog_audit.parquet"
MARKET_SIGNALS_PATH = MARKET_DIR / "market_signals.parquet"
MARKET_SIGNALS_AUDIT_PATH = AUDITS_DIR / "market_signals_audit.parquet"
MARKET_INTELLIGENCE_HISTORY_PATH = MARKET_DIR / "market_intelligence_history.parquet"
MARKET_INTELLIGENCE_HISTORY_AUDIT_PATH = AUDITS_DIR / "market_intelligence_history_audit.parquet"

MODEL_MARKET_SNAPSHOTS_PATH = SNAPSHOTS_DIR / "model_market_snapshots.parquet"
MODEL_MARKET_SNAPSHOT_AUDIT_PATH = AUDITS_DIR / "model_market_snapshot_audit.parquet"
CLOSING_LINE_SNAPSHOTS_PATH = SNAPSHOTS_DIR / "closing_line_snapshots.parquet"
CLOSING_LINE_SNAPSHOT_AUDIT_PATH = AUDITS_DIR / "closing_line_snapshot_audit.parquet"
FIGHT_DAY_MONITOR_STATUS_PATH = STATUS_DIR / "fight_day_monitor_status.json"

LIVE_FEATURE_AUDIT_PATH = AUDITS_DIR / "ufc_live_feature_audit.parquet"
LIVE_MATCH_AUDIT_PATH = AUDITS_DIR / "ufc_live_match_audit.parquet"
LIVE_ODDS_AUDIT_PATH = AUDITS_DIR / "ufc_live_odds_audit.parquet"
CLV_MARKET_NORMALIZATION_AUDIT_PATH = AUDITS_DIR / "ufc_clv_market_normalization_audit.parquet"

BET_LEDGER_PATH = BANKROLL_DIR / "ufc_bet_ledger.parquet"
OPEN_BETS_PATH = BANKROLL_DIR / "ufc_open_bets.parquet"
BANKROLL_SNAPSHOTS_PATH = BANKROLL_DIR / "ufc_bankroll_snapshots.parquet"
BANKROLL_SETTINGS_PATH = BANKROLL_DIR / "ufc_bankroll_settings.parquet"

DATASET_STATUS_PATH = STATUS_DIR / "ufc_dataset_status.parquet"
DATASET_EVENT_STATUS_PATH = STATUS_DIR / "ufc_dataset_event_status.parquet"
UFCSTATS_EVENT_CHECK_PATH = STATUS_DIR / "ufc_ufcstats_event_check.parquet"

INGESTION_PIPELINE_REGISTRY_DOC = DOCS_DIR / "UFC_INGESTION_PIPELINE_REGISTRY.md"
MASTER_SCHEMA_DOC = DOCS_DIR / "UFC_MASTER_SCHEMA.md"
PATH_REGISTRY_DOC = DOCS_DIR / "UFC_PATH_REGISTRY.md"
DM_DASHBOARD_ARCHITECTURE_DOC = DOCS_DIR / "UFC_DM_DASHBOARD_ARCHITECTURE.md"
PREDICTION_PIPELINE_DOC = DOCS_DIR / "UFC_PREDICTION_PIPELINE.md"
CLV_TRACKING_DOC = DOCS_DIR / "UFC_CLV_TRACKING.md"


def master_backup_path(run_id: str) -> Path:
    return BACKUPS_DIR / f"ufc_master_backup_{run_id}.parquet"


def ensure_data_dirs() -> None:
    for path in [
        MASTER_DIR,
        STAGING_DIR,
        AUDITS_DIR,
        STATUS_DIR,
        BACKUPS_DIR,
        FEATURES_DIR,
        PREDICTIONS_DIR,
        CARDS_DIR,
        FIGHT_DETAILS_DIR,
        MARKET_DIR,
        MARKET_RAW_DIR,
        DRAFTKINGS_RAW_DIR,
        FANDUEL_RAW_DIR,
        BANKROLL_DIR,
        MODEL_LAB_DIR,
        SNAPSHOTS_DIR,
        CLV_DIR,
        MODELS_DIR,
        MODEL_DIR,
        DOCS_DIR,
        MARKET_CONFIG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
