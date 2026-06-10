from pathlib import Path


# ============================================================
# ROOT
# ============================================================

ROOT_DIR = Path(".")


# ============================================================
# DATA DIRECTORIES
# ============================================================

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
MARKET_DIR = DATA_DIR / "market"
BANKROLL_DIR = DATA_DIR / "bankroll"
MODEL_LAB_DIR = DATA_DIR / "model_lab"
DOCS_DIR = ROOT_DIR / "docs"
MARKET_CONFIG_DIR = CONFIGS_DIR / "market"


# ============================================================
# MASTER
# ============================================================

MASTER_PATH = MASTER_DIR / "ufc_master.parquet"


# ============================================================
# STAGING ARTIFACTS
# ============================================================

STAGED_FIGHT_ROWS_PATH = STAGING_DIR / "ufc_staged_fight_rows.parquet"
STAGED_FIGHT_DETAILS_PATH = STAGING_DIR / "ufc_staged_fight_details.parquet"
STAGED_FIGHTER_PROFILES_PATH = STAGING_DIR / "ufc_staged_fighter_profiles.parquet"

STAGED_MASTER_ROWS_PATH = STAGING_DIR / "ufc_staged_master_rows.parquet"
STAGED_MASTER_ROWS_ENRICHED_PATH = STAGING_DIR / "ufc_staged_master_rows_enriched.parquet"
STAGED_MASTER_ROWS_PROFILED_PATH = STAGING_DIR / "ufc_staged_master_rows_profiled.parquet"

MISSING_EVENTS_PATH = STAGING_DIR / "ufc_missing_events.parquet"


# ============================================================
# AUDITS
# ============================================================

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


# ============================================================
# MODEL / FEATURE / PREDICTION ARTIFACTS
# ============================================================

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
# Compatibility alias used by live_feature_builder; now points to fighter-state artifact.
CURRENT_FIGHTER_FEATURES_PATH = FEATURES_DIR / "latest_fighter_state.parquet"
FIGHTER_STATE_HISTORY_PATH = FEATURES_DIR / "fighter_state_history.parquet"
LATEST_FIGHTER_STATE_PATH = FEATURES_DIR / "latest_fighter_state.parquet"
MONEYLINE_FEATURE_VIEW_PATH = FEATURES_DIR / "moneyline_feature_view.parquet"

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
OFFICIAL_BETS_PATH = PREDICTIONS_DIR / "ufc_official_bets.parquet"

# Legacy side-based market artifacts.
MARKET_ODDS_PATH = MARKET_DIR / "ufc_market_odds.parquet"
MARKET_SNAPSHOTS_PATH = MARKET_DIR / "ufc_market_snapshots.parquet"
NORMALIZED_MARKET_SNAPSHOTS_PATH = MARKET_DIR / "ufc_normalized_market_snapshots.parquet"
MARKET_MATCH_AUDIT_PATH = MARKET_DIR / "ufc_market_match_audit.parquet"
CLOSING_LINES_PATH = MARKET_DIR / "ufc_closing_lines.parquet"
LINE_MOVEMENT_PATH = MARKET_DIR / "ufc_line_movement.parquet"
CLV_RESULTS_PATH = MARKET_DIR / "ufc_clv_results.parquet"

# Market Pipeline V2 outcome-based artifacts.
MARKET_REGISTRY_PATH = MARKET_CONFIG_DIR / "market_registry.yaml"
MARKET_OUTCOMES_PATH = MARKET_DIR / "market_outcomes.parquet"
