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
CURRENT_FIGHTER_FEATURES_PATH = FEATURES_DIR / "ufc_current_fighter_features.parquet"

UPCOMING_EVENTS_PATH = CARDS_DIR / "ufcstats_upcoming_events.parquet"
UPCOMING_FIGHTS_PATH = CARDS_DIR / "ufcstats_upcoming_fights.parquet"
SELECTED_LIVE_CARD_EVENT_PATH = CARDS_DIR / "ufc_selected_live_card_event.parquet"

LIVE_CARD_PATH = PREDICTIONS_DIR / "ufc_live_card.parquet"
MODEL_PREDICTIONS_PATH = PREDICTIONS_DIR / "ufc_model_predictions.parquet"
LIVE_ACTION_BOARD_PATH = PREDICTIONS_DIR / "ufc_live_action_board.parquet"
LIVE_WATCHLIST_PATH = PREDICTIONS_DIR / "ufc_live_watchlist.parquet"
BETTING_BOARD_PATH = PREDICTIONS_DIR / "ufc_betting_board.parquet"
OFFICIAL_BETS_PATH = PREDICTIONS_DIR / "ufc_official_bets.parquet"

MARKET_ODDS_PATH = MARKET_DIR / "ufc_market_odds.parquet"
MARKET_SNAPSHOTS_PATH = MARKET_DIR / "ufc_market_snapshots.parquet"
MARKET_MATCH_AUDIT_PATH = MARKET_DIR / "ufc_market_match_audit.parquet"
CLOSING_LINES_PATH = MARKET_DIR / "ufc_closing_lines.parquet"
LINE_MOVEMENT_PATH = MARKET_DIR / "ufc_line_movement.parquet"
CLV_RESULTS_PATH = MARKET_DIR / "ufc_clv_results.parquet"

LIVE_FEATURE_AUDIT_PATH = AUDITS_DIR / "ufc_live_feature_audit.parquet"
LIVE_MATCH_AUDIT_PATH = AUDITS_DIR / "ufc_live_match_audit.parquet"
LIVE_ODDS_AUDIT_PATH = AUDITS_DIR / "ufc_live_odds_audit.parquet"

# ============================================================
# BANKROLL ARTIFACTS
# ============================================================

BET_LEDGER_PATH = BANKROLL_DIR / "ufc_bet_ledger.parquet"
OPEN_BETS_PATH = BANKROLL_DIR / "ufc_open_bets.parquet"
BANKROLL_SNAPSHOTS_PATH = BANKROLL_DIR / "ufc_bankroll_snapshots.parquet"
BANKROLL_SETTINGS_PATH = BANKROLL_DIR / "ufc_bankroll_settings.parquet"

# ============================================================
# STATUS
# ============================================================

DATASET_STATUS_PATH = STATUS_DIR / "ufc_dataset_status.parquet"
DATASET_EVENT_STATUS_PATH = STATUS_DIR / "ufc_dataset_event_status.parquet"
UFCSTATS_EVENT_CHECK_PATH = STATUS_DIR / "ufc_ufcstats_event_check.parquet"

# ============================================================
# DOCS
# ============================================================

INGESTION_PIPELINE_REGISTRY_DOC = DOCS_DIR / "UFC_INGESTION_PIPELINE_REGISTRY.md"
MASTER_SCHEMA_DOC = DOCS_DIR / "UFC_MASTER_SCHEMA.md"
PATH_REGISTRY_DOC = DOCS_DIR / "UFC_PATH_REGISTRY.md"
DM_DASHBOARD_ARCHITECTURE_DOC = DOCS_DIR / "UFC_DM_DASHBOARD_ARCHITECTURE.md"
PREDICTION_PIPELINE_DOC = DOCS_DIR / "UFC_PREDICTION_PIPELINE.md"
CLV_TRACKING_DOC = DOCS_DIR / "UFC_CLV_TRACKING.md"

# ============================================================
# BACKUPS
# ============================================================

def master_backup_path(run_id: str) -> Path:
    return BACKUPS_DIR / f"ufc_master_backup_{run_id}.parquet"


# ============================================================
# HELPERS
# ============================================================

def ensure_data_dirs():
    for path in [
        MASTER_DIR,
        STAGING_DIR,
        AUDITS_DIR,
        STATUS_DIR,
        BACKUPS_DIR,
        FEATURES_DIR,
        PREDICTIONS_DIR,
        CARDS_DIR,
        MARKET_DIR,
        BANKROLL_DIR,
        MODEL_LAB_DIR,
        MODELS_DIR,
        MODEL_DIR,
        DOCS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
