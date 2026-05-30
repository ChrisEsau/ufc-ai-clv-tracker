from pathlib import Path


# ============================================================
# ROOT
# ============================================================

ROOT_DIR = Path(".")


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = ROOT_DIR / "data"

MASTER_DIR = DATA_DIR / "master"
STAGING_DIR = DATA_DIR / "staging"
AUDITS_DIR = DATA_DIR / "audits"
STATUS_DIR = DATA_DIR / "status"
BACKUPS_DIR = DATA_DIR / "backups"


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
STAGED_DERIVED_STATS_AUDIT_PATH =  AUDITS_DIR / "ufc_staged_derived_stats_audit.parquet"

# ============================================================
# STATUS
# ============================================================

DATASET_STATUS_PATH = STATUS_DIR / "ufc_dataset_status.parquet"
DATASET_EVENT_STATUS_PATH = STATUS_DIR / "ufc_dataset_event_status.parquet"
UFCSTATS_EVENT_CHECK_PATH = STATUS_DIR / "ufc_ufcstats_event_check.parquet"


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
    ]:
        path.mkdir(parents=True, exist_ok=True)