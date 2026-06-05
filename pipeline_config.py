# ============================================================
# pipeline_config.py
# Centralized UFC pipeline configuration
# ============================================================

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

from pipeline.common.paths import MODEL_VERSION
from pipeline.common.risk_settings import load_risk_settings

BASE_PATH = "."


# ------------------------------------------------------------
# API KEYS
# ------------------------------------------------------------

import os

ODDS_API_KEY = os.getenv("ODDS_API_KEY")


# ------------------------------------------------------------
# ODDS API SETTINGS
# ------------------------------------------------------------

SPORT = "mma_mixed_martial_arts"
REGIONS = "us"
MARKETS = "h2h"
ODDS_FORMAT = "american"
PREFERRED_BOOKMAKER = "DraftKings"


# ------------------------------------------------------------
# PRODUCTION FILTERS
# ------------------------------------------------------------

_RISK_SETTINGS = load_risk_settings()

MIN_EDGE = _RISK_SETTINGS.min_edge
MIN_CONFIDENCE = _RISK_SETTINGS.min_confidence

MIN_ODDS = _RISK_SETTINGS.min_odds
MAX_ODDS = _RISK_SETTINGS.max_odds


# ------------------------------------------------------------
# BANKROLL SETTINGS
# ------------------------------------------------------------

STARTING_BANKROLL = _RISK_SETTINGS.starting_bankroll

KELLY_MULTIPLIER = _RISK_SETTINGS.kelly_fraction
MAX_STAKE_PCT = _RISK_SETTINGS.max_stake_pct


# ------------------------------------------------------------
# PROBABILITY CLIPPING
# ------------------------------------------------------------

PROB_CLIP_LOW = 0.03
PROB_CLIP_HIGH = 0.97


# ------------------------------------------------------------
# ODDS MATCHING
# ------------------------------------------------------------

MIN_ODDS_MATCH_SCORE = 80


# ------------------------------------------------------------
# WATCHLIST THRESHOLDS
# ------------------------------------------------------------

WATCHLIST_EV_THRESHOLD = 40
WATCHLIST_CONFIDENCE_THRESHOLD = 0.65
