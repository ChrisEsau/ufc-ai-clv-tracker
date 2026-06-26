"""Lightweight repository smoke tests.

These checks intentionally avoid scraping, model training, odds API calls,
or mutating any parquet artifacts.  The goal is to catch broken imports,
missing dependencies, and path-registry issues early.

Run locally:
    python tests/smoke_test_imports.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODULES_TO_IMPORT = [
    # Dashboard shell
    "dashboard",
    "utils.sidebar",
    "utils.theme",
    "utils.data_loader",
    # Shared pipeline utilities
    "pipeline.common.paths",
    "pipeline.common.booleans",
    "pipeline.common.risk_settings",
    # Betting board / bankroll / CLV support
    "utils.betting_board_rules",
    "utils.betting_board_artifacts",
    "utils.bankroll_artifacts",
    "utils.clv_artifacts",
    "pipeline.clv.run_clv_pipeline",
    "pipeline.bankroll.run_bankroll_status",
    # Data maintenance runners
    "pipeline.data_maintenance.run_dataset_status",
    "pipeline.data_maintenance.run_master_column_validation",
    "pipeline.data_maintenance.run_append_precheck_validation",
    "pipeline.data_maintenance.run_staged_final_review",
    # Prediction runners
    "pipeline.prediction.run_refresh_upcoming_events",
    "pipeline.prediction.run_build_live_card",
    # Round Fighter State runners
    "pipeline.round_stats.round_state_formulas",
    "pipeline.round_stats.build_round_fighter_state",
    "pipeline.round_stats.validate_round_fighter_state",
    "pipeline.round_stats.join_round_fighter_state",
    "pipeline.features.views.moneyline_round_fighter_state",
    "pipeline.features.views.moneyline",
]


REQUIRED_PATH_CONSTANTS = [
    "MASTER_PATH",
    "STAGED_FIGHT_ROWS_PATH",
    "STAGED_FIGHT_DETAILS_PATH",
    "STAGED_MASTER_ROWS_PATH",
    "APPEND_PRECHECK_PATH",
    "DATASET_STATUS_PATH",
    "ROLLING_FEATURES_PATH",
    "CURRENT_FIGHTER_FEATURES_PATH",
    "LIVE_CARD_PATH",
    "MODEL_PREDICTIONS_PATH",
    "BETTING_BOARD_PATH",
    "MARKET_SNAPSHOTS_PATH",
    "CLV_RESULTS_PATH",
    "BET_LEDGER_PATH",
    "BANKROLL_SETTINGS_PATH",
    "FIGHT_DETAILS_DIR",
    "ROUND_STATS_PATH",
    "ROUND_FIGHTER_STATE_HISTORY_PATH",
    "ROUND_LATEST_FIGHTER_STATE_PATH",
    "ROUND_FIGHTER_STATE_P0_1_VALIDATION_PATH",
]


def import_module(module_name: str) -> ModuleType:
    """Import a module and raise a clear error if it fails."""

    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - script-style smoke test
        raise RuntimeError(f"Failed to import {module_name}: {exc}") from exc


def check_imports() -> None:
    """Verify key dashboard and pipeline modules import cleanly."""

    print("========== UFC REPO IMPORT SMOKE TEST ==========")
    for module_name in MODULES_TO_IMPORT:
        import_module(module_name)
        print(f"OK import: {module_name}")


def check_path_registry() -> None:
    """Verify important canonical path constants exist and are path-like."""

    paths = import_module("pipeline.common.paths")
    missing = [name for name in REQUIRED_PATH_CONSTANTS if not hasattr(paths, name)]
    if missing:
        raise AssertionError(f"Missing path constants: {missing}")

    for name in REQUIRED_PATH_CONSTANTS:
        value = getattr(paths, name)
        if not isinstance(value, Path):
            raise AssertionError(f"{name} is not a pathlib.Path: {type(value)!r}")
        print(f"OK path constant: {name} -> {value}")


def check_risk_settings() -> None:
    """Verify risk settings load and contain sane basic values."""

    risk_settings = import_module("pipeline.common.risk_settings")
    settings = risk_settings.load_risk_settings()

    if settings.starting_bankroll <= 0:
        raise AssertionError("starting_bankroll must be positive")
    if settings.kelly_fraction <= 0:
        raise AssertionError("kelly_fraction must be positive")
    if settings.max_stake_pct <= 0:
        raise AssertionError("max_stake_pct must be positive")
    if settings.min_odds >= settings.max_odds:
        raise AssertionError("min_odds must be less than max_odds")

    print("OK risk settings loaded")


def main() -> None:
    check_imports()
    check_path_registry()
    check_risk_settings()
    print("========== SMOKE TEST PASSED ==========")


if __name__ == "__main__":
    main()


# P0.2 Round Fighter Suppression smoke imports
import importlib
importlib.import_module("pipeline.round_stats.build_round_fighter_suppression")
importlib.import_module("pipeline.round_stats.validate_round_fighter_suppression")


# P0.3 Round Fighter Wrestling smoke imports
import importlib
importlib.import_module("pipeline.round_stats.build_round_fighter_wrestling")
importlib.import_module("pipeline.round_stats.validate_round_fighter_wrestling")
