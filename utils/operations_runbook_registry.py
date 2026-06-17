from __future__ import annotations

from copy import deepcopy
from typing import Any


WorkflowSpec = dict[str, Any]
RunbookStep = dict[str, Any]
RunbookSpec = dict[str, Any]


MARKET_REFRESH_V2: RunbookSpec = {
    "runbook_id": "market_refresh_v2",
    "display_name": "Market Refresh",
    "description": "Refresh upcoming UFC events, predictions, DraftKings markets, betting outcomes, and model-market snapshots.",
    "steps": [
        {
            "step_id": "refresh_upcoming_events",
            "display_name": "Refresh Upcoming Events",
            "description": "Update upcoming UFCStats event and fight artifacts.",
            "workflows": [
                {
                    "workflow_file": "run-refresh-upcoming-events.yml",
                    "display_name": "Refresh UFCStats Upcoming Events",
                    "inputs": {},
                    "outputs": [
                        "data/cards/ufcstats_upcoming_events.parquet",
                        "data/cards/ufcstats_upcoming_fights.parquet",
                    ],
                },
            ],
        },
        {
            "step_id": "run_predictions",
            "display_name": "Run Predictions",
            "description": "Refresh fighter state, rebuild the production feature view, and generate model outcome predictions.",
            "workflows": [
                {
                    "workflow_file": "run-build-fighter-state-v2.yml",
                    "display_name": "Build Fighter State V2",
                    "inputs": {},
                    "outputs": [
                        "data/features/fighter_state_history.parquet",
                        "data/features/latest_fighter_state.parquet",
                    ],
                },
                {
                    "workflow_file": "run-build-feature-view-v2.yml",
                    "display_name": "Build Feature View V2",
                    "inputs": {
                        "config_path": "configs/feature_views/moneyline_base.yaml",
                        "output_path": "data/features/moneyline_feature_view.parquet",
                    },
                    "outputs": [
                        "data/features/moneyline_feature_view.parquet",
                    ],
                },
                {
                    "workflow_file": "run-prediction-v2.yml",
                    "display_name": "Run Prediction V2",
                    "inputs": {
                        "model_family": "moneyline",
                        "model_id": "moneyline_xgboost_v5",
                    },
                    "outputs": [
                        "data/predictions/ufc_live_card.parquet",
                        "data/predictions/live_model_features.parquet",
                        "data/predictions/model_outcomes.parquet",
                        "data/predictions/by_model/{model_id}/model_outcomes.parquet",
                        "data/audits/ufc_live_feature_audit.parquet",
                    ],
                },
            ],
        },
        {
            "step_id": "get_market_odds",
            "display_name": "Get Market Odds",
            "description": "Discover DraftKings events, filter to live-card matches, collect markets, then normalize and match market outcomes.",
            "workflows": [
                {
                    "workflow_file": "run-draftkings-event-index.yml",
                    "display_name": "Build DraftKings Event Index",
                    "inputs": {},
                    "outputs": [
                        "data/market/draftkings_event_index.parquet",
                    ],
                },
                {
                    "workflow_file": "run-draftkings-card-filter.yml",
                    "display_name": "Match DraftKings Events To Card",
                    "inputs": {
                        "min_match_score": "80",
                        "min_single_score": "70",
                    },
                    "outputs": [
                        "data/market/draftkings_event_card_matches.parquet",
                    ],
                },
                {
                    "workflow_file": "run-draftkings-matched-discovery.yml",
                    "display_name": "Discover DraftKings Markets",
                    "inputs": {
                        "sleep_seconds": "3",
                        "max_events": "5",
                    },
                    "outputs": [
                        "data/market/draftkings_market_diagnostic.parquet",
                        "data/market/draftkings_raw_index.parquet",
                    ],
                },
                {
                    "workflow_file": "run-draftkings-normalize-match-only.yml",
                    "display_name": "Normalize And Match DraftKings Markets",
                    "inputs": {
                        "min_match_score": "65",
                    },
                    "outputs": [
                        "data/market/canonical_market_catalog.parquet",
                        "data/market/market_outcomes.parquet",
                        "data/audits/canonical_market_catalog_audit.parquet",
                        "data/audits/ufc_market_match_audit_v2.parquet",
                    ],
                },
            ],
        },
        {
            "step_id": "build_betting_outcomes",
            "display_name": "Build Betting Outcomes",
            "description": "Join production model outcomes to market outcomes and update the active betting outcomes artifacts.",
            "workflows": [
                {
                    "workflow_file": "run-betting-outcomes-v2.yml",
                    "display_name": "Run Betting Outcomes V2",
                    "inputs": {
                        "model_mode": "production",
                    },
                    "outputs": [
                        "data/predictions/betting_outcomes.parquet",
                        "data/audits/ufc_betting_outcomes_audit.parquet",
                        "data/audits/ufc_betting_join_key_diagnostic.parquet",
                    ],
                },
            ],
        },
        {
            "step_id": "capture_snapshots",
            "display_name": "Capture Snapshots",
            "description": "Append model-market snapshots for future CLV and model comparison analysis.",
            "workflows": [
                {
                    "workflow_file": "run-market-refresh-orchestrator.yml",
                    "display_name": "Capture Model-Market Snapshots",
                    "inputs": {
                        "snapshot_model_mode": "all",
                    },
                    "outputs": [
                        "data/snapshots/model_market_snapshots.parquet",
                        "data/audits/model_market_snapshot_audit.parquet",
                    ],
                },
            ],
        },
    ],
}


MONDAY_RESET_V1: RunbookSpec = {
    "runbook_id": "monday_reset_v1",
    "display_name": "Monday Reset",
    "description": "Post-event settlement, dataset refresh, CLV processing, model monitoring, and weekly reset preparation.",
    "steps": [
        {
            "step_id": "process_completed_events",
            "display_name": "Process Completed Events",
            "description": "Discover missing completed events, ingest results, and validate staged fight data.",
            "workflows": [
                {
                    "workflow_file": "run-monday-reset-orchestrator.yml",
                    "display_name": "Ingest Missing Completed Events",
                    "script": "python -m pipeline.data_maintenance.run_ingest_missing_events",
                    "inputs": {
                        "max_events": "1",
                        "auto_append": "false",
                    },
                    "outputs": [
                        "data/status/ufc_ufcstats_event_check.parquet",
                        "data/staging/ufc_missing_events.parquet",
                        "data/staging/ufc_staged_master_rows_profiled.parquet",
                        "data/audits/ufc_missing_event_ingestion_audit.parquet",
                        "data/audits/ufc_append_precheck.parquet",
                        "data/audits/ufc_staged_final_review.parquet",
                    ],
                    "validation_artifacts": [
                        "data/audits/ufc_missing_event_ingestion_audit.parquet",
                    ],
                },
            ],
        },
        {
            "step_id": "update_master_dataset",
            "display_name": "Update Master Dataset",
            "description": "Append validated fight results when auto-append is enabled and all hard gates pass.",
            "workflows": [
                {
                    "workflow_file": "run-monday-reset-orchestrator.yml",
                    "display_name": "Gated Master Append",
                    "script": "python -m pipeline.data_maintenance.run_append_staged_to_master",
                    "inputs": {
                        "auto_append": "false",
                    },
                    "outputs": [
                        "data/master/ufc_master.parquet",
                        "data/backups/ufc_master_backup_before_append_*.parquet",
                        "data/audits/ufc_append_audit.parquet",
                    ],
                    "validation_artifacts": [
                        "data/audits/ufc_append_audit.parquet",
                    ],
                },
            ],
        },
        {
            "step_id": "refresh_platform_status",
            "display_name": "Refresh Platform Status",
            "description": "Rebuild dataset health, event status, and maintenance artifacts after ingestion.",
            "workflows": [
                {
                    "workflow_file": "run-monday-reset-orchestrator.yml",
                    "display_name": "Refresh Dataset Status",
                    "script": "python -m pipeline.data_maintenance.run_dataset_status",
                    "inputs": {},
                    "outputs": [
                        "data/status/ufc_dataset_status.parquet",
                        "data/status/ufc_dataset_event_status.parquet",
                    ],
                    "validation_artifacts": [
                        "data/status/ufc_dataset_status.parquet",
                    ],
                },
            ],
        },
        {
            "step_id": "reconcile_performance",
            "display_name": "Reconcile Performance",
            "description": "Update bankroll status, CLV analysis, and future model-performance tracking.",
            "workflows": [
                {
                    "workflow_file": "run-monday-reset-orchestrator.yml",
                    "display_name": "Refresh Bankroll And CLV",
                    "script": "python -m pipeline.bankroll.run_bankroll_status && python -m pipeline.clv.run_clv_pipeline",
                    "inputs": {
                        "skip_bankroll": "false",
                        "skip_clv": "false",
                    },
                    "outputs": [
                        "data/bankroll/ufc_bankroll_snapshots.parquet",
                        "data/market/ufc_clv_results.parquet",
                        "data/market/ufc_line_movement.parquet",
                    ],
                    "validation_artifacts": [
                        "data/bankroll/ufc_bankroll_snapshots.parquet",
                        "data/market/ufc_clv_results.parquet",
                    ],
                },
            ],
        },
        {
            "step_id": "prepare_next_week",
            "display_name": "Prepare Next Week",
            "description": "Planned step: archive completed-week artifacts and prepare the platform for the next UFC cycle.",
            "status": "planned",
            "workflows": [],
            "planned_outputs": [
                "data/backups/weekly/<week_id>/",
                "data/status/monday_reset_status.json",
                "data/performance/model_prediction_results.parquet",
                "data/performance/model_performance_summary.parquet",
            ],
        },
    ],
}


RUNBOOKS: dict[str, RunbookSpec] = {
    MARKET_REFRESH_V2["runbook_id"]: MARKET_REFRESH_V2,
    MONDAY_RESET_V1["runbook_id"]: MONDAY_RESET_V1,
}

DEFAULT_RUNBOOK_ID = "market_refresh_v2"


def get_runbook(runbook_id: str = DEFAULT_RUNBOOK_ID) -> RunbookSpec:
    """Return a defensive copy of a registered Operations Center runbook."""

    if runbook_id not in RUNBOOKS:
        raise KeyError(f"Unknown runbook_id: {runbook_id}")
    return deepcopy(RUNBOOKS[runbook_id])


def list_runbooks() -> list[RunbookSpec]:
    """Return defensive copies of all registered Operations Center runbooks."""

    return [deepcopy(runbook) for runbook in RUNBOOKS.values()]


def iter_step_workflows(runbook: RunbookSpec) -> list[tuple[RunbookStep, WorkflowSpec]]:
    """Flatten a runbook into step/workflow pairs for launchers and audits."""

    pairs: list[tuple[RunbookStep, WorkflowSpec]] = []
    for step in runbook.get("steps", []):
        for workflow in step.get("workflows", []):
            pairs.append((step, workflow))
    return pairs
