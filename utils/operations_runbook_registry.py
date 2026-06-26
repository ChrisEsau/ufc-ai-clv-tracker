from __future__ import annotations

from copy import deepcopy
from typing import Any


WorkflowSpec = dict[str, Any]
RunbookStep = dict[str, Any]
RunbookSpec = dict[str, Any]


MARKET_REFRESH_V2: RunbookSpec = {
    "runbook_id": "market_refresh_v2",
    "display_name": "Market Refresh",
    "description": "Refresh all production model predictions, DraftKings markets, betting outcomes, and model-market snapshots using the Monday-selected target card.",
    "steps": [
        {
            "step_id": "run_predictions",
            "display_name": "Run All Production Predictions",
            "description": "Refresh fighter state, rebuild the production feature view, and generate model-scoped outcomes for every registry status=production model. The target live card is owned by Monday Reset.",
            "workflows": [
                {"workflow_file": "run-build-fighter-state-v2.yml", "display_name": "Build Fighter State V2", "inputs": {}, "outputs": ["data/features/fighter_state_history.parquet", "data/features/latest_fighter_state.parquet"]},
                {"workflow_file": "run-build-feature-view-v2.yml", "display_name": "Build Feature View V2", "inputs": {"config_path": "configs/feature_views/moneyline_base.yaml", "output_path": "data/features/moneyline_feature_view.parquet"}, "outputs": ["data/features/moneyline_feature_view.parquet"]},
                {"workflow_file": "run-market-refresh-orchestrator.yml", "display_name": "Run Production Registry Predictions", "inputs": {"mode": "production"}, "outputs": ["data/predictions/by_model/moneyline_xgboost_v7/model_outcomes.parquet", "data/predictions/by_model/prop_goes_distance_xgboost_v1/model_outcomes.parquet", "data/audits/production_prediction_audit.parquet", "data/audits/ufc_live_feature_audit.parquet"]},
            ],
        },
        {
            "step_id": "get_market_odds",
            "display_name": "Get Market Odds",
            "description": "Discover DraftKings events, filter to live-card matches, collect markets, then normalize and match market outcomes.",
            "workflows": [
                {"workflow_file": "run-draftkings-event-index.yml", "display_name": "Build DraftKings Event Index", "inputs": {}, "outputs": ["data/market/draftkings_event_index.parquet"]},
                {"workflow_file": "run-draftkings-card-filter.yml", "display_name": "Match DraftKings Events To Card", "inputs": {"min_match_score": "80", "min_single_score": "70"}, "outputs": ["data/market/draftkings_event_card_matches.parquet"]},
                {"workflow_file": "run-draftkings-matched-discovery.yml", "display_name": "Discover DraftKings Markets", "inputs": {"sleep_seconds": "3", "max_events": "all"}, "outputs": ["data/market/draftkings_market_diagnostic.parquet", "data/market/draftkings_raw_index.parquet"]},
                {"workflow_file": "run-draftkings-normalize-match-only.yml", "display_name": "Normalize And Match DraftKings Markets", "inputs": {"min_match_score": "65"}, "outputs": ["data/market/canonical_market_catalog.parquet", "data/market/market_outcomes.parquet", "data/audits/canonical_market_catalog_audit.parquet", "data/audits/ufc_market_match_audit_v2.parquet"]},
            ],
        },
        {
            "step_id": "build_betting_outcomes",
            "display_name": "Build Betting Outcomes",
            "description": "Join production model outcomes to market outcomes and update the active betting outcomes artifacts.",
            "workflows": [{"workflow_file": "run-betting-outcomes-v2.yml", "display_name": "Run Betting Outcomes V2", "inputs": {"model_mode": "production"}, "outputs": ["data/predictions/betting_outcomes.parquet", "data/audits/ufc_betting_outcomes_audit.parquet", "data/audits/ufc_betting_join_key_diagnostic.parquet"]}],
        },
        {
            "step_id": "capture_snapshots",
            "display_name": "Capture Snapshots",
            "description": "Append production model-market snapshots for future CLV and model comparison analysis.",
            "workflows": [{"workflow_file": "run-market-refresh-orchestrator.yml", "display_name": "Capture Model-Market Snapshots", "inputs": {"mode": "production"}, "outputs": ["data/snapshots/model_market_snapshots.parquet", "data/audits/model_market_snapshot_audit.parquet"]}],
        },
    ],
}


MONDAY_RESET_V1: RunbookSpec = {
    "runbook_id": "monday_reset_v1",
    "display_name": "Monday Reset",
    "description": "Post-event settlement, dataset refresh, CLV processing, model monitoring, and weekly reset preparation.",
    "steps": [
        {"step_id": "process_completed_events", "display_name": "Process Completed Events", "description": "Discover missing completed events, ingest results, and validate staged fight data.", "workflows": [{"workflow_file": "run-monday-reset-orchestrator.yml", "display_name": "Ingest Missing Completed Events", "inputs": {"max_events": "all", "auto_append": "true"}, "outputs": ["data/audits/ufc_missing_event_ingestion_audit.parquet"]}]},
        {"step_id": "update_master_dataset", "display_name": "Update Master Dataset", "description": "Append validated fight results when auto-append is enabled and all hard gates pass.", "workflows": [{"workflow_file": "run-monday-reset-orchestrator.yml", "display_name": "Gated Master Append", "inputs": {"auto_append": "true"}, "outputs": ["data/master/ufc_master.parquet", "data/audits/ufc_append_audit.parquet"]}]},
        {"step_id": "refresh_platform_status", "display_name": "Refresh Platform Status", "description": "Rebuild dataset health, event status, target event, and live-card artifacts after ingestion.", "workflows": [{"workflow_file": "run-monday-reset-orchestrator.yml", "display_name": "Refresh Dataset Status And Target Card", "inputs": {}, "outputs": ["data/status/ufc_dataset_status.parquet", "data/status/ufc_dataset_event_status.parquet", "data/cards/ufc_selected_live_card_event.parquet", "data/predictions/ufc_live_card.parquet"]}]},
        {"step_id": "reconcile_performance", "display_name": "Reconcile Performance", "description": "Update bankroll status, CLV analysis, and future model-performance tracking.", "workflows": [{"workflow_file": "run-monday-reset-orchestrator.yml", "display_name": "Refresh Bankroll And CLV", "inputs": {"skip_bankroll": "false", "skip_clv": "false"}, "outputs": ["data/bankroll/ufc_bankroll_snapshots.parquet", "data/market/ufc_clv_results.parquet", "data/market/ufc_line_movement.parquet"]}]},
        {"step_id": "prepare_next_week", "display_name": "Prepare Next Week", "description": "Planned step: archive completed-week artifacts and prepare the platform for the next UFC cycle.", "status": "planned", "workflows": [], "planned_outputs": ["data/backups/weekly/<week_id>/", "data/status/monday_reset_status.json", "data/performance/model_prediction_results.parquet", "data/performance/model_performance_summary.parquet"]},
    ],
}


FIGHT_DAY_MONITOR_V1: RunbookSpec = {
    "runbook_id": "fight_day_monitor_v1",
    "display_name": "Fight Day Monitor",
    "description": "Refresh today's DraftKings markets, recalculate EV and Kelly, capture model-market snapshots, and store official closing lines.",
    "steps": [
        {
            "step_id": "refresh_live_markets",
            "display_name": "Refresh Live Markets",
            "description": "Update DraftKings prices for the selected target card without rebuilding UFCStats, fighter state, feature views, or predictions.",
            "workflows": [{"workflow_file": "run-fight-day-monitor.yml", "display_name": "Refresh Live Markets", "inputs": {"mode": "production", "max_draftkings_events": "all"}, "outputs": ["data/market/draftkings_event_index.parquet", "data/market/draftkings_event_card_matches.parquet", "data/market/market_outcomes.parquet"]}],
        },
        {
            "step_id": "recalculate_betting_board",
            "display_name": "Recalculate Betting Board",
            "description": "Recalculate EV, edge, bet status, and Kelly sizing from current market prices and existing production model outputs.",
            "workflows": [{"workflow_file": "run-fight-day-monitor.yml", "display_name": "Run Betting Outcomes V2", "inputs": {"model_mode": "production"}, "outputs": ["data/predictions/betting_outcomes.parquet", "data/audits/ufc_betting_outcomes_audit.parquet"]}],
        },
        {
            "step_id": "capture_model_market_snapshot",
            "display_name": "Capture Snapshot",
            "description": "Append current production model-vs-market state for CLV and model comparison analysis.",
            "workflows": [{"workflow_file": "run-fight-day-monitor.yml", "display_name": "Capture Model-Market Snapshot", "inputs": {"snapshot_model_mode": "production"}, "outputs": ["data/snapshots/model_market_snapshots.parquet", "data/audits/model_market_snapshot_audit.parquet"]}],
        },
        {
            "step_id": "capture_closing_lines",
            "display_name": "Capture Closing Lines",
            "description": "Append the official pre-fight closing-line snapshot from the latest betting outcomes artifact.",
            "workflows": [{"workflow_file": "run-fight-day-monitor.yml", "display_name": "Capture Closing-Line Snapshot", "inputs": {"official_closing_snapshot": True}, "outputs": ["data/snapshots/closing_line_snapshots.parquet", "data/audits/closing_line_snapshot_audit.parquet"]}],
        },
    ],
}


RUNBOOKS: dict[str, RunbookSpec] = {
    MARKET_REFRESH_V2["runbook_id"]: MARKET_REFRESH_V2,
    MONDAY_RESET_V1["runbook_id"]: MONDAY_RESET_V1,
    FIGHT_DAY_MONITOR_V1["runbook_id"]: FIGHT_DAY_MONITOR_V1,
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
