from __future__ import annotations

from copy import deepcopy
from typing import Any


WorkflowSpec = dict[str, Any]
RunbookStep = dict[str, Any]
RunbookSpec = dict[str, Any]


MARKET_REFRESH_V2: RunbookSpec = {
    "runbook_id": "market_refresh_v2",
    "display_name": "Market Refresh",
    "description": "Refresh upcoming UFC events, production predictions, DraftKings markets, and betting outcomes.",
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
            "description": "Refresh fighter state and generate production model outcome predictions.",
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
            "description": "Join model outcomes to market outcomes and update the betting outcomes artifacts.",
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
    ],
}


RUNBOOKS: dict[str, RunbookSpec] = {
    MARKET_REFRESH_V2["runbook_id"]: MARKET_REFRESH_V2,
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
