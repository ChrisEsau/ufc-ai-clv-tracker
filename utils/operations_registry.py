from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationAction:
    label: str
    description: str
    workflow: str | None
    status_key: str
    enabled: bool = True


@dataclass(frozen=True)
class OperationGroup:
    title: str
    icon: str
    subtitle: str
    status_key: str
    actions: tuple[OperationAction, ...]


OPERATION_GROUPS: tuple[OperationGroup, ...] = (
    OperationGroup(
        title="Market Operations",
        icon="🌐",
        subtitle="Update odds, market snapshots, and CLV inputs.",
        status_key="market",
        actions=(
            OperationAction("Refresh Market Odds", "Run active market odds update.", "run-market-update.yml", "market"),
            OperationAction("Refresh DraftKings", "DraftKings-specific market refresh placeholder.", None, "market", enabled=False),
            OperationAction("Refresh Props", "Prop market refresh placeholder.", None, "market", enabled=False),
            OperationAction("Update CLV Tracking", "Run closing-line value tracker.", "run-clv-tracker.yml", "clv"),
        ),
    ),
    OperationGroup(
        title="Prediction Operations",
        icon="📈",
        subtitle="Build live features, predictions, and betting board artifacts.",
        status_key="predictions",
        actions=(
            OperationAction("Build Live Features", "Refresh current fighter feature store.", "run-current-fighter-features.yml", "features"),
            OperationAction("Generate Predictions", "Run model predictions for the live card.", "run-model-predictions.yml", "predictions"),
            OperationAction("Run Live Prediction", "Full live prediction workflow.", "run-live-prediction.yml", "predictions"),
            OperationAction("Build Betting Board", "Betting board artifact builder placeholder.", None, "betting", enabled=False),
        ),
    ),
    OperationGroup(
        title="Model Operations",
        icon="🧠",
        subtitle="Train, backtest, compare, and promote model versions.",
        status_key="model",
        actions=(
            OperationAction("Train New Model", "Training workflow placeholder.", None, "model", enabled=False),
            OperationAction("Run Backtest", "Backtest workflow placeholder.", None, "model", enabled=False),
            OperationAction("Compare Models", "Model comparison placeholder.", None, "model", enabled=False),
            OperationAction("Promote to Production", "Promotion workflow placeholder.", None, "model", enabled=False),
        ),
    ),
    OperationGroup(
        title="Data Operations",
        icon="🗄️",
        subtitle="Discover, ingest, validate, and append historical data.",
        status_key="data",
        actions=(
            OperationAction("Run Dataset Status", "Refresh master dataset health.", "run-dataset-status.yml", "data"),
            OperationAction("Discover New Events", "Find UFCStats events missing from master.", "run-ufcstats-event-check.yml", "data"),
            OperationAction("Validate Staged Data", "Run append precheck and final review.", "run-append-precheck-validation.yml", "data"),
            OperationAction("Append To Master", "Use the existing Data Maintenance append gate.", None, "data", enabled=False),
        ),
    ),
    OperationGroup(
        title="System Operations",
        icon="⚙️",
        subtitle="Utility checks and maintenance placeholders.",
        status_key="system",
        actions=(
            OperationAction("System Health Check", "System status placeholder.", None, "system", enabled=False),
            OperationAction("Storage Optimization", "Storage maintenance placeholder.", None, "system", enabled=False),
            OperationAction("Backup Verification", "Backup verification placeholder.", None, "system", enabled=False),
            OperationAction("Clear Cache", "Use the global refresh action for now.", None, "system", enabled=False),
        ),
    ),
)
