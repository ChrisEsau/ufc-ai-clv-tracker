"""Shadow-only artifact paths for the simulator research workspace."""

from pipeline.common.paths import AUDITS_DIR, MODEL_LAB_DIR

SIMULATION_DIR = MODEL_LAB_DIR / "simulation"
SIMULATION_TRAINING_DIR = SIMULATION_DIR / "training"
SIMULATION_TRAINING_DATASET_PATH = (
    SIMULATION_TRAINING_DIR / "fighter_round_parameter_training.parquet"
)
SIMULATION_TRAINING_AUDIT_PATH = (
    AUDITS_DIR / "simulation_fighter_round_training_audit.parquet"
)
SIMULATION_LATEST_SUMMARY_PATH = SIMULATION_DIR / "latest_round_simulation_summary.json"


def ensure_simulation_dirs() -> None:
    """Create generated-artifact folders used only by simulator research."""
    for path in (SIMULATION_DIR, SIMULATION_TRAINING_DIR, SIMULATION_TRAINING_AUDIT_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
