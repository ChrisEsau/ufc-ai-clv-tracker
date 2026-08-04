"""Shadow-only artifact paths for the simulator research workspace."""

from pipeline.common.paths import AUDITS_DIR, MODEL_LAB_DIR

SIMULATION_DIR = MODEL_LAB_DIR / "simulation"
SIMULATION_TRAINING_DIR = SIMULATION_DIR / "training"
SIMULATION_MODELS_DIR = SIMULATION_DIR / "models"

SIMULATION_TRAINING_DATASET_PATH = (
    SIMULATION_TRAINING_DIR / "fighter_round_parameter_training.parquet"
)
SIMULATION_TRAINING_AUDIT_PATH = (
    AUDITS_DIR / "simulation_fighter_round_training_audit.parquet"
)
SIMULATION_LATEST_SUMMARY_PATH = SIMULATION_DIR / "latest_round_simulation_summary.json"

SIG_ATTEMPT_MODEL_DIR = SIMULATION_MODELS_DIR / "sig_attempt_pace_v0"
SIG_ATTEMPT_FOLD_METRICS_PATH = SIG_ATTEMPT_MODEL_DIR / "fold_metrics.csv"
SIG_ATTEMPT_AGGREGATE_METRICS_PATH = SIG_ATTEMPT_MODEL_DIR / "aggregate_metrics.csv"
SIG_ATTEMPT_PREDICTIONS_PATH = SIG_ATTEMPT_MODEL_DIR / "walk_forward_predictions.parquet"
SIG_ATTEMPT_FEATURE_IMPORTANCE_PATH = SIG_ATTEMPT_MODEL_DIR / "feature_importance.csv"
SIG_ATTEMPT_MODEL_BUNDLE_PATH = SIG_ATTEMPT_MODEL_DIR / "holdout_model_bundle.joblib"
SIG_ATTEMPT_SUMMARY_PATH = SIG_ATTEMPT_MODEL_DIR / "benchmark_summary.json"
SIG_ATTEMPT_MODEL_CARD_PATH = SIG_ATTEMPT_MODEL_DIR / "model_card.md"

SIG_ATTEMPT_CALIBRATION_DIR = SIG_ATTEMPT_MODEL_DIR / "calibration"
SIG_ATTEMPT_CALIBRATION_SCHEDULE_PATH = (
    SIG_ATTEMPT_CALIBRATION_DIR / "sequential_calibration_schedule.csv"
)
SIG_ATTEMPT_CALIBRATED_PREDICTIONS_PATH = (
    SIG_ATTEMPT_CALIBRATION_DIR / "calibrated_walk_forward_predictions.parquet"
)
SIG_ATTEMPT_CALIBRATION_METRICS_PATH = (
    SIG_ATTEMPT_CALIBRATION_DIR / "calibration_metrics.csv"
)
SIG_ATTEMPT_FINAL_PARAMETERS_PATH = (
    SIG_ATTEMPT_CALIBRATION_DIR / "final_distribution_parameters.csv"
)
SIG_ATTEMPT_CALIBRATION_SUMMARY_PATH = (
    SIG_ATTEMPT_CALIBRATION_DIR / "calibration_summary.json"
)


def ensure_simulation_dirs() -> None:
    """Create generated-artifact folders used only by simulator research."""
    for path in (
        SIMULATION_DIR,
        SIMULATION_TRAINING_DIR,
        SIMULATION_MODELS_DIR,
        SIG_ATTEMPT_MODEL_DIR,
        SIG_ATTEMPT_CALIBRATION_DIR,
        SIMULATION_TRAINING_AUDIT_PATH.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
