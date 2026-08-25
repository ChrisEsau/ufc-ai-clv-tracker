"""Controlled historical calibration infrastructure for Event Clock V2."""

COHORT_VERSION = "event_clock_v2_calibration_cohort_v1"
TARGET_VERSION = "event_clock_v2_historical_targets_v1"
SEED_SET_VERSION = "event_clock_v2_matched_seeds_v1"
# Pinned outside the payload so candidates cannot rewrite targets and their
# self-declared digest together. Updated only when a reviewed version is frozen.
EXPECTED_TARGET_DIGEST = (
    "sha256:13c24864a047831ab7f1b7939348baae73c12cb8bf2a26016beb0edf85e65d13"
)
