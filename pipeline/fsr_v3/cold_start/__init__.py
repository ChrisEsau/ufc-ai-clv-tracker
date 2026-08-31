"""Leakage-safe external-evidence cold-start research layer for FSR V3.

This package is intentionally not wired into production publication.  It builds
and validates fighter-specific priors from objective pre-UFC evidence; promotion
requires held-out native-target gains.
"""

from .features import build_external_feature_snapshots
from .model import ColdStartNB2RateModel, calibrate_extra_evidence_seconds
from .priors import combine_positive_rate_prior

__all__ = [
    "ColdStartNB2RateModel",
    "build_external_feature_snapshots",
    "calibrate_extra_evidence_seconds",
    "combine_positive_rate_prior",
]
