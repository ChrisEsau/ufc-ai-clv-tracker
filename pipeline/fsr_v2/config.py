"""Visible preliminary parameters for FSR V2.

These values are deliberately centralized and fingerprinted. They are starting
points for the sanity gate, not silently inherited production calibration.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FSRV2Config:
    prior_rating: float = 0.0
    rating_scale: float = 1.0
    elo_k: float = 0.35
    evidence_saturation_attempts: float = 12.0
    behavior_prior_seconds: float = 900.0
    behavior_prior_opportunities: float = 20.0
    suppression_prior_seconds: float = 900.0
    escape_prior_entries: float = 3.0
    zero_td_control_threshold_seconds: float = 5.0
    zero_control_ground_fallback_seconds: float = 5.0
    maximum_round_seconds: float = 300.0
    rate_seconds: float = 60.0

    def fingerprint_payload(self) -> dict[str, float]:
        return asdict(self)
