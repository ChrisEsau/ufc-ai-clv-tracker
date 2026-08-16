"""Locked constants for the final validated FSR V2 trait contract."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FSRV2Config:
    prior_rating: float = 0.0
    rating_scale: float = 1.0
    elo_k: float = 0.35
    evidence_saturation_attempts: float = 12.0
    submission_effectiveness_saturation_attempts: float = 5.0
    behavior_prior_seconds: float = 900.0
    suppression_prior_seconds: float = 900.0
    target_composition_prior_attempts: float = 200.0
    takedown_effectiveness_prior_attempts: float = 10.0
    escape_prior_entries: float = 5.0
    zero_td_control_threshold_seconds: float = 5.0
    zero_control_ground_fallback_seconds: float = 5.0
    maximum_round_seconds: float = 300.0
    logit_epsilon: float = 1e-6
    rate_seconds: float = 60.0

    def fingerprint_payload(self) -> dict[str, float]:
        return asdict(self)
