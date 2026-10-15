"""Resolve cached RFS state into evidence bundles for simulator targets.

This module does not yet calibrate final simulator parameter values.
Its job is to bridge the leakage-safe RFS profile to the approved
37-target simulator feature contract in a transparent way.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import pandas as pd

from pipeline.round_stats.rfs_phase_baseline_feature_contracts import (
    PHASE_BASELINE_TARGET_EVIDENCE,
)
from pipeline.round_stats.rfs_phase_interaction_feature_contracts import (
    PHASE_INTERACTION_TARGET_EVIDENCE,
)
from pipeline.round_stats.rfs_dynamic_response_feature_contracts import (
    DYNAMIC_RESPONSE_TARGET_EVIDENCE,
)
from pipeline.round_stats.rfs_finish_state_feature_contracts import (
    FINISH_STATE_TARGET_EVIDENCE,
)
from pipeline.round_stats.rfs_simulator_feature_contracts import (
    SIMULATOR_TARGET_BY_NAME,
    SIMULATOR_TARGET_SPECS,
)


class RFSTargetResolutionError(RuntimeError):
    """Raised when RFS evidence cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedEvidence:
    """One historical state value supporting one simulator target."""

    fight_feature_name: str
    state_feature_name: str
    state_kind: str
    value: float


@dataclass(frozen=True)
class ResolvedTargetEvidence:
    """Resolved evidence bundle for one simulator parameter."""

    target_parameter: str
    family: str
    value_kind: str
    support_level: str
    evidence: tuple[ResolvedEvidence, ...]
    requested_evidence_count: int

    @property
    def resolved_evidence_count(self) -> int:
        return len(self.evidence)

    @property
    def coverage(self) -> float:
        if self.requested_evidence_count == 0:
            return 0.0

        return (
            self.resolved_evidence_count
            / self.requested_evidence_count
        )


TARGET_EVIDENCE_MAPS = (
    PHASE_BASELINE_TARGET_EVIDENCE,
    PHASE_INTERACTION_TARGET_EVIDENCE,
    DYNAMIC_RESPONSE_TARGET_EVIDENCE,
    FINISH_STATE_TARGET_EVIDENCE,
)


def _build_target_evidence_registry() -> dict[str, tuple[str, ...]]:
    """Combine the four approved family evidence mappings."""

    combined: dict[str, tuple[str, ...]] = {}

    for mapping in TARGET_EVIDENCE_MAPS:
        for target, evidence in mapping.items():
            if target in combined:
                raise RFSTargetResolutionError(
                    f"duplicate target evidence mapping: {target}"
                )

            combined[target] = tuple(evidence)

    expected = set(SIMULATOR_TARGET_BY_NAME)
    observed = set(combined)

    if observed != expected:
        raise RFSTargetResolutionError(
            "target evidence registry does not exactly match simulator "
            f"targets; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )

    return combined


TARGET_EVIDENCE = _build_target_evidence_registry()


def _state_prefix_and_suffix(
    fight_feature_name: str,
) -> tuple[str, str]:
    """Convert a fight-observation feature into state namespace + suffix."""

    mappings = (
        (
            "rfs_phase_base_fight_",
            "rfs_phase_base_",
        ),
        (
            "rfs_phase_interact_fight_",
            "rfs_phase_interact_",
        ),
        (
            "rfs_dynamic_response_fight_",
            "rfs_dynamic_response_",
        ),
        (
            "rfs_finish_state_fight_",
            "rfs_finish_state_",
        ),
    )

    for fight_prefix, state_prefix in mappings:
        if fight_feature_name.startswith(fight_prefix):
            return (
                state_prefix,
                fight_feature_name.removeprefix(fight_prefix),
            )

    raise RFSTargetResolutionError(
        "unsupported fight evidence feature: "
        f"{fight_feature_name}"
    )


def _candidate_state_names(
    fight_feature_name: str,
) -> tuple[tuple[str, str], ...]:
    """Return state candidates in resolver preference order."""

    prefix, suffix = _state_prefix_and_suffix(
        fight_feature_name
    )

    return (
        ("ewm", f"{prefix}ewm_{suffix}"),
        ("exp", f"{prefix}exp_{suffix}"),
        ("last3", f"{prefix}last3_{suffix}"),
    )


def _finite_float(value: object) -> float | None:
    """Return a finite float or None for unavailable evidence."""

    if value is None or pd.isna(value):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(numeric):
        return None

    return numeric


def _resolve_one_evidence(
    profile: Mapping[str, object],
    fight_feature_name: str,
) -> ResolvedEvidence | None:
    """Resolve one approved fight feature to available prior state."""

    for state_kind, state_name in _candidate_state_names(
        fight_feature_name
    ):
        if state_name not in profile:
            continue

        value = _finite_float(
            profile[state_name]
        )

        if value is None:
            continue

        return ResolvedEvidence(
            fight_feature_name=fight_feature_name,
            state_feature_name=state_name,
            state_kind=state_kind,
            value=value,
        )

    return None


def resolve_target_evidence(
    profile: Mapping[str, object],
) -> dict[str, ResolvedTargetEvidence]:
    """Resolve all 37 simulator targets from one cached RFS profile."""

    resolved: dict[str, ResolvedTargetEvidence] = {}

    for target_spec in SIMULATOR_TARGET_SPECS:
        target = target_spec.target_parameter
        requested = TARGET_EVIDENCE[target]

        evidence: list[ResolvedEvidence] = []

        for fight_feature_name in requested:
            item = _resolve_one_evidence(
                profile,
                fight_feature_name,
            )

            if item is not None:
                evidence.append(item)

        resolved[target] = ResolvedTargetEvidence(
            target_parameter=target,
            family=target_spec.primary_family.value,
            value_kind=target_spec.value_kind.value,
            support_level=target_spec.support_level.value,
            evidence=tuple(evidence),
            requested_evidence_count=len(requested),
        )

    return resolved


def summarize_target_resolution(
    resolved: Mapping[str, ResolvedTargetEvidence],
) -> pd.DataFrame:
    """Return a compact audit table for one fighter profile."""

    rows = []

    for target in sorted(resolved):
        bundle = resolved[target]

        rows.append(
            {
                "target_parameter": target,
                "family": bundle.family,
                "value_kind": bundle.value_kind,
                "support_level": bundle.support_level,
                "requested_evidence": (
                    bundle.requested_evidence_count
                ),
                "resolved_evidence": (
                    bundle.resolved_evidence_count
                ),
                "coverage": bundle.coverage,
                "state_kinds": ",".join(
                    sorted(
                        {
                            item.state_kind
                            for item in bundle.evidence
                        }
                    )
                ),
            }
        )

    return pd.DataFrame(rows)
