"""Explicit replay-group registry and deterministic version fingerprints."""

from dataclasses import dataclass
import hashlib
import json

from pipeline.fsr_v2.config import FSRV2Config


@dataclass(frozen=True)
class TraitGroup:
    name: str
    kind: str
    traits: tuple[str, ...]
    numerator: str
    denominator: str
    dependencies: tuple[str, ...] = ()
    experimental: bool = False

    def fingerprint(self, config: FSRV2Config, source_fingerprint: str) -> str:
        payload = {**self.__dict__, "config": config.fingerprint_payload(), "source": source_fingerprint}
        if self.name == "submission_tendency":
            payload["implementation_version"] = "fight_count_population_prior_v1"
        if self.name == "takedown_tendency":
            payload["implementation_version"] = "endpoint2_raw_after_two_prior_ufc_fights_v1"
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


GROUPS = {
    "standing_striking_tendency": TraitGroup("standing_striking_tendency", "behavior", ("standing_striking_tendency",), "distance_attempted", "standing_exposure_seconds"),
    "standing_striking_suppression": TraitGroup("standing_striking_suppression", "suppression", ("standing_striking_suppression",), "opponent_distance_attempted", "standing_exposure_seconds", ("standing_striking_tendency",)),
    "standing_striking_effectiveness": TraitGroup("standing_striking_effectiveness", "paired", ("standing_striking_offense", "standing_striking_defense"), "distance_landed", "distance_attempted"),
    "head_body_tendency": TraitGroup("head_body_tendency", "composition", ("head_strike_tendency", "body_strike_tendency"), "head_attempted", "head_body_attempted"),
    "leg_strike_tendency": TraitGroup("leg_strike_tendency", "composition", ("leg_strike_tendency",), "leg_attempted", "target_attempted"),
    "takedown_tendency": TraitGroup("takedown_tendency", "behavior", ("takedown_tendency",), "td_attempted", "td_tendency_exposure_seconds"),
    "takedown_suppression": TraitGroup("takedown_suppression", "suppression", ("takedown_suppression",), "opponent_td_attempted", "td_suppression_exposure_seconds", ("takedown_tendency",)),
    "takedown_effectiveness": TraitGroup("takedown_effectiveness", "takedown", ("takedown_offense", "takedown_defense"), "td_landed", "td_attempted"),
    "escape_effectiveness": TraitGroup("escape_effectiveness", "escape", ("escape_offense", "escape_defense"), "opponent_ctrl_sec", "opponent_ground_entries"),
    "ground_striking_tendency": TraitGroup("ground_striking_tendency", "behavior", ("ground_striking_tendency",), "ground_attempted", "modeled_ground_exposure_seconds"),
    "ground_striking_suppression": TraitGroup("ground_striking_suppression", "suppression", ("ground_striking_suppression",), "opponent_ground_attempted", "modeled_ground_exposure_seconds", ("ground_striking_tendency",)),
    "ground_striking_effectiveness": TraitGroup("ground_striking_effectiveness", "paired", ("ground_striking_offense", "ground_striking_defense"), "ground_landed", "ground_attempted"),
    "submission_tendency": TraitGroup("submission_tendency", "behavior", ("submission_tendency",), "effective_submission_attempts", "fight_elapsed_seconds"),
    "submission_suppression": TraitGroup("submission_suppression", "suppression", ("submission_suppression",), "opponent_effective_submission_attempts", "fight_elapsed_seconds", ("submission_tendency",)),
    "submission_effectiveness": TraitGroup("submission_effectiveness", "paired", ("submission_offense", "submission_defense"), "submission_finish", "effective_submission_attempts"),
    "reversal_tendency": TraitGroup("reversal_tendency", "behavior", ("reversal_tendency",), "rev", "modeled_ground_exposure_seconds", experimental=True),
}

ALIASES = {
    "standing_striking": ("standing_striking_tendency", "standing_striking_suppression", "standing_striking_effectiveness"),
    "targets": ("head_body_tendency", "leg_strike_tendency"),
    "takedowns": ("takedown_tendency", "takedown_suppression", "takedown_effectiveness"),
    "ground_striking": ("ground_striking_tendency", "ground_striking_suppression", "ground_striking_effectiveness"),
    "submissions": ("submission_tendency", "submission_suppression", "submission_effectiveness"),
}


def resolve_groups(names: list[str] | None, include_experimental: bool = False) -> list[TraitGroup]:
    expanded: list[str] = []
    for name in names or GROUPS:
        expanded.extend(ALIASES.get(name, (name,)))
    unknown = set(expanded).difference(GROUPS)
    if unknown:
        raise ValueError(f"Unknown FSR V2 trait groups: {sorted(unknown)}")
    groups = [GROUPS[name] for name in dict.fromkeys(expanded)]
    return [group for group in groups if include_experimental or not group.experimental]
