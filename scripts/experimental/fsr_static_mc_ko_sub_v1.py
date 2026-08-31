"""Shadow combined KO/TKO + submission finish layer for the current FSR MC.

Scope
-----
This module preserves the current locked KO/damage/stamina/phase architecture
and adds only submission finish resolution after the existing submission-attempt
event has already been generated.

Existing behavior remains authoritative for:
- phase selection and transitions
- takedowns and ground ownership
- submission-attempt generation via ``submission_pressure``
- stamina costs and timing
- KO/TKO damage-reservoir physics

New behavior:

    existing submission attempt
        -> attacker ``submission_conversion``
        vs defender ``submission_resistance``
        -> probabilistic submission finish

No fatigue modifier is applied to submission traits because the current rolling
FSR contract fatigues striking power only.

The neutral per-attempt finish rate and rating scale are provisional global
simulator constants. They are intentionally isolated here for later cohort
calibration; they are not fighter-specific parameters and are not production
locks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_mature_2020plus_r3_recovery_compare_curve16_exp2_200 as recovery
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_v0 as base


SUBMISSION_NEUTRAL_FINISH_PROBABILITY_PER_ATTEMPT = 0.12
SUBMISSION_RATING_SCALE = 12.0
SUBMISSION_MAX_FINISH_PROBABILITY_PER_ATTEMPT = 0.60
SUBMISSION_FINISH_TRAITS = (
    "submission_conversion",
    "submission_resistance",
)

# Keep the currently frozen round-entry recovery benchmark unchanged.
LOCKED_RECOVERY = recovery.RecoveryCandidate(
    "locked_r3_d60_s0",
    0.60,
    0.00,
)


@dataclass(frozen=True)
class SubmissionFinishEvent:
    """Audit record for one successful submission attempt."""

    winner: int
    loser: int
    probability: float
    conversion: float
    resistance: float
    position: str


def configure_current_finish_candidate() -> None:
    """Apply the same global KO/fatigue constants as the current benchmark."""

    recovery._configure_locked_candidate()


def _strict_submission_rating(profile: pd.Series, trait: str) -> float:
    """Read one required 10-90 FSR submission rating without silent fallback."""

    if trait not in profile.index:
        raise ValueError(f"profile missing required submission trait: {trait}")
    value = profile[trait]
    if pd.isna(value):
        raise ValueError(f"profile has missing required submission trait: {trait}")
    rating = float(value)
    if not np.isfinite(rating):
        raise ValueError(f"profile has non-finite submission trait {trait}: {rating}")
    if not 10.0 <= rating <= 90.0:
        raise ValueError(f"submission trait {trait} outside 10-90: {rating}")
    return rating


class StaticFSRMCKOSUBV1(recovery.RecoveryAuditSim):
    """Current locked KO MC plus per-attempt submission finishes."""

    def __init__(
        self,
        red: pd.Series,
        blue: pd.Series,
        *,
        rounds: int = base.DEFAULT_ROUNDS,
        seed: int = 7,
        red_age: float | None = None,
        blue_age: float | None = None,
        recovery_candidate: recovery.RecoveryCandidate = LOCKED_RECOVERY,
    ) -> None:
        for profile in (red, blue):
            for trait in SUBMISSION_FINISH_TRAITS:
                _strict_submission_rating(profile, trait)

        super().__init__(
            red,
            blue,
            recovery_candidate=recovery_candidate,
            rounds=rounds,
            seed=seed,
            red_age=red_age,
            blue_age=blue_age,
        )
        self.submission_finishes_scored = [0, 0]
        self.submission_finish_events: list[SubmissionFinishEvent] = []

    def _submission_finish_probability(self, attacker: int) -> float:
        """Return P(SUB finish | one existing submission attempt)."""

        defender = self._other(attacker)
        conversion = _strict_submission_rating(
            self.base_fighters[attacker],
            "submission_conversion",
        )
        resistance = _strict_submission_rating(
            self.base_fighters[defender],
            "submission_resistance",
        )
        logit_probability = (
            base._logit(SUBMISSION_NEUTRAL_FINISH_PROBABILITY_PER_ATTEMPT)
            + (conversion - resistance) / SUBMISSION_RATING_SCALE
        )
        return float(
            np.clip(
                base._sigmoid(logit_probability),
                0.0,
                SUBMISSION_MAX_FINISH_PROBABILITY_PER_ATTEMPT,
            )
        )

    def _resolve_submission_attempt(self, attacker: int, position: str) -> bool:
        """Sample one already-generated submission attempt for a finish."""

        defender = self._other(attacker)
        probability = self._submission_finish_probability(attacker)
        if self.rng.random() >= probability:
            return False

        conversion = _strict_submission_rating(
            self.base_fighters[attacker],
            "submission_conversion",
        )
        resistance = _strict_submission_rating(
            self.base_fighters[defender],
            "submission_resistance",
        )
        reservoir = float(self.damage_state[defender].reservoir_current)

        # Reuse the existing finish payload so downstream path consumers keep
        # the same winner/loser/method/round/segment contract. Damage fields are
        # zero because no strike caused this stoppage.
        self.finish = ko.FinishResult(
            winner=attacker,
            loser=defender,
            method="SUB",
            raw_strike_damage=0.0,
            effective_strike_damage=0.0,
            reservoir_before=reservoir,
            reservoir_after=reservoir,
            knockdown_on_strike=False,
            recent_kd_before=bool(self.damage_state[defender].recent_knockdown),
        )
        self.submission_finishes_scored[attacker] += 1
        self.submission_finish_events.append(
            SubmissionFinishEvent(
                winner=attacker,
                loser=defender,
                probability=probability,
                conversion=conversion,
                resistance=resistance,
                position=str(position),
            )
        )
        return True

    def _ground_transition(self) -> str:
        """Preserve existing ground behavior, then resolve any generated attempts."""

        controller = self.ground_controller
        if controller is None:
            return super()._ground_transition()
        bottom = self._other(controller)

        sub_before = [int(self.stats[0].sub_att), int(self.stats[1].sub_att)]

        # This call remains authoritative for control credit, submission-attempt
        # generation, ground exits/reversals, and all existing stamina costs.
        original_note = super()._ground_transition()

        attempted: list[tuple[int, str]] = []
        if int(self.stats[controller].sub_att) > sub_before[controller]:
            attempted.append((controller, "top"))
        if int(self.stats[bottom].sub_att) > sub_before[bottom]:
            attempted.append((bottom, "bottom"))

        if not attempted:
            return original_note

        # If both fighters generated an attempt in the same 10-second segment,
        # randomize finish-resolution order to avoid a permanent top-side bias.
        for idx in self.rng.permutation(len(attempted)):
            attacker, position = attempted[int(idx)]
            if self._resolve_submission_attempt(attacker, position):
                probability = self._submission_finish_probability(attacker)
                return (
                    f"{self.names[attacker]} submission attempt from {position}; "
                    f"SUB SUCCESS ({probability:.3f}) -> STOPPAGE"
                )

        return original_note

    def run(self) -> ko.KOPath:
        """Run the current segment loop with method-agnostic finish metadata."""

        events: list[dict[str, Any]] = []

        for round_no in range(1, self.rounds + 1):
            self.phase = "DISTANCE"
            self.ground_controller = None
            self.clinch_controller = None
            self.clinch_initiator = None

            for segment_no in range(1, base.SEGMENTS_PER_ROUND + 1):
                self._refresh_effective_fighters(round_no, segment_no)
                self.pending_stamina_costs = [[], []]

                phase_start = self.phase
                ground_controller_start = self.ground_controller
                clinch_controller_start = self.clinch_controller

                for stats in self.stats:
                    stats.phase_segments[phase_start] += 1

                strike_notes = self._generate_striking(phase_start)

                if self.finish is not None:
                    transition_note = (
                        f"fight stopped: {self.names[self.finish.winner]} "
                        f"{self.finish.method} {self.names[self.finish.loser]}"
                    )
                elif phase_start == "DISTANCE":
                    transition_note = self._distance_transition()
                elif phase_start == "CLINCH":
                    transition_note = self._clinch_transition()
                else:
                    transition_note = self._ground_transition()

                # Submission finishes are created during the ground transition;
                # KO finishes are usually created during striking. Stamp either
                # method after all actions in the segment have resolved.
                if self.finish is not None and self.finish.round is None:
                    self.finish.round = round_no
                    self.finish.segment = segment_no
                    self.finish.clock_start = self._clock_start(segment_no)

                # Preserve locked action-first, fatigue-after-segment timing.
                self._flush_pending_stamina_costs()

                events.append(
                    {
                        "round": round_no,
                        "segment": segment_no,
                        "clock_start": self._clock_start(segment_no),
                        "phase_start": phase_start,
                        "phase_end": self.phase,
                        "top_start": (
                            self.names[ground_controller_start]
                            if ground_controller_start is not None
                            else None
                        ),
                        "top_end": (
                            self.names[self.ground_controller]
                            if self.ground_controller is not None
                            else None
                        ),
                        "clinch_controller_start": (
                            self.names[clinch_controller_start]
                            if clinch_controller_start is not None
                            else None
                        ),
                        "clinch_controller_end": (
                            self.names[self.clinch_controller]
                            if self.clinch_controller is not None
                            else None
                        ),
                        "striking": "; ".join(strike_notes) if strike_notes else "no sig attempts",
                        "transition": transition_note,
                        "finish": self.finish is not None,
                        "finish_method": self.finish.method if self.finish is not None else "",
                        "red_stamina_after": self.stamina_state[0].fraction,
                        "blue_stamina_after": self.stamina_state[1].fraction,
                    }
                )

                if self.finish is not None:
                    return ko.KOPath(events=events, stats=self.stats, finish=self.finish)

            if round_no < self.rounds:
                self._apply_between_round_recovery(round_no)

        return ko.KOPath(events=events, stats=self.stats, finish=None)
