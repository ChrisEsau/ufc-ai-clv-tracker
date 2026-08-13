"""Phase 5A round-local deterministic judging and decision resolution."""

from collections import Counter
from dataclasses import dataclass, field

from .calibration import DEFAULT_CALIBRATION, EventMCCalibration
from .components.actions import ActionAttempt, ActionOutcome
from .components.profiles import Side
from .events import ConsequenceEvent, PrimaryEvent
from .physiology import PhysiologyOutcome
from .state import StateDelta
from .submission_finishes import SubmissionFinishOutcome

OFFENSIVE_INITIATIVE_FAMILIES = frozenset(
    {
        "strike",
        "takedown",
        "clinch_entry",
        "clinch_strike",
        "clinch_takedown",
        "ground_strike",
        "submission_attempt",
    }
)


@dataclass(frozen=True)
class RoundScore:
    round_number: int
    winner: Side
    red_effective_striking: float
    blue_effective_striking: float
    red_effective_grappling: float
    blue_effective_grappling: float
    primary_difference: float
    criterion: str
    red_score: int
    blue_score: int


@dataclass
class RoundEvidence:
    striking: dict[str, float] = field(default_factory=lambda: {"red": 0.0, "blue": 0.0})
    grappling: dict[str, float] = field(default_factory=lambda: {"red": 0.0, "blue": 0.0})
    aggression: dict[str, float] = field(default_factory=lambda: {"red": 0.0, "blue": 0.0})
    control: dict[str, float] = field(default_factory=lambda: {"red": 0.0, "blue": 0.0})
    successes: dict[str, int] = field(default_factory=lambda: {"red": 0, "blue": 0})


@dataclass
class DeterministicJudgingModel:
    calibration: EventMCCalibration = DEFAULT_CALIBRATION
    evidence: RoundEvidence = field(default_factory=RoundEvidence)
    cards: list[RoundScore] = field(default_factory=list)

    def on_time_advance(self, dt, before, after):
        if before.clinch_controller:
            self.evidence.control[before.clinch_controller] += dt
        if before.ground_controller:
            self.evidence.control[before.ground_controller] += dt

    def on_event(self, event, before, after):
        c = self.calibration.section("judging")
        if (
            isinstance(event, PrimaryEvent)
            and isinstance(event.payload, ActionAttempt)
            and event.payload.action_family in OFFENSIVE_INITIATIVE_FAMILIES
        ):
            self.evidence.aggression[event.payload.side.value] += c["aggression_attempt_weight"]
        if not isinstance(event, ConsequenceEvent):
            return
        payload = event.payload
        if isinstance(payload, PhysiologyOutcome):
            side = payload.attacker.value
            self.evidence.striking[side] += c["impact_weight"] * payload.impact + c["knockdown_weight"] * payload.knockdown
        elif isinstance(payload, SubmissionFinishOutcome):
            self.evidence.grappling[payload.attacker.value] += c["submission_threat_weight"] * payload.finish_probability
        elif isinstance(payload, ActionOutcome):
            side = payload.side.value
            if payload.outcome in {"landed", "reversed"}:
                self.evidence.successes[side] += 1
            if "strike" in payload.action_family and payload.outcome == "landed":
                self.evidence.striking[side] += c["landed_strike_weight"]
            if "takedown" in payload.action_family and payload.outcome == "landed":
                self.evidence.grappling[side] += c["takedown_weight"]
            if payload.action_family == "ground_reversal" and payload.outcome == "reversed":
                self.evidence.grappling[side] += c["reversal_weight"]

    def score_round(self, round_number, judging_rng) -> RoundScore:
        c = self.calibration.section("judging")
        red_primary = self.evidence.striking["red"] + self.evidence.grappling["red"]
        blue_primary = self.evidence.striking["blue"] + self.evidence.grappling["blue"]
        diff = red_primary - blue_primary
        if abs(diff) > c["primary_close_band"]:
            winner, criterion = (Side.RED if diff > 0 else Side.BLUE), "PRIMARY"
        else:
            aggression = self.evidence.aggression["red"] - self.evidence.aggression["blue"]
            if abs(aggression) > c["aggression_close_band"]:
                winner, criterion = (Side.RED if aggression > 0 else Side.BLUE), "AGGRESSION"
            else:
                control = self.evidence.control["red"] - self.evidence.control["blue"]
                if abs(control) > c["control_close_band_seconds"]:
                    winner, criterion = (Side.RED if control > 0 else Side.BLUE), "CONTROL"
                else:
                    successful = self.evidence.successes["red"] - self.evidence.successes["blue"]
                    if successful:
                        winner = Side.RED if successful > 0 else Side.BLUE
                    else:
                        winner = Side.RED if judging_rng.random() < 0.5 else Side.BLUE
                    criterion = "FINAL_TIEBREAKER"
        card = RoundScore(round_number, winner, self.evidence.striking["red"], self.evidence.striking["blue"], self.evidence.grappling["red"], self.evidence.grappling["blue"], diff, criterion, 10 if winner is Side.RED else 9, 10 if winner is Side.BLUE else 9)
        self.cards.append(card)
        self.evidence = RoundEvidence()
        return card

    def decision_delta(self):
        wins = Counter(card.winner.value for card in self.cards)
        winner = "red" if wins["red"] > wins["blue"] else "blue"
        return StateDelta(finished=True, finish_reason="DEC", winner=winner, finish_method="DEC")
