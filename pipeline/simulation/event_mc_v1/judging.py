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
        "standing_strike",
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
            side = event.payload.side.value
            self.evidence.aggression[side] += c["aggression_attempt_weight"]

            # Historical decision calibration:
            # a submission attempt is itself observable scoring evidence.
            if event.payload.action_family == "submission_attempt":
                self.evidence.grappling[side] += c["submission_attempt_weight"]
        if not isinstance(event, ConsequenceEvent):
            return
        payload = event.payload
        if isinstance(payload, PhysiologyOutcome):
            # Do NOT score hidden physical impact directly.
            # Historical decision calibration supports observable knockdowns.
            side = payload.attacker.value
            self.evidence.striking[side] += c["knockdown_weight"] * payload.knockdown
        elif isinstance(payload, SubmissionFinishOutcome):
            # Submission attempts are scored when attempted above.
            # Do not additionally score hidden finish probability.
            pass
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
        # Judging V2 historical calibration.
        #
        # Approximate historical decision value, normalized to one landed
        # significant strike:
        #   landed strike      = 1.000000
        #   knockdown          = 10.080282
        #   takedown landed    = 2.021731
        #   control second     = 0.048904
        #   submission attempt = 2.854417
        #
        # Striking/grappling buckets already contain the weighted observable
        # events. Control is now continuous primary scoring evidence rather
        # than a late tiebreaker.
        red_primary = (
            self.evidence.striking["red"]
            + self.evidence.grappling["red"]
            + c["control_weight_per_second"] * self.evidence.control["red"]
        )
        blue_primary = (
            self.evidence.striking["blue"]
            + self.evidence.grappling["blue"]
            + c["control_weight_per_second"] * self.evidence.control["blue"]
        )
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
